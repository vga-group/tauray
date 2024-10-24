#include "device_transfer.hh"
#include "timer.hh"
#include "misc.hh"

namespace
{
using namespace tr;

size_t get_transfer_size(const device_transfer_interface::image_transfer& t)
{
    size_t sz = t.info.extent.width*t.info.extent.height*t.info.extent.depth;
    sz *= t.bytes_per_pixel*t.info.srcSubresource.layerCount;
    return sz;
}

size_t get_transfer_size(const device_transfer_interface::buffer_transfer& t)
{
    return t.info.size;
}

struct external_semaphore_host_buffer: public device_transfer_interface
{
    device* from;
    device* to;
    timer src_to_host_timer;
    timer host_to_dst_timer;

    struct transfer_buffer
    {
        size_t capacity = 0;
        void* host_ptr = nullptr;
        vk::Buffer src_to_host;
        vk::DeviceMemory src_to_host_mem;
        vk::Buffer host_to_dst;
        vk::DeviceMemory host_to_dst_mem;
    };

    struct per_frame_data
    {
        transfer_buffer transfer;
        vkm<vk::Semaphore> src_to_host_sem;
        vkm<vk::Semaphore> src_to_host_sem_dst_copy;
        int external_sem_fd;
        vkm<vk::CommandBuffer> src_to_host_cb;
        vkm<vk::CommandBuffer> host_to_dst_cb;
    };

    per_frame_data frames[MAX_FRAMES_IN_FLIGHT];
    vkm<vk::Semaphore> host_to_dst_sem;
    uint64_t timeline;

    external_semaphore_host_buffer(device& from, device& to):
        from(&from), to(&to),
        src_to_host_timer(from, std::string("Transfer from ") + from.props.deviceName.data() + " to host"),
        host_to_dst_timer(to, std::string("Transfer from host to ") + to.props.deviceName.data()),
        timeline(0)
    {
        for(auto& f: frames)
        {
            vk::SemaphoreCreateInfo sem_info;
            vk::ExportSemaphoreCreateInfo esem_info(
                vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd
            );
            sem_info.pNext = &esem_info;
            f.src_to_host_sem = vkm(from, from.logical.createSemaphore(sem_info));
            f.external_sem_fd = from.logical.getSemaphoreFdKHR({f.src_to_host_sem});

            f.src_to_host_sem_dst_copy = create_binary_semaphore(to);
            to.logical.importSemaphoreFdKHR({
                f.src_to_host_sem_dst_copy, {},
                vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd,
                f.external_sem_fd
            });
        }
        host_to_dst_sem = create_timeline_semaphore(to);
    }

    ~external_semaphore_host_buffer()
    {
        destroy();
    }

    vk::ImageUsageFlagBits required_src_img_flags() override
    {
        return vk::ImageUsageFlagBits::eTransferSrc;
    }

    vk::ImageUsageFlagBits required_dst_img_flags() override
    {
        return vk::ImageUsageFlagBits::eTransferDst;
    }

    vk::BufferUsageFlagBits required_src_buffer_flags() override
    {
        return vk::BufferUsageFlagBits::eTransferSrc;
    }

    vk::BufferUsageFlagBits required_dst_buffer_flags() override
    {
        return vk::BufferUsageFlagBits::eTransferDst;
    }

    void reserve(
        const std::vector<image_transfer>& images,
        const std::vector<buffer_transfer>& buffers
    ) override
    {
        size_t total_transfer_memory = 0;
        for(const image_transfer& t: images)
            total_transfer_memory += get_transfer_size(t);

        for(const buffer_transfer& t: buffers)
            total_transfer_memory += get_transfer_size(t);

        bool need_realloc = false;
        for(auto& f: frames)
        {
            if(f.transfer.capacity < total_transfer_memory)
                need_realloc = true;
        }
        if(need_realloc)
            destroy();
        else return;

        for(auto& f: frames)
        {
            f.transfer.capacity = total_transfer_memory;
            f.transfer.host_ptr = allocate_host_buffer({to, from}, f.transfer.capacity);
            create_host_allocated_buffer(
                *from, f.transfer.src_to_host, f.transfer.src_to_host_mem,
                f.transfer.capacity, f.transfer.host_ptr
            );
            create_host_allocated_buffer(
                *to, f.transfer.host_to_dst, f.transfer.host_to_dst_mem,
                f.transfer.capacity, f.transfer.host_ptr
            );
        }
    }

    void build(
        const std::vector<image_transfer>& images,
        const std::vector<buffer_transfer>& buffers
    ) override
    {
        reserve(images, buffers);

        int frame_index = 0;
        for(auto& f: frames)
        {
            f.src_to_host_cb = create_graphics_command_buffer(*from);

            f.src_to_host_cb->begin(vk::CommandBufferBeginInfo{});
            src_to_host_timer.begin(f.src_to_host_cb, from->id, frame_index);

            f.host_to_dst_cb = create_graphics_command_buffer(*to);
            f.host_to_dst_cb->begin(vk::CommandBufferBeginInfo{});

            host_to_dst_timer.begin(f.host_to_dst_cb, to->id, frame_index);

            size_t offset = 0;
            for(const image_transfer& t: images)
            {
                size_t size = get_transfer_size(t);
                if(size == 0) continue;

                // SRC -> HOST
                vk::BufferImageCopy src_region(
                    offset, 0, 0,
                    t.info.srcSubresource,
                    t.info.srcOffset,
                    t.info.extent
                );

                vk::ImageMemoryBarrier img_barrier(
                    {}, vk::AccessFlagBits::eTransferRead,
                    t.src_layout,
                    vk::ImageLayout::eTransferSrcOptimal,
                    VK_QUEUE_FAMILY_IGNORED,
                    VK_QUEUE_FAMILY_IGNORED,
                    t.dst,
                    {
                        t.info.srcSubresource.aspectMask,
                        t.info.srcSubresource.mipLevel,
                        1,
                        t.info.srcSubresource.baseArrayLayer,
                        t.info.srcSubresource.layerCount
                    }
                );

                if(t.src_layout != vk::ImageLayout::eTransferSrcOptimal)
                {
                    f.host_to_dst_cb->pipelineBarrier(
                        vk::PipelineStageFlagBits::eTopOfPipe,
                        vk::PipelineStageFlagBits::eTransfer,
                        {},
                        {}, {},
                        img_barrier
                    );
                }

                f.src_to_host_cb->copyImageToBuffer(
                    t.src, vk::ImageLayout::eTransferSrcOptimal,
                    f.transfer.src_to_host, 1, &src_region
                );

                if(t.src_layout != vk::ImageLayout::eTransferSrcOptimal)
                {
                    std::swap(img_barrier.newLayout, img_barrier.oldLayout);
                    f.host_to_dst_cb->pipelineBarrier(
                        vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eBottomOfPipe,
                        {},
                        {}, {},
                        img_barrier
                    );
                }

                // HOST -> DST
                img_barrier = vk::ImageMemoryBarrier(
                    {}, vk::AccessFlagBits::eTransferWrite,
                    vk::ImageLayout::eUndefined,
                    vk::ImageLayout::eTransferDstOptimal,
                    VK_QUEUE_FAMILY_IGNORED,
                    VK_QUEUE_FAMILY_IGNORED,
                    t.dst,
                    {
                        t.info.dstSubresource.aspectMask,
                        t.info.dstSubresource.mipLevel,
                        1,
                        t.info.dstSubresource.baseArrayLayer,
                        t.info.dstSubresource.layerCount
                    }
                );

                f.host_to_dst_cb->pipelineBarrier(
                    vk::PipelineStageFlagBits::eTopOfPipe,
                    vk::PipelineStageFlagBits::eTransfer,
                    {},
                    {}, {},
                    img_barrier
                );

                vk::BufferImageCopy dst_region(
                    offset, 0, 0,
                    t.info.dstSubresource,
                    t.info.dstOffset,
                    t.info.extent
                );

                f.host_to_dst_cb->copyBufferToImage(
                    f.transfer.host_to_dst, t.dst,
                    vk::ImageLayout::eTransferDstOptimal,
                    1, &dst_region
                );

                img_barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
                img_barrier.dstAccessMask = {};
                img_barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
                img_barrier.newLayout = t.dst_layout;
                f.host_to_dst_cb->pipelineBarrier(
                    vk::PipelineStageFlagBits::eTransfer,
                    vk::PipelineStageFlagBits::eBottomOfPipe,
                    {}, {}, {}, img_barrier
                );

                offset += size;
            }

            for(const buffer_transfer& t: buffers)
            {
                if(t.info.size == 0)
                    continue;
                // SRC -> HOST
                vk::BufferCopy src_region(t.info.srcOffset, offset, t.info.size);
                f.src_to_host_cb->copyBuffer(t.src, f.transfer.src_to_host, 1, &src_region);

                // HOST -> DST
                vk::BufferCopy dst_region(offset, t.info.dstOffset, t.info.size);
                f.host_to_dst_cb->copyBuffer(f.transfer.host_to_dst, t.dst, 1, &dst_region);

                offset += get_transfer_size(t);
            }

            src_to_host_timer.end(f.src_to_host_cb, from->id, frame_index);
            f.src_to_host_cb->end();
            host_to_dst_timer.end(f.host_to_dst_cb, to->id, frame_index);
            f.host_to_dst_cb->end();
            frame_index++;
        }
    }

    void destroy()
    {
        bool synced = false;
        for(auto& f: frames)
        {
            if(!f.transfer.host_ptr) continue;
            if(!synced)
            {
                // Can't destroy the shared buffer while someone may be using it!
                from->ctx->sync();
                synced = true;
            }

            f.transfer.capacity = 0;
            f.src_to_host_cb.destroy();
            f.host_to_dst_cb.destroy();
            release_host_buffer(f.transfer.host_ptr);
            destroy_host_allocated_buffer(
                *from, f.transfer.src_to_host, f.transfer.src_to_host_mem
            );
            destroy_host_allocated_buffer(
                *to, f.transfer.host_to_dst, f.transfer.host_to_dst_mem
            );
        }
    }

    dependency run(const dependencies& deps, uint32_t frame_index) override
    {
        timeline++;
        auto& f = frames[frame_index];
        vk::TimelineSemaphoreSubmitInfo timeline_info = deps.get_timeline_info(from->id);
        vk::SubmitInfo submit_info = deps.get_submit_info(from->id, timeline_info);
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = f.src_to_host_cb.get();
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = f.src_to_host_sem.get();

        vk::PipelineStageFlags wait_stage =
            vk::PipelineStageFlagBits::eTopOfPipe;
        from->graphics_queue.submit(submit_info, {});

        timeline_info.waitSemaphoreValueCount = 0;
        timeline_info.pWaitSemaphoreValues = nullptr;
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = &timeline;
        submit_info.pWaitDstStageMask = &wait_stage;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = f.src_to_host_sem_dst_copy.get();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = f.host_to_dst_cb.get();
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = host_to_dst_sem.get();

        to->graphics_queue.submit(submit_info, {});
        return {to->id, host_to_dst_sem, timeline};
    }
};

}

namespace tr
{

std::unique_ptr<device_transfer_interface> create_device_transfer_interface(
    device& from,
    device& to,
    device_transfer_strategy /*strat*/
){
    // TODO: Actually use selected strategy
    return std::make_unique<external_semaphore_host_buffer>(from, to);
}

}
