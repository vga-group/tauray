#include "misc.hh"
#include "context.hh"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <iostream>
namespace fs = std::filesystem;
#ifdef _WIN32
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define free          _aligned_free
#endif

namespace tr
{

vk::CommandBuffer begin_command_buffer(device_data& d)
{
    vk::CommandBuffer cb = d.dev.allocateCommandBuffers({
        d.graphics_pool, vk::CommandBufferLevel::ePrimary, 1
    })[0];

    cb.begin(vk::CommandBufferBeginInfo{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    });
    return cb;
}

void end_command_buffer(device_data& d, vk::CommandBuffer cb)
{
    cb.end();

    d.graphics_queue.submit(
        vk::SubmitInfo(0, nullptr, nullptr, 1, &cb, 0, nullptr), {}
    );
    d.graphics_queue.waitIdle();

    d.dev.freeCommandBuffers(d.graphics_pool, cb);
}

vkm<vk::CommandBuffer> create_compute_command_buffer(device_data& d)
{
    return vkm<vk::CommandBuffer>(d,
        d.dev.allocateCommandBuffers({
            d.compute_pool, vk::CommandBufferLevel::ePrimary, 1
        })[0],
        d.compute_pool
    );
}

vkm<vk::CommandBuffer> create_graphics_command_buffer(device_data& d)
{
    return vkm<vk::CommandBuffer>(d,
        d.dev.allocateCommandBuffers({
            d.graphics_pool, vk::CommandBufferLevel::ePrimary, 1
        })[0],
        d.graphics_pool
    );
}

vkm<vk::CommandBuffer> create_transfer_command_buffer(device_data& d)
{
    return vkm<vk::CommandBuffer>(d,
        d.dev.allocateCommandBuffers({
            d.transfer_pool, vk::CommandBufferLevel::ePrimary, 1
        })[0],
        d.transfer_pool
    );
}

vkm<vk::Semaphore> create_binary_semaphore(device_data& d)
{
    return vkm<vk::Semaphore>(
        d, d.dev.createSemaphore(vk::SemaphoreCreateInfo{})
    );
}

vkm<vk::Semaphore> create_timeline_semaphore(device_data& d)
{
    vk::SemaphoreTypeCreateInfo type(vk::SemaphoreType::eTimeline, 0);
    vk::SemaphoreCreateInfo info;
    info.pNext = &type;
    return vkm<vk::Semaphore>(d, d.dev.createSemaphore(info));
}

void transition_image_layout(
    vk::CommandBuffer cb,
    vk::Image img,
    vk::Format fmt,
    vk::ImageLayout src_layout,
    vk::ImageLayout dst_layout,
    uint32_t mip_level,
    uint32_t mip_count,
    uint32_t base_layer,
    uint32_t layer_count,
    bool ignore_src_stage_mask,
    bool ignore_dst_stage_mask
){
    if(dst_layout == src_layout)
        return;

    vk::ImageMemoryBarrier img_barrier(
        {}, {},
        src_layout,
        dst_layout,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        img,
        {
            deduce_aspect_mask(fmt),
            mip_level, mip_count, base_layer, layer_count
        }
    );

    vk::PipelineStageFlags src_stage = {};
    vk::PipelineStageFlags dst_stage = {};
    deduce_layout_access_stage(
        src_layout, img_barrier.srcAccessMask, src_stage
    );
    deduce_layout_access_stage(
        dst_layout, img_barrier.dstAccessMask, dst_stage
    );

    if(ignore_src_stage_mask)
    {
        img_barrier.srcAccessMask = {};
        src_stage = vk::PipelineStageFlagBits::eTopOfPipe;
    }
    if(ignore_dst_stage_mask)
    {
        img_barrier.dstAccessMask = {};
        dst_stage = vk::PipelineStageFlagBits::eTopOfPipe;
    }

    cb.pipelineBarrier(
        src_stage, dst_stage,
        {},
        {}, {},
        img_barrier
    );
}

vkm<vk::Buffer> create_buffer(
    device_data& dev,
    vk::BufferCreateInfo info,
    VmaAllocationCreateFlagBits flags,
    const void* data,
    vk::CommandBuffer shared_cb
){
    vk::Buffer res;
    VmaAllocation alloc;
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = flags;
    if(data)
        info.usage |= vk::BufferUsageFlagBits::eTransferDst;

    vmaCreateBuffer(
        dev.allocator, (VkBufferCreateInfo*)&info,
        &alloc_info, reinterpret_cast<VkBuffer*>(&res),
        &alloc, nullptr
    );

    if(data)
    {
        vkm<vk::Buffer> staging = create_staging_buffer(dev, info.size, data);
        vk::CommandBuffer cb = shared_cb ? shared_cb : begin_command_buffer(dev);
        cb.copyBuffer(staging, res, {{0, 0, info.size}});
        if(!shared_cb)
        {
            end_command_buffer(dev, cb);
            staging.destroy();
        }
    }
    return vkm<vk::Buffer>(dev, res, alloc);
}

vkm<vk::Buffer> create_buffer_aligned(
    device_data& dev,
    vk::BufferCreateInfo info,
    VmaAllocationCreateFlagBits flags,
    size_t alignment,
    const void* data
){
    vk::Buffer res;
    VmaAllocation alloc;
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = flags;
    if(data)
        info.usage |= vk::BufferUsageFlagBits::eTransferDst;

    vmaCreateBufferWithAlignment(
        dev.allocator, (VkBufferCreateInfo*)&info,
        &alloc_info, alignment, reinterpret_cast<VkBuffer*>(&res),
        &alloc, nullptr
    );

    if(data)
    {
        vkm<vk::Buffer> staging = create_staging_buffer(dev, info.size, data);
        vk::CommandBuffer cb = begin_command_buffer(dev);
        cb.copyBuffer(staging, res, {{0, 0, info.size}});
        end_command_buffer(dev, cb);
        staging.destroy();
    }
    return vkm<vk::Buffer>(dev, res, alloc);
}

vkm<vk::Buffer> create_staging_buffer(
    device_data& dev,
    size_t size,
    const void* data
){
    vk::Buffer res;
    VmaAllocation alloc;
    vk::BufferCreateInfo staging_info(
        {}, size, vk::BufferUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive
    );
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    vmaCreateBuffer(
        dev.allocator, (VkBufferCreateInfo*)&staging_info,
        &alloc_info, reinterpret_cast<VkBuffer*>(&res),
        &alloc, nullptr
    );

    if(data)
    {
        void* mem = nullptr;
        vmaMapMemory(dev.allocator, alloc, &mem);
        memcpy(mem, data, size);
        vmaUnmapMemory(dev.allocator, alloc);
    }
    return vkm<vk::Buffer>(dev, res, alloc);
}

vkm<vk::Buffer> create_download_buffer(
    device_data& dev,
    size_t size
){
    vk::Buffer res;
    VmaAllocation alloc;
    vk::BufferCreateInfo staging_info(
        {}, size, vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer,
        vk::SharingMode::eExclusive
    );
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    vmaCreateBuffer(
        dev.allocator, (VkBufferCreateInfo*)&staging_info,
        &alloc_info, reinterpret_cast<VkBuffer*>(&res),
        &alloc, nullptr
    );
    return vkm<vk::Buffer>(dev, res, alloc);
}

void* allocate_host_buffer(
    const std::vector<device_data*>& supported_devices,
    size_t size
){
    vk::DeviceSize alignment = 16;
    for(device_data* dev: supported_devices)
        alignment = max(
            dev->ext_mem_props.minImportedHostPointerAlignment,
            alignment
        );

    return aligned_alloc(alignment, (size+alignment-1)/alignment*alignment);
}

void release_host_buffer(void* host_buffer)
{
    free(host_buffer);
}

void create_host_allocated_buffer(
    device_data& dev,
    vk::Buffer& res,
    vk::DeviceMemory& mem,
    size_t size,
    void* data
){
    vk::BufferCreateInfo staging_info(
        {}, size,
        vk::BufferUsageFlagBits::eTransferSrc |
        vk::BufferUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive
    );
    vk::ExternalMemoryBufferCreateInfo ext_info(
        vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT
    );
    staging_info.pNext = &ext_info;
    res = dev.dev.createBuffer(staging_info);

    vk::MemoryHostPointerPropertiesEXT host_ptr_props =
        dev.dev.getMemoryHostPointerPropertiesEXT(
            vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, data
        );

    uint32_t memory_type_index = 0;
    vk::PhysicalDeviceMemoryProperties mem_props =
        dev.pdev.getMemoryProperties();
    for(uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
    {
        if(host_ptr_props.memoryTypeBits & (1 << i))
        {
            memory_type_index = i;
            break;
        }
    }

    vk::MemoryAllocateInfo alloc_info(
        size, memory_type_index
    );
    vk::ImportMemoryHostPointerInfoEXT host_ptr_info(
        vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, data
    );
    alloc_info.pNext = &host_ptr_info;
    mem = dev.dev.allocateMemory(alloc_info);
    dev.dev.bindBufferMemory(res, mem, 0);
}

void destroy_host_allocated_buffer(
    device_data& dev, vk::Buffer& res, vk::DeviceMemory& mem
){
    dev.dev.destroyBuffer(res);
    dev.dev.freeMemory(mem);
}

vk::ImageAspectFlags deduce_aspect_mask(vk::Format fmt)
{
    if(
        fmt == vk::Format::eD16Unorm ||
        fmt == vk::Format::eD32Sfloat
    ){
        return vk::ImageAspectFlagBits::eDepth;
    }
    else if(
        fmt == vk::Format::eD16UnormS8Uint ||
        fmt == vk::Format::eD24UnormS8Uint ||
        fmt == vk::Format::eD32SfloatS8Uint
    ){
        return vk::ImageAspectFlagBits::eDepth |
            vk::ImageAspectFlagBits::eStencil;
    }
    else return vk::ImageAspectFlagBits::eColor;
}

void deduce_layout_access_stage(
    vk::ImageLayout layout,
    vk::AccessFlags& access,
    vk::PipelineStageFlags& stage
){
    switch(layout)
    {
    case vk::ImageLayout::eUndefined:
    case vk::ImageLayout::ePresentSrcKHR:
        access = {};
        stage = vk::PipelineStageFlagBits::eTopOfPipe;
        break;
    case vk::ImageLayout::eTransferDstOptimal:
        access = vk::AccessFlagBits::eTransferWrite;
        stage = vk::PipelineStageFlagBits::eTransfer;
        break;
    case vk::ImageLayout::eTransferSrcOptimal:
        access = vk::AccessFlagBits::eTransferRead;
        stage = vk::PipelineStageFlagBits::eTransfer;
        break;
    case vk::ImageLayout::eShaderReadOnlyOptimal:
        access = vk::AccessFlagBits::eShaderRead;
        stage = vk::PipelineStageFlagBits::eFragmentShader;
        break;
    case vk::ImageLayout::eColorAttachmentOptimal:
        access =
            vk::AccessFlagBits::eColorAttachmentRead |
            vk::AccessFlagBits::eColorAttachmentWrite;
        stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        break;
    case vk::ImageLayout::eDepthAttachmentOptimal:
    case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        access =
            vk::AccessFlagBits::eDepthStencilAttachmentRead |
            vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        stage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
        break;
    case vk::ImageLayout::eGeneral:
        access = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
        stage = vk::PipelineStageFlagBits::eTopOfPipe;
        break;
    default:
        throw std::runtime_error("Unknown layout " + std::to_string((uint64_t)layout));
    }
}

vkm<vk::Image> sync_create_gpu_image(
    device_data& dev,
    vk::ImageCreateInfo info,
    vk::ImageLayout final_layout,
    size_t data_size,
    void* data
){
    vk::Image img;
    VmaAllocation alloc;
    if(!data)
    {
        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

        vmaCreateImage(
            dev.allocator, (VkImageCreateInfo*)&info,
            &alloc_info, reinterpret_cast<VkImage*>(&img),
            &alloc, nullptr
        );

        vk::CommandBuffer cb = begin_command_buffer(dev);
        transition_image_layout(
            cb, img, info.format, vk::ImageLayout::eUndefined, final_layout
        );
        end_command_buffer(dev, cb);
    }
    else
    {
        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        info.usage |= vk::ImageUsageFlagBits::eTransferDst |
            vk::ImageUsageFlagBits::eTransferSrc;

        VmaAllocationInfo vma_alloc_info;
        vmaCreateImage(
            dev.allocator, (VkImageCreateInfo*)&info,
            &alloc_info, reinterpret_cast<VkImage*>(&img),
            &alloc, &vma_alloc_info
        );

        vkm<vk::Buffer> staging_buffer = create_staging_buffer(
            dev, data_size, data
        );

        vk::CommandBuffer cb = begin_command_buffer(dev);
        transition_image_layout(
            cb, img, info.format,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal,
            0, info.mipLevels
        );

        vk::BufferImageCopy region(
            0, 0, 0,
            {deduce_aspect_mask(info.format), 0, 0, 1},
            {0,0,0},
            info.extent
        );
        cb.copyBufferToImage(
            staging_buffer, img, vk::ImageLayout::eTransferDstOptimal, 1, &region
        );

        // Generate mipmaps.
        ivec2 sz = ivec2(info.extent.width, info.extent.height);
        for(uint32_t i = 1; i < info.mipLevels; ++i)
        {
            transition_image_layout(
                cb, img, info.format,
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eTransferSrcOptimal,
                i-1, 1
            );
            ivec2 next_sz = max(sz/2, ivec2(1));
            vk::ImageAspectFlags mask = deduce_aspect_mask(info.format);
            vk::ImageBlit blit(
                {mask, i-1, 0, 1},
                {{{0,0,0}, {sz.x,sz.y,1}}},
                {mask, i, 0, 1},
                {{{0,0,0}, {next_sz.x,next_sz.y,1}}}
            );
            cb.blitImage(
                img, vk::ImageLayout::eTransferSrcOptimal,
                img, vk::ImageLayout::eTransferDstOptimal,
                blit, vk::Filter::eLinear
            );
            sz = next_sz;
            transition_image_layout(
                cb, img, info.format,
                vk::ImageLayout::eTransferSrcOptimal,
                final_layout,
                i-1, 1
            );
        }

        transition_image_layout(
            cb, img, info.format,
            vk::ImageLayout::eTransferDstOptimal,
            final_layout,
            info.mipLevels-1, 1
        );

        end_command_buffer(dev, cb);

        staging_buffer.destroy();
    }
    return vkm<vk::Image>(dev, img, alloc);
}

void full_barrier(vk::CommandBuffer cb)
{
    vk::MemoryBarrier barrier(
        vk::AccessFlagBits::eMemoryRead|vk::AccessFlagBits::eMemoryWrite,
        vk::AccessFlagBits::eMemoryRead|vk::AccessFlagBits::eMemoryWrite
    );

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eAllCommands,
        {}, barrier, {}, {}
    );
}

vk::SampleCountFlagBits get_max_available_sample_count(context& ctx)
{
    vk::SampleCountFlags mask = (vk::SampleCountFlags)0xFFFFFFFF;

    for(device_data& dev: ctx.get_devices())
    {
        mask &= dev.props.limits.storageImageSampleCounts;
        mask &= dev.props.limits.sampledImageColorSampleCounts;
        mask &= dev.props.limits.sampledImageDepthSampleCounts;
        mask &= dev.props.limits.sampledImageStencilSampleCounts;
    }

    if(mask & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
    if(mask & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
    if(mask & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
    if(mask & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
    if(mask & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
    if(mask & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

    return vk::SampleCountFlagBits::e1;
}

std::string get_resource_path(const std::string& path)
{
    std::string resource_path = (fs::path(TR_RESOURCE_PATH)/path).string();
#if !defined(TR_DISABLE_LOCAL_PATH) && defined(TR_RESOURCE_PATH)
    if(fs::exists(path)) return path;
    else if(fs::exists(resource_path)) return resource_path;
    else throw std::runtime_error("Could not find resource " + path);
#elif defined(TR_DISABLE_LOCAL_PATH)
    return resource_path;
#else
    return path;
#endif
}

std::string load_text_file(const std::string& path)
{
    FILE* f = fopen(path.c_str(), "rb");

    if(!f) throw std::runtime_error("Unable to open " + path);

    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* data = new char[sz];
    if(fread(data, 1, sz, f) != sz)
    {
        fclose(f);
        delete [] data;
        throw std::runtime_error("Unable to read " + path);
    }
    fclose(f);
    std::string ret(data, sz);

    delete [] data;
    return ret;
}

bool nonblock_getline(std::string& line)
{
    static std::stringstream reading;
    line.clear();

    char c;
    while(std::cin.readsome(&c, 1))
    {
        if(c == '\n')
        {
            line = reading.str();
            reading.str("");
            return true;
        }
        reading << c;
    }
    return false;
}

std::string to_uppercase(const std::string& str)
{
    std::string ret(str);
    std::transform(ret.begin(), ret.end(), ret.begin(), ::toupper);
    return ret;
}


static std::vector<std::chrono::high_resolution_clock::time_point> profile_begin;
void profile_tick()
{
    profile_begin.push_back(std::chrono::high_resolution_clock::now());
}

void profile_tock(const char* message)
{
    auto begin = profile_begin.back();
    profile_begin.pop_back();
    auto profile_end = std::chrono::high_resolution_clock::now();
    auto duration = profile_end - begin;
    std::cout << message << std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() << std::endl;
}

}
