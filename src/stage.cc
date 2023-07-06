#include "stage.hh"
#include "misc.hh"

namespace tr
{

stage::stage(device_mask dev, command_buffer_strategy strategy)
: buffers(dev), strategy(strategy)
{
    buffers([&](device& dev, cb_data& c){
        c.local_step_counter = 0;
        c.progress = create_timeline_semaphore(dev);
        switch(strategy)
        {
        case COMMAND_BUFFER_PER_FRAME:
            c.command_buffers.resize(MAX_FRAMES_IN_FLIGHT);
            break;
        case COMMAND_BUFFER_PER_SWAPCHAIN_IMAGE:
            c.command_buffers.resize(dev.ctx->get_swapchain_image_count());
            break;
        case COMMAND_BUFFER_PER_FRAME_AND_SWAPCHAIN_IMAGE:
            c.command_buffers.resize(MAX_FRAMES_IN_FLIGHT * dev.ctx->get_swapchain_image_count());
            break;
        }
    });
}

stage::~stage()
{
    buffers([&](device& dev, cb_data& c){
        dev.ctx->get_progress_tracker().erase_timeline(*c.progress);
    });
}

dependencies stage::run(dependencies deps)
{
    uint32_t swapchain_index = 0, frame_index = 0;
    buffers.get_context()->get_indices(swapchain_index, frame_index);

    update(frame_index);

    size_t cb_index = get_command_buffer_index(frame_index, swapchain_index);

    buffers([&](device& dev, cb_data& c){
        dev.ctx->get_progress_tracker().set_timeline(dev.index, c.progress, c.command_buffers[cb_index].size());

        for(size_t i = 0; i < c.command_buffers[cb_index].size(); ++i)
        {
            const vkm<vk::CommandBuffer>& cmd = c.command_buffers[cb_index][i];

            vk::TimelineSemaphoreSubmitInfo timeline_info = deps.get_timeline_info(dev.index);
            c.local_step_counter++;
            timeline_info.signalSemaphoreValueCount = 1;
            timeline_info.pSignalSemaphoreValues = &c.local_step_counter;

            vk::SubmitInfo submit_info = deps.get_submit_info(dev.index, timeline_info);
            submit_info.signalSemaphoreCount = 1;
            submit_info.pSignalSemaphores = c.progress.get();
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = cmd.get();

            if(cmd.get_pool() == dev.graphics_pool)
                dev.graphics_queue.submit(submit_info, {});
            if(cmd.get_pool() == dev.compute_pool)
                dev.compute_queue.submit(submit_info, {});
            if(cmd.get_pool() == dev.transfer_pool)
                dev.transfer_queue.submit(submit_info, {});

            deps.clear(dev.index);
            deps.add({dev.index, *c.progress, c.local_step_counter});
        }
    });

    return deps;
}

size_t stage::get_command_buffer_index(uint32_t frame_index, uint32_t swapchain_index)
{
    switch(strategy)
    {
    case COMMAND_BUFFER_PER_FRAME:
        return frame_index;
    case COMMAND_BUFFER_PER_SWAPCHAIN_IMAGE:
        return swapchain_index;
    case COMMAND_BUFFER_PER_FRAME_AND_SWAPCHAIN_IMAGE:
        return frame_index + swapchain_index * MAX_FRAMES_IN_FLIGHT;
    }
    assert(false);
    return 0;
}

void stage::update(uint32_t) { /* NO-OP by default */ }

vk::CommandBuffer stage::begin_compute(device_id id, bool single_use)
{
    return begin_commands(buffers.get_device(id).compute_pool, id, single_use);
}

void stage::end_compute(vk::CommandBuffer buf, device_id id, uint32_t frame_index, uint32_t swapchain_index)
{
    end_commands(buf, buffers.get_device(id).compute_pool, id, frame_index, swapchain_index);
}

vk::CommandBuffer stage::begin_graphics(device_id id, bool single_use)
{
    return begin_commands(buffers.get_device(id).graphics_pool, id, single_use);
}

void stage::end_graphics(vk::CommandBuffer buf, device_id id, uint32_t frame_index, uint32_t swapchain_index)
{
    end_commands(buf, buffers.get_device(id).graphics_pool, id, frame_index, swapchain_index);
}

vk::CommandBuffer stage::begin_transfer(device_id id, bool single_use)
{
    return begin_commands(buffers.get_device(id).transfer_pool, id, single_use);
}

void stage::end_transfer(vk::CommandBuffer buf, device_id id, uint32_t frame_index, uint32_t swapchain_index)
{
    end_commands(buf, buffers.get_device(id).transfer_pool, id, frame_index, swapchain_index);
}

vk::CommandBuffer stage::begin_commands(vk::CommandPool pool, device_id id, bool single_use)
{
    vk::CommandBuffer cb = buffers.get_device(id).dev.allocateCommandBuffers({
        pool, vk::CommandBufferLevel::ePrimary, 1
    })[0];

    cb.begin(vk::CommandBufferBeginInfo{
        single_use ?
            vk::CommandBufferUsageFlagBits::eOneTimeSubmit :
            vk::CommandBufferUsageFlagBits::eSimultaneousUse
    });
    return cb;
}

void stage::end_commands(vk::CommandBuffer buf, vk::CommandPool pool, device_id id, uint32_t frame_index, uint32_t swapchain_index)
{
    buf.end();
    size_t index = get_command_buffer_index(frame_index, swapchain_index);
    buffers[id].command_buffers[index].emplace_back(buffers.get_device(id), buf, pool);
}

void stage::clear_commands()
{
    buffers([&](device&, cb_data& c){
        for(auto& vec: c.command_buffers) vec.clear();
    });
}

single_device_stage::single_device_stage(device& dev, command_buffer_strategy strategy)
: stage(dev, strategy), dev(&dev)
{
}

vk::CommandBuffer single_device_stage::begin_compute(bool single_use)
{
    return stage::begin_compute(dev->index, single_use);
}

void single_device_stage::end_compute(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index)
{
    stage::end_compute(buf, dev->index, frame_index, swapchain_index);
}

vk::CommandBuffer single_device_stage::begin_graphics(bool single_use)
{
    return stage::begin_graphics(dev->index, single_use);
}

void single_device_stage::end_graphics(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index)
{
    stage::end_graphics(buf, dev->index, frame_index, swapchain_index);
}

vk::CommandBuffer single_device_stage::begin_transfer(bool single_use)
{
    return stage::begin_transfer(dev->index, single_use);
}

void single_device_stage::end_transfer(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index)
{
    stage::end_transfer(buf, dev->index, frame_index, swapchain_index);
}

}
