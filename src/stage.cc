#include "stage.hh"
#include "misc.hh"

namespace tr
{

multi_device_stage::multi_device_stage(device_mask dev, command_buffer_strategy strategy)
: buffers(dev), strategy(strategy)
{
    for(auto[dev, c]: buffers)
    {
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
    }
}

multi_device_stage::~multi_device_stage()
{
    for(auto[dev, c]: buffers)
        dev.ctx->get_progress_tracker().erase_timeline(*c.progress);
}

dependencies multi_device_stage::run(dependencies deps)
{
    uint32_t swapchain_index = 0, frame_index = 0;
    buffers.get_context()->get_indices(swapchain_index, frame_index);

    update(frame_index);

    size_t cb_index = get_command_buffer_index(frame_index, swapchain_index);

    for(auto[dev, c]: buffers)
    {
        dev.ctx->get_progress_tracker().set_timeline(dev.id, c.progress, c.command_buffers[cb_index].size());

        for(size_t i = 0; i < c.command_buffers[cb_index].size(); ++i)
        {
            const vkm<vk::CommandBuffer>& cmd = c.command_buffers[cb_index][i];

            vk::TimelineSemaphoreSubmitInfo timeline_info = deps.get_timeline_info(dev.id);
            c.local_step_counter++;
            timeline_info.signalSemaphoreValueCount = 1;
            timeline_info.pSignalSemaphoreValues = &c.local_step_counter;

            vk::SubmitInfo submit_info = deps.get_submit_info(dev.id, timeline_info);
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

            deps.clear(dev.id);
            deps.add({dev.id, *c.progress, c.local_step_counter});
        }
    }

    return deps;
}

size_t multi_device_stage::get_command_buffer_index(uint32_t frame_index, uint32_t swapchain_index)
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

void multi_device_stage::update(uint32_t) { /* NO-OP by default */ }

device_mask multi_device_stage::get_device_mask() const
{
    return buffers.get_mask();
}

context* multi_device_stage::get_context() const
{
    return buffers.get_context();
}

vk::CommandBuffer multi_device_stage::begin_compute(device_id id, bool single_use)
{
    return begin_commands(buffers.get_device(id).compute_pool, id, single_use);
}

void multi_device_stage::end_compute(vk::CommandBuffer buf, device_id id, uint32_t frame_index, uint32_t swapchain_index)
{
    end_commands(buf, buffers.get_device(id).compute_pool, id, frame_index, swapchain_index);
}

vk::CommandBuffer multi_device_stage::begin_graphics(device_id id, bool single_use)
{
    return begin_commands(buffers.get_device(id).graphics_pool, id, single_use);
}

void multi_device_stage::end_graphics(vk::CommandBuffer buf, device_id id, uint32_t frame_index, uint32_t swapchain_index)
{
    end_commands(buf, buffers.get_device(id).graphics_pool, id, frame_index, swapchain_index);
}

vk::CommandBuffer multi_device_stage::begin_transfer(device_id id, bool single_use)
{
    return begin_commands(buffers.get_device(id).transfer_pool, id, single_use);
}

void multi_device_stage::end_transfer(vk::CommandBuffer buf, device_id id, uint32_t frame_index, uint32_t swapchain_index)
{
    end_commands(buf, buffers.get_device(id).transfer_pool, id, frame_index, swapchain_index);
}

vk::CommandBuffer multi_device_stage::begin_commands(vk::CommandPool pool, device_id id, bool single_use)
{
    vk::CommandBuffer cb = buffers.get_device(id).logical.allocateCommandBuffers({
        pool, vk::CommandBufferLevel::ePrimary, 1
    })[0];

    cb.begin(vk::CommandBufferBeginInfo{
        single_use ?
            vk::CommandBufferUsageFlagBits::eOneTimeSubmit :
            vk::CommandBufferUsageFlagBits::eSimultaneousUse
    });
    return cb;
}

void multi_device_stage::end_commands(vk::CommandBuffer buf, vk::CommandPool pool, device_id id, uint32_t frame_index, uint32_t swapchain_index)
{
    buf.end();
    size_t index = get_command_buffer_index(frame_index, swapchain_index);
    buffers[id].command_buffers[index].emplace_back(buffers.get_device(id), buf, pool);
}

void multi_device_stage::clear_commands()
{
    for(auto[d, c]: buffers)
        for(auto& vec: c.command_buffers)
            vec.clear();
}

single_device_stage::single_device_stage(device& dev, command_buffer_strategy strategy)
: multi_device_stage(dev, strategy), dev(&dev)
{
}

vk::CommandBuffer single_device_stage::begin_compute(bool single_use)
{
    return multi_device_stage::begin_compute(dev->id, single_use);
}

void single_device_stage::end_compute(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index)
{
    multi_device_stage::end_compute(buf, dev->id, frame_index, swapchain_index);
}

vk::CommandBuffer single_device_stage::begin_graphics(bool single_use)
{
    return multi_device_stage::begin_graphics(dev->id, single_use);
}

void single_device_stage::end_graphics(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index)
{
    multi_device_stage::end_graphics(buf, dev->id, frame_index, swapchain_index);
}

vk::CommandBuffer single_device_stage::begin_transfer(bool single_use)
{
    return multi_device_stage::begin_transfer(dev->id, single_use);
}

void single_device_stage::end_transfer(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index)
{
    multi_device_stage::end_transfer(buf, dev->id, frame_index, swapchain_index);
}

}
