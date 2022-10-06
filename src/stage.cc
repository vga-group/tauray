#include "stage.hh"
#include "misc.hh"

namespace tr
{

stage::stage(device_data& dev, command_buffer_strategy strategy)
: dev(&dev), local_frame_counter(0), strategy(strategy)
{
    switch(strategy)
    {
    case COMMAND_BUFFER_PER_FRAME:
        command_buffers.resize(MAX_FRAMES_IN_FLIGHT);
        break;
    case COMMAND_BUFFER_PER_SWAPCHAIN_IMAGE:
        command_buffers.resize(dev.ctx->get_swapchain_image_count());
        break;
    case COMMAND_BUFFER_PER_FRAME_AND_SWAPCHAIN_IMAGE:
        command_buffers.resize(MAX_FRAMES_IN_FLIGHT * dev.ctx->get_swapchain_image_count());
        break;
    }
}

dependency stage::run(dependencies deps)
{
    local_frame_counter++;
    dependency dep;

    uint32_t swapchain_index = 0, frame_index = 0;
    dev->ctx->get_indices(swapchain_index, frame_index);

    update(frame_index);

    size_t cb_index = get_command_buffer_index(frame_index, swapchain_index);

    for(size_t i = 0; i < command_buffers[cb_index].size(); ++i)
    {
        const vkm<vk::CommandBuffer>& cmd = command_buffers[cb_index][i];
        const vkm<vk::Semaphore>& cur = finished[i];

        if(local_frame_counter > 1 && i == 0)
            deps.add({*finished.back(), local_frame_counter-1});
        else if(i > 0)
            deps.add(dep);

        vk::TimelineSemaphoreSubmitInfo timeline_info = deps.get_timeline_info();
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = &local_frame_counter;

        vk::SubmitInfo submit_info = deps.get_submit_info(timeline_info);
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = cur.get();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = cmd.get();

        if(cmd.get_pool() == dev->graphics_pool)
            dev->graphics_queue.submit(submit_info, {});
        if(cmd.get_pool() == dev->compute_pool)
            dev->compute_queue.submit(submit_info, {});
        if(cmd.get_pool() == dev->transfer_pool)
            dev->transfer_queue.submit(submit_info, {});

        if((local_frame_counter > 1 && i == 0) || i > 0)
            deps.pop();

        dep = {*cur, local_frame_counter};
    }

    return dep;
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

vk::CommandBuffer stage::begin_compute(bool single_use)
{
    return begin_commands(dev->compute_pool, single_use);
}

void stage::end_compute(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index)
{
    end_commands(buf, dev->compute_pool, frame_index, swapchain_index);
}

vk::CommandBuffer stage::begin_graphics(bool single_use)
{
    return begin_commands(dev->graphics_pool, single_use);
}

void stage::end_graphics(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index)
{
    end_commands(buf, dev->graphics_pool, frame_index, swapchain_index);
}

vk::CommandBuffer stage::begin_transfer(bool single_use)
{
    return begin_commands(dev->transfer_pool, single_use);
}

void stage::end_transfer(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index)
{
    end_commands(buf, dev->transfer_pool, frame_index, swapchain_index);
}

vk::CommandBuffer stage::begin_commands(vk::CommandPool pool, bool single_use)
{
    vk::CommandBuffer cb = dev->dev.allocateCommandBuffers({
        pool, vk::CommandBufferLevel::ePrimary, 1
    })[0];

    cb.begin(vk::CommandBufferBeginInfo{
        single_use ?
            vk::CommandBufferUsageFlagBits::eOneTimeSubmit :
            vk::CommandBufferUsageFlagBits::eSimultaneousUse
    });
    return cb;
}

void stage::end_commands(vk::CommandBuffer buf, vk::CommandPool pool, uint32_t frame_index, uint32_t swapchain_index)
{
    buf.end();
    size_t index = get_command_buffer_index(frame_index, swapchain_index);
    command_buffers[index].emplace_back(*dev, buf, pool);
    while(finished.size() < command_buffers[index].size())
        finished.push_back(create_timeline_semaphore(*dev));
}

void stage::clear_commands()
{
    for(auto& vec: command_buffers) vec.clear();
}

}
