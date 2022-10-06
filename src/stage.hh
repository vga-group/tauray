#ifndef TAURAY_STAGE_HH
#define TAURAY_STAGE_HH
#include "context.hh"

namespace tr
{

// Stages are steps of the entire rendering pipeline that can be considered
// separate and reusable modules.
class stage
{
public:
    enum command_buffer_strategy
    {
        COMMAND_BUFFER_PER_FRAME,
        COMMAND_BUFFER_PER_SWAPCHAIN_IMAGE,
        COMMAND_BUFFER_PER_FRAME_AND_SWAPCHAIN_IMAGE
    };

    stage(device_data& dev, command_buffer_strategy strategy = COMMAND_BUFFER_PER_FRAME);
    stage(const stage& other) = delete;
    stage(stage&& other) = delete;
    virtual ~stage() = default;

    dependency run(dependencies deps);

    size_t get_command_buffer_index(uint32_t frame_index, uint32_t swapchain_index);

protected:
    device_data* dev;
    virtual void update(uint32_t frame_index);

    vk::CommandBuffer begin_compute(bool single_use = false);
    void end_compute(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index = 0);

    vk::CommandBuffer begin_graphics(bool single_use = false);
    void end_graphics(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index = 0);

    vk::CommandBuffer begin_transfer(bool single_use = false);
    void end_transfer(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index = 0);

    void clear_commands();

private:
    vk::CommandBuffer begin_commands(vk::CommandPool pool, bool single_use);
    void end_commands(vk::CommandBuffer buf, vk::CommandPool pool, uint32_t frame_index, uint32_t swapchain_index);

    uint64_t local_frame_counter;
    std::vector<std::vector<vkm<vk::CommandBuffer>>> command_buffers;
    std::vector<vkm<vk::Semaphore>> finished;
    command_buffer_strategy strategy;
};

}

#endif


