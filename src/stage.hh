#ifndef TAURAY_STAGE_HH
#define TAURAY_STAGE_HH
#include "context.hh"

namespace tr
{

// Stages are steps of the entire rendering pipeline that can be considered
// separate and reusable modules.
class multi_device_stage
{
public:
    enum command_buffer_strategy
    {
        COMMAND_BUFFER_PER_FRAME,
        COMMAND_BUFFER_PER_SWAPCHAIN_IMAGE,
        COMMAND_BUFFER_PER_FRAME_AND_SWAPCHAIN_IMAGE
    };

    multi_device_stage(device_mask dev, command_buffer_strategy strategy = COMMAND_BUFFER_PER_FRAME);
    multi_device_stage(const multi_device_stage& other) = delete;
    multi_device_stage(multi_device_stage&& other) = delete;
    virtual ~multi_device_stage();

    dependencies run(dependencies deps);

    size_t get_command_buffer_index(uint32_t frame_index, uint32_t swapchain_index);

protected:
    virtual void update(uint32_t frame_index);
    device_mask get_device_mask() const;
    context* get_context() const;

    vk::CommandBuffer begin_compute(device_id id, bool single_use = false);
    void end_compute(vk::CommandBuffer buf, device_id id, uint32_t frame_index, uint32_t swapchain_index = 0);

    vk::CommandBuffer begin_graphics(device_id id, bool single_use = false);
    void end_graphics(vk::CommandBuffer buf, device_id id, uint32_t frame_index, uint32_t swapchain_index = 0);

    vk::CommandBuffer begin_transfer(device_id id, bool single_use = false);
    void end_transfer(vk::CommandBuffer buf, device_id id, uint32_t frame_index, uint32_t swapchain_index = 0);

    void clear_commands();

private:
    vk::CommandBuffer begin_commands(vk::CommandPool pool, device_id id, bool single_use);
    void end_commands(vk::CommandBuffer buf, vk::CommandPool pool, device_id id, uint32_t frame_index, uint32_t swapchain_index);

    struct cb_data
    {
        std::vector<std::vector<vkm<vk::CommandBuffer>>> command_buffers;
        uint64_t local_step_counter = 0;
        vkm<vk::Semaphore> progress;
    };
    per_device<cb_data> buffers;
    command_buffer_strategy strategy;
};

// Many stages can only take one device at a time. They should derive from this
// class. This simplifies their implementation considerably.
class single_device_stage: public multi_device_stage
{
public:
    single_device_stage(device& dev, command_buffer_strategy strategy = COMMAND_BUFFER_PER_FRAME);

protected:
    device* dev;

    vk::CommandBuffer begin_compute(bool single_use = false);
    void end_compute(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index = 0);

    vk::CommandBuffer begin_graphics(bool single_use = false);
    void end_graphics(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index = 0);

    vk::CommandBuffer begin_transfer(bool single_use = false);
    void end_transfer(vk::CommandBuffer buf, uint32_t frame_index, uint32_t swapchain_index = 0);
private:
};

}

#endif


