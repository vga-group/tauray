#ifndef TAURAY_SERVER_CONTEXT_HH
#define TAURAY_SERVER_CONTEXT_HH
#include "context.hh"

namespace tr
{

// While the regular headless context is still geared towards rendering images,
// this context doesn't do that either. It has no outputs at all. It's only
// intended to be used by resource-streaming server modes that need Vulkan but
// never produce images.
class server_context: public context
{
public:
    using options = context::options;

    server_context(const options& opt);
    server_context(const server_context& other) = delete;
    server_context(server_context&& other) = delete;
    ~server_context();

    bool init_frame() override;

protected:
    uint32_t prepare_next_image(uint32_t frame_index) override;
    void finish_image(
        uint32_t frame_index,
        uint32_t swapchain_index,
        bool display
    ) override;
    bool queue_can_present(
        const vk::PhysicalDevice& device,
        uint32_t queue_index,
        const vk::QueueFamilyProperties& props
    ) override final;
};

}

#endif
