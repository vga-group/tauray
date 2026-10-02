#ifndef TAURAY_WINDOW_HH
#define TAURAY_WINDOW_HH

#include "context.hh"
#include "multiview_composition_stage.hh"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

namespace tr
{

class window: public context
{
public:
    struct options: context::options
    {
        const char* title = "TauRay";
        uvec2 size = uvec2(1280, 720);
        uvec2 views = uvec2(1, 1);
        bool fullscreen = false;
        bool vsync = false;
        bool hdr_display = false;
    };

    window(const options& opt);
    window(const window& other) = delete;
    window(window&& other) = delete;
    ~window();

    size_t get_swapchain_image_count() const override;
    std::vector<render_target> get_array_render_target() override;
    void recreate_swapchains();

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
    dependencies fill_end_frame_dependencies(const dependencies& deps) override;

private:
    void init_sdl();
    void deinit_sdl();

    void init_swapchain();
    void deinit_swapchain();

    options opt;

    SDL_Window* win;
    VkSurfaceKHR surface;
    vk::SwapchainKHR swapchain;

    // These are the actual swapchain images in multiview setups. In single-view
    // setups, context.images contains them.
    std::unique_ptr<multiview_composition_stage> composition;
    std::vector<vkm<vk::Image>> window_images;
    std::vector<vkm<vk::ImageView>> window_image_views;
};

}

#endif

