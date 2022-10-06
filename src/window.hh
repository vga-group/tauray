#ifndef TAURAY_WINDOW_HH
#define TAURAY_WINDOW_HH

#include "context.hh"

#if _WIN32
#include <SDL.h>
#include <SDL_vulkan.h>
#else
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#endif

namespace tr
{

class window: public context
{
public:
    struct options: context::options
    {
        const char* title = "TauRay";
        uvec2 size = uvec2(1280, 720);
        bool fullscreen = false;
        bool vsync = false;
        bool hdr_display = false;
    };

    window(const options& opt);
    window(const window& other) = delete;
    window(window&& other) = delete;
    ~window();

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

private:
    void init_sdl();
    void deinit_sdl();

    void init_swapchain();
    void deinit_swapchain();

    options opt;

    SDL_Window* win;
    VkSurfaceKHR surface;
    vk::SwapchainKHR swapchain;
};

}

#endif

