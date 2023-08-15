#ifndef TAURAY_OPENXR_HH
#define TAURAY_OPENXR_HH

#include "context.hh"
#include "camera.hh"
#include "scene.hh"

#if _WIN32
#include <SDL.h>
#include <SDL_vulkan.h>
#define XR_USE_PLATFORM_WIN32
#include "windows.h"
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#include "unknwn.h"
#else
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#define XR_USE_PLATFORM_XLIB
#define XR_USE_PLATFORM_WAYLAND
#endif

#define XR_USE_GRAPHICS_API_VULKAN
//#define XR_EXTENSION_PROTOTYPES
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

namespace tr
{

// TODO: This is currently just a clone of tr::window with hacky fake-VR.
// Make it actually use OpenXR.
class openxr: public context
{
public:
    struct options: context::options
    {
        const char* title = "TauRay";
        uvec2 size = uvec2(1280, 720);
        bool fullscreen = false;
        bool hdr_display = false;
    };

    openxr(const options& opt);
    openxr(const openxr& other) = delete;
    openxr(openxr&& other) = delete;
    ~openxr();

    bool init_frame() override;

    // Places cameras and controllers under the given reference frame.
    void setup_xr_surroundings(
        scene& s,
        transformable_node* reference_frame = nullptr
    );

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

    vk::Instance create_instance(
        const vk::InstanceCreateInfo& info,
        PFN_vkGetInstanceProcAddr getInstanceProcAddr
    ) override final;

    vk::Device create_device(
        const vk::PhysicalDevice& device,
        const vk::DeviceCreateInfo& info
    ) override final;

private:
    void init_sdl();
    void deinit_sdl();

    void init_xr();
    void deinit_xr();

    void init_session();
    void deinit_session();

    void init_xr_swapchain();
    void deinit_xr_swapchain();

    void init_window_swapchain();
    void deinit_window_swapchain();

    void init_local_resources();
    void deinit_local_resources();

    void blit_images(uint32_t frame_index, uint32_t swapchain_index);

    bool poll();
    void update_xr_views();

    VkPhysicalDevice get_xr_device();

    options opt;

    // Resources for the XR session
    XrInstance xr_instance;
    XrDebugUtilsMessengerEXT messenger;
    XrSystemId system_id;
    XrViewConfigurationType view_config;
    std::vector<XrView> view_states;
    VkPhysicalDevice xr_device;
    XrSession xr_session;
    XrReferenceSpaceType reference_space_type;
    XrSpace xr_reference_space;
    XrSwapchain xr_swapchain;
    XrFrameState frame_state;
    XrSessionState session_state;
    XrCompositionLayerProjection projection_layer;
    std::vector<XrCompositionLayerProjectionView> projection_layer_views;
    std::vector<XrCompositionLayerBaseHeader*> projection_layer_headers;
    vkm<vk::Fence> finish_fence;

    std::vector<vkm<vk::Image>> xr_images;
    std::vector<vkm<vk::ImageView>> xr_image_views;

    // Resources for the preview window
    SDL_Window* win;
    VkSurfaceKHR surface;
    vk::SwapchainKHR window_swapchain;
    vk::Format window_image_format;
    uint32_t window_swapchain_index;

    std::vector<vkm<vk::Image>> window_images;
    std::vector<vkm<vk::ImageView>> window_image_views;
    std::vector<vkm<vk::Semaphore>> window_frame_available;
    std::vector<vkm<vk::Semaphore>> window_frame_finished;
    std::vector<camera*> cameras;
};

}

#endif


