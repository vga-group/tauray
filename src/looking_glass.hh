#ifndef TAURAY_LOOKING_GLASS_HH
#define TAURAY_LOOKING_GLASS_HH

#include "context.hh"
#include "scene.hh"
#include "camera.hh"
#include "looking_glass_composition_stage.hh"

#if _WIN32
#include <SDL.h>
#include <SDL_vulkan.h>
#else
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#endif

namespace tr
{

// I used a couple of publicly available sources to figure out how to get the
// image on the display, since Tauray doesn't use the (as of writing) closed
// beta API to do the rendering.
// Figured out what we need to know from:
//     https://github.com/zalo/Holopladertoy
// Figured out how to get the info from HoloPlay:
//     https://github.com/regcs/AliceLG/blob/master/lib/pylightio/lookingglass/services.py
// Figured out how to lay out the image from (and how to place cameras):
//    https://www.shadertoy.com/view/3tBGDR
//    https://www.shadertoy.com/view/ttXSDN
// TODO: The API would be great because it probably does things more correctly,
// this implementation has lots of guesswork. For example, we don't do the
// depth-of-field filtering that would hide some moire patterns.
class looking_glass: public context
{
public:
    struct options: context::options
    {
        const char* title = "TauRay";
        bool vsync = false;
        uvec2 viewport_size = uvec2(256, 341);
        unsigned viewport_count = 115;
        float mid_plane_dist = 2.0f;
        float depthiness = 2.0f;
        float relative_view_distance = 2.0f;

        struct calibration_data
        {
            int display_index;
            float pitch;
            float slope;
            float center;
            float fringe;
            float viewCone;
            int invView;
            float verticalAngle;
            float DPI;
            int screenW;
            int screenH;
            int flipImageX;
            int flipImageY;
            int flipSubp;
        };
        std::optional<calibration_data> calibration_override;
    };

    looking_glass(const options& opt);
    looking_glass(const looking_glass& other) = delete;
    looking_glass(looking_glass&& other) = delete;
    ~looking_glass();

    void recreate_swapchains();

    void setup_cameras(
        scene& s,
        transformable_node* reference_frame = nullptr
    );

protected:
    uint32_t prepare_next_image(uint32_t frame_index) override;
    dependencies fill_end_frame_dependencies(const dependencies& deps) override;
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
    struct device_metadata
    {
        float dpi = 0;
        float center = 0;
        std::string config_version;
        bvec2 flip_image = bvec2(false);
        bool flip_subpixel = false;
        float fringe = 0.0f;
        bool invert = false;
        float pitch = 0;
        float corrected_pitch = 0;
        float tilt = 0;
        uvec2 size = uvec2(1536, 2048);
        std::string serial;
        float slope = 0;
        float vertical_angle = 0;
        float view_cone = 0;
        std::string hardware_version;
        std::string hardware_id;
        size_t index;
        uvec2 window_coords;
    };

    void get_lkg_metadata();
    device_metadata get_lkg_device_metadata(void* lkg_device);

    void init_sdl();
    void deinit_sdl();

    void init_swapchain();
    void deinit_swapchain();

    void init_render_target();
    void deinit_render_target();

    options opt;

    device_metadata metadata;
    std::string service_version;
    SDL_Window* win;
    VkSurfaceKHR surface;
    vk::SwapchainKHR swapchain;
    std::unique_ptr<looking_glass_composition_stage> composition;
    std::vector<vkm<vk::Image>> window_images;
    std::vector<vkm<vk::ImageView>> window_image_views;
};

}

#endif
