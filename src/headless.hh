#ifndef TAURAY_HEADLESS_HH
#define TAURAY_HEADLESS_HH

#include "context.hh"

#if _WIN32
#include <SDL.h>
#include <SDL_vulkan.h>
#else
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#endif

#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>

namespace tr
{

class headless: public context
{
public:
    enum compression_type
    {
        NONE = 0,
        RLE,
        ZIPS,
        ZIP,
        PIZ
    };

    enum pixel_format
    {
        RGB16 = 0,
        RGB32,
        RGBA16,
        RGBA32
    };

    enum image_file_type
    {
        EXR = 0,
        PNG,
        BMP,
        HDR,
        RAW,
        EMPTY
    };

    struct options: context::options
    {
        uvec2 size = uvec2(1280, 720);
        std::string output_prefix = "capture";
        compression_type output_compression = PIZ;
        pixel_format output_format = RGB16;
        image_file_type output_file_type = EXR;

        // The viewer mode only exists as a workaround for Nvidia's
        // incompetence.
        bool viewer = false;
        bool viewer_fullscreen = false;

        // If display_count > 1, viewer must be false.
        unsigned display_count = 1;

        // If only a single frame will be saved, this can enable a simpler
        // naming scheme where '0' won't be appended to the name.
        bool single_frame = false;

        // If true, the NaN check will not be done. You may want to enable this
        // when NaN is expected behaviour.
        bool skip_nan_check = false;

        // If you want the first number to be something other than 0, set this
        // to that number.
        unsigned first_frame_index = 0;
    };

    headless(const options& opt);
    headless(const headless& other) = delete;
    headless(headless&& other) = delete;
    ~headless();

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
    void init_images();
    void deinit_images();

    // These are only used when opt.viewer = true
    void init_sdl();
    void deinit_sdl();

    void save_image(uint32_t swapchain_index);
    void view_image(uint32_t swapchain_index);

    void reap_workers(bool finished_only);

    options opt;
    SDL_Window* win;
    SDL_Surface* display_surface;

    struct per_image_data
    {
        vkm<vk::Buffer> staging_buffer;
        vkm<vk::CommandBuffer> copy_cb;
        vkm<vk::Fence> copy_fence;
        bool copy_ongoing = false;
        uint32_t frame_number = 0;
    };

    std::vector<per_image_data> per_image;

    struct worker
    {
        std::thread t;
        bool finished = false;
    };
    std::vector<std::unique_ptr<worker>> save_workers;
    std::condition_variable save_workers_cv;
    std::mutex save_workers_mutex;
};

}

#endif

