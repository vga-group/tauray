#ifndef TAURAY_FRAME_SERVER_HH
#define TAURAY_FRAME_SERVER_HH
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
#include <atomic>

namespace tr
{

// This context renders an image and streams it over the network. It also
// accepts input events, which it stuffs into SDLs event buffer. This allows for
// remote control. Has no LF or VR support, a different scheme should be used
// for that.
class frame_server: public context
{
public:
    struct options: context::options
    {
        uvec2 size = uvec2(1280, 720);
        uint16_t port_number;

        // TODO: Compression schemes?
    };

    frame_server(const options& opt);
    frame_server(const frame_server& other) = delete;
    frame_server(frame_server&& other) = delete;
    ~frame_server();

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

private:
    void init_images();
    void deinit_images();

    // These only init events.
    void init_sdl();
    void deinit_sdl();

    static void read_image_worker(frame_server* s);
    static void streamer_worker(frame_server* s);

    options opt;

    struct per_image_data
    {
        vkm<vk::Buffer> staging_buffer;
        vkm<vk::CommandBuffer> copy_cb;
        vkm<vk::Fence> copy_fence;
        bool copy_ongoing = false;
    };

    std::vector<per_image_data> per_image;
    std::mutex image_mutex;
    std::condition_variable copy_start_cv, copy_finish_cv;
    std::vector<uint32_t> image_read_queue;

    std::mutex frame_queue_mutex;
    std::condition_variable frame_queue_cv;
    std::vector<std::vector<uint8_t>> frame_queue;

    std::atomic_bool exit_streamer;
    std::atomic_bool pause_rendering;
    std::thread image_reader_thread;
    std::thread streamer_thread;
};

}

#endif

