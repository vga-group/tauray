#ifndef TAURAY_DSHGI_SERVER_HH
#define TAURAY_DSHGI_SERVER_HH
#include "context.hh"
#include "texture.hh"
#include "scene_stage.hh"
#include "sh_renderer.hh"
#include "renderer.hh"
#include <condition_variable>
#include <thread>
#include <mutex>
#include <atomic>

namespace tr
{

class sh_grid_to_cpu_stage;
class dshgi_server: public renderer
{
public:
    struct options
    {
        sh_renderer::options sh;
        uint16_t port_number;
    };

    dshgi_server(context& ctx, const options& opt);
    dshgi_server(const dshgi_server& other) = delete;
    dshgi_server(dshgi_server&& other) = delete;
    ~dshgi_server();

    void set_scene(scene* s) override;
    void render() override;

private:
    static void sender_worker(dshgi_server* s);

    context* ctx;
    options opt;
    scene* cur_scene = nullptr;
    std::unique_ptr<scene_stage> scene_update;
    std::unique_ptr<sh_grid_to_cpu_stage> sh_grid_to_cpu;
    std::optional<sh_renderer> sh;

    std::mutex frame_queue_mutex;
    std::condition_variable frame_queue_cv;
    std::vector<dependencies> frame_queue;
    vkm<vk::Semaphore> sender_semaphore;

    std::optional<event_subscription> update_event;
    time_ticks timestamp;

    std::atomic_bool exit_sender;
    std::atomic_uint subscriber_count;
    std::thread sender_thread;
};

}

#endif
