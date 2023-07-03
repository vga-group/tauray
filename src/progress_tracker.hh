#ifndef TAURAY_PROGRESS_TRACKER_HH
#define TAURAY_PROGRESS_TRACKER_HH

#include "device.hh"
#include <chrono>
#include <optional>
#include <condition_variable>
#include <thread>

namespace tr
{

class context;

class progress_tracker
{
public:
    progress_tracker(context* ctx);
    ~progress_tracker();

    struct options
    {
        size_t expected_frame_count;
        size_t poll_ms = 10;
    };

    void begin(options opt);
    void end();

    void set_timeline(device_id id, vk::Semaphore timeline, size_t expected_steps_per_frame);
    void erase_timeline(vk::Semaphore timeline);

private:
    void update_progress_bar(
        std::chrono::steady_clock::time_point start, float progress
    );

    context* ctx;
    options opt;
    std::optional<std::thread> poll_thread;
    std::condition_variable cv;
    bool running;

    static void poll_worker(progress_tracker* self);

    struct tracking_data
    {
        device_id id;
        vk::Semaphore timeline;
        size_t expected_steps_per_frame;
    };
    std::mutex tracking_mutex;
    std::vector<tracking_data> tracking_resources;
};

}

#endif
