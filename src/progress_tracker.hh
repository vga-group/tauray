#ifndef TAURAY_PROGRESS_TRACKER_HH
#define TAURAY_PROGRESS_TRACKER_HH

#include "vkm.hh"
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
        size_t expected_steps_per_frame;
        size_t poll_ms = 10;
    };

    void begin(options opt);
    void end();

    void prepare_frame(size_t device_index, uint32_t frame_index, uint64_t frame);
    void record_tracking(size_t device_index, vk::CommandBuffer cmd, uint32_t frame_index, uint32_t step);

private:
    struct tracking_stamp
    {
        uint32_t frame = 0;
        uint32_t step = 0;

        bool operator<(const tracking_stamp& other) const;
        bool operator==(const tracking_stamp& other) const;
    };

    void update_progress_bar(
        std::chrono::steady_clock::time_point start,
        const tracking_stamp& latest
    );

    context* ctx;
    options opt;
    std::optional<std::thread> poll_thread;
    std::condition_variable cv;
    bool running;

    static void poll_worker(progress_tracker* self);

    struct tracking_data
    {
        vkm<vk::Buffer> frame_buffer;
        uint32_t* frame_buffer_ptr;
        vkm<vk::Buffer> tracking_data;
        tracking_stamp* tracking_data_ptr;
        tracking_stamp last_value;
    };
    std::vector<tracking_data> tracking_resources;
};

}

#endif


