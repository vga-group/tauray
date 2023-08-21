#include "progress_tracker.hh"
#include "context.hh"
#include "misc.hh"
#include <mutex>
#include <iostream>
#include <iomanip>
#include <string>

namespace tr
{

progress_tracker::progress_tracker(context* ctx)
: ctx(ctx), running(false)
{

}

progress_tracker::~progress_tracker()
{
    end();
}

void progress_tracker::begin(options opt)
{
    end();
    running = true;
    this->opt = opt;
    poll_thread.emplace(poll_worker, this);
}

void progress_tracker::end()
{
    if(!running) return;

    {
        std::unique_lock<std::mutex> lk(tracking_mutex);
        tracking_resources.clear();
        running = false;
    }
    cv.notify_all();
    poll_thread->join();
    // Show cursor again
    std::cout << "\x1b[?25h";
}

void progress_tracker::set_timeline(device_id id, vk::Semaphore timeline, size_t expected_steps_per_frame)
{
    if(!running) return;

    std::unique_lock<std::mutex> lk(tracking_mutex);
    for(tracking_data& d: tracking_resources)
    {
        if(d.timeline == timeline)
        {
            d.expected_steps_per_frame = expected_steps_per_frame;
            return;
        }
    }
    tracking_resources.push_back({
        id,
        timeline,
        expected_steps_per_frame
    });
}

void progress_tracker::erase_timeline(vk::Semaphore timeline)
{
    if(!running) return;

    for(auto it = tracking_resources.begin(); it != tracking_resources.end();)
    {
        if(it->timeline == timeline) it = tracking_resources.erase(it);
        else ++it;
    }
}

void progress_tracker::update_progress_bar(
    std::chrono::steady_clock::time_point start,
    float progress
){
    auto delta = std::chrono::duration_cast<std::chrono::duration<float>>(
        std::chrono::steady_clock::now() - start).count();

    float total_time = delta / progress;
    int time_left = ceil(total_time - delta);

    int bar_width = 80-2;
    int fill_width = bar_width * progress;
    std::cout << "\r\x1b[?25l\x1b[2K[";
    for(int i = 0; i < bar_width; ++i)
    {
        char c = ' ';
        if(i < fill_width || progress >= 1.0f) c = '=';
        else if(i == fill_width) c = '>';

        std::cout << c;
    }
    std::cout << "] ";

    std::cout << std::setprecision(1) << 100.0f * progress << "%";

    if(progress != 0.0f)
    {
        std::cout << ", ";
        int hours = time_left / 3600;
        if(hours > 0) std::cout << hours << "h ";
        time_left %= 3600;

        int minutes = time_left / 60;
        if(minutes > 0) std::cout << minutes << "m ";
        time_left %= 60;

        std::cout << time_left << "s left";
    }

    std::cout << std::flush;
}

void progress_tracker::poll_worker(progress_tracker* self)
{
    std::vector<device>& devices = self->ctx->get_devices();

    std::chrono::steady_clock::time_point start_time;
    bool first = true;
    float last_progress = -1.0f;

    std::vector<size_t> device_total_steps(devices.size(), 0);
    std::vector<size_t> device_finished_steps(devices.size(), 0);

    for(;;)
    {
        std::unique_lock<std::mutex> lk(self->tracking_mutex);
        self->cv.wait_for(lk, std::chrono::milliseconds(self->opt.poll_ms));
        if(!self->running)
            return;

        device_total_steps.assign(devices.size(), 0);
        device_finished_steps.assign(devices.size(), 0);
        for(tracking_data& d: self->tracking_resources)
        {
            size_t steps = d.expected_steps_per_frame * self->opt.expected_frame_count;
            device_total_steps[d.id] += steps;
            uint64_t finished = devices[d.id].logical.getSemaphoreCounterValue(d.timeline);
            device_finished_steps[d.id] += finished;
        }

        float progress = 10.0f;
        for(size_t i = 0; i < devices.size(); ++i)
        {
            if(device_total_steps[i] == 0) continue;

            float device_progress = float(device_finished_steps[i]) / float(device_total_steps[i]);
            if(device_progress < progress)
                progress = device_progress;
        }

        if(last_progress != progress && progress <= 1.0f)
        {
            last_progress = progress;
            if(first)
            {
                start_time = std::chrono::steady_clock::now();
                first = false;
            }
            self->update_progress_bar(start_time, progress);
        }
    }
}

}
