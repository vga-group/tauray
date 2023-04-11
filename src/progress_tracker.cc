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

    std::vector<device_data>& devices = ctx->get_devices();
    tracking_resources.resize(devices.size());

    for(size_t i = 0; i < devices.size(); ++i)
    {
        vk::BufferCreateInfo create_info;
        create_info.size = sizeof(tracking_stamp);
        create_info.usage =
            vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eTransferSrc;
        create_info.sharingMode = vk::SharingMode::eExclusive;

        vk::Buffer res;
        VmaAllocation alloc;
        VmaAllocationInfo info;
        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT|VMA_ALLOCATION_CREATE_MAPPED_BIT;

        vmaCreateBuffer(
            devices[i].allocator, (VkBufferCreateInfo*)&create_info,
            &alloc_info, reinterpret_cast<VkBuffer*>(&res),
            &alloc, &info
        );
        tracking_resources[i].tracking_data = vkm<vk::Buffer>(devices[i], res, alloc);
        tracking_resources[i].tracking_data_ptr = (tracking_stamp*)info.pMappedData;

        create_info.size = sizeof(uint32_t) * MAX_FRAMES_IN_FLIGHT;
        vmaCreateBuffer(
            devices[i].allocator, (VkBufferCreateInfo*)&create_info,
            &alloc_info, reinterpret_cast<VkBuffer*>(&res),
            &alloc, &info
        );
        tracking_resources[i].frame_buffer = vkm<vk::Buffer>(devices[i], res, alloc);
        tracking_resources[i].frame_buffer_ptr = (uint32_t*)info.pMappedData;
    }

    poll_thread.emplace(poll_worker, this);
}

void progress_tracker::end()
{
    if(!running) return;

    running = false;
    cv.notify_all();
    poll_thread->join();
    tracking_resources.clear();
    // Show cursor again
    std::cout << "\x1b[?25h";
}

void progress_tracker::prepare_frame(size_t device_index, uint32_t frame_index, uint64_t frame)
{
    if(running)
    {
        tracking_resources[device_index].frame_buffer_ptr[frame_index] = frame;
    }
}

void progress_tracker::record_tracking(size_t device_index, vk::CommandBuffer cmd, uint32_t frame_index, uint32_t step)
{
    if(running)
    {
        if(step == 0)
        {
            cmd.copyBuffer(
                *tracking_resources[device_index].frame_buffer,
                *tracking_resources[device_index].tracking_data,
                vk::BufferCopy(frame_index * sizeof(uint32_t), 0, sizeof(uint32_t))
            );
        }
        cmd.updateBuffer(
            *tracking_resources[device_index].tracking_data,
            sizeof(uint32_t),
            sizeof(uint32_t),
            &step
        );
    }
}

bool progress_tracker::tracking_stamp::operator<(const tracking_stamp& other) const
{
    if(frame < other.frame) return true;
    else if(frame > other.frame) return false;
    else return step < other.step;
}

bool progress_tracker::tracking_stamp::operator==(const tracking_stamp& other) const
{
    return frame == other.frame && step == other.step;
}

void progress_tracker::update_progress_bar(
    std::chrono::steady_clock::time_point start,
    const tracking_stamp& latest
){
    if(latest.frame >= opt.expected_frame_count)
        opt.expected_frame_count = latest.frame;

    if(latest.step >= opt.expected_steps_per_frame)
        opt.expected_steps_per_frame = latest.step;

    size_t total_steps = opt.expected_frame_count * opt.expected_steps_per_frame;
    size_t cur_steps = (latest.frame-1) * opt.expected_steps_per_frame + latest.step;
    float progress = float(cur_steps) / float(total_steps);

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
        if(i < fill_width || cur_steps == total_steps) c = '=';
        else if(i == fill_width) c = '>';

        std::cout << c;
    }
    std::cout << "] ";

    std::cout << std::setprecision(1) << 100.0f * progress << "%, ";
    int hours = time_left / 3600;
    if(hours > 0) std::cout << hours << "h ";
    time_left %= 3600;

    int minutes = time_left / 60;
    if(minutes > 0) std::cout << minutes << "m ";
    time_left %= 60;

    std::cout << time_left << "s left";

    std::cout << std::flush;
}

void progress_tracker::poll_worker(progress_tracker* self)
{
    std::mutex wait_mutex;
    std::unique_lock<std::mutex> lk(wait_mutex);
    tracking_stamp latest = {0, 0};

    std::vector<device_data>& devices = self->ctx->get_devices();

    std::chrono::steady_clock::time_point start_time;
    bool first = true;

    while(self->running)
    {
        self->cv.wait_for(lk, std::chrono::milliseconds(self->opt.poll_ms));

        tracking_stamp oldest;

        bool failed = false;
        for(size_t i = 0; i < devices.size(); ++i)
        {
            tracking_stamp stamp = *self->tracking_resources[i].tracking_data_ptr;
            if(stamp < self->tracking_resources[i].last_value)
            {
                failed = true;
                break;
            }
            self->tracking_resources[i].last_value = stamp;
            if(i == 0 || stamp < oldest)
                oldest = stamp;
        }
        if(failed) continue;

        if(!(latest == oldest))
        {
            latest = oldest;
            if(first)
            {
                start_time = std::chrono::steady_clock::now();
                first = false;
            }
            self->update_progress_bar(start_time, latest);
        }
    }
}

}
