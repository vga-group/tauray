#include "frame_server.hh"
#include "misc.hh"
#include "tinyexr.h"
#include "stb_image_write.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <csignal>
#include <nng/nng.h>
#include <nng/protocol/bus0/bus.h>

namespace
{
bool should_exit = false;

void exit_handler(int)
{
    should_exit = true;
}

}

namespace tr
{

frame_server::frame_server(const options& opt)
:   context(opt), opt(opt), exit_streamer(0), pause_rendering(false),
    image_reader_thread(read_image_worker, this),
    streamer_thread(streamer_worker, this)
{
    init_sdl();
    init_vulkan(vkGetInstanceProcAddr);
    init_devices();
    init_images();
    init_resources();
    signal(SIGINT, exit_handler);
}

frame_server::~frame_server()
{
    exit_streamer = true;
    frame_queue_cv.notify_one();
    copy_start_cv.notify_one();
    streamer_thread.join();

    deinit_resources();
    deinit_images();
    deinit_devices();
    deinit_vulkan();
    deinit_sdl();
}

bool frame_server::init_frame()
{
    while(!should_exit && pause_rendering)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return should_exit;
}

uint32_t frame_server::prepare_next_image(uint32_t frame_index)
{
    device& d = get_display_device();
    d.graphics_queue.submit(
        vk::SubmitInfo(
            0, nullptr, nullptr, 0, nullptr,
            1, frame_available[frame_index].get()
        ),
        {}
    );
    return frame_index;
}

void frame_server::finish_image(
    uint32_t frame_index,
    uint32_t swapchain_index,
    bool display
){
    device& d = get_display_device();
    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTopOfPipe;

    if(!display)
    {
        // Eat the binary semaphore.
        d.graphics_queue.submit(
            vk::SubmitInfo(
                1, frame_finished[frame_index], &wait_stage,
                0, nullptr,
                0, nullptr
            ), {}
        );
        return;
    }

    {
        std::unique_lock lk(image_mutex);

        copy_finish_cv.wait(
            lk,
            [&](){ return !per_image[swapchain_index].copy_ongoing; }
        );
    }

    per_image_data& id = per_image[swapchain_index];
    d.graphics_queue.submit(
        vk::SubmitInfo(
            1, frame_finished[frame_index], &wait_stage,
            1, id.copy_cb,
            0, nullptr
        ),
        id.copy_fence
    );

    {
        std::unique_lock lk(image_mutex);
        id.copy_ongoing = true;
        image_read_queue.push_back(swapchain_index);
    }
    copy_start_cv.notify_one();
}

bool frame_server::queue_can_present(
    const vk::PhysicalDevice&, uint32_t,
    const vk::QueueFamilyProperties&
){
    return false;
}

void frame_server::init_images()
{
    device& dev_data = get_display_device();

    image_size = opt.size;
    image_array_layers = 1;
    image_format = vk::Format::eR8G8B8A8Unorm;
    expected_image_layout = vk::ImageLayout::eTransferSrcOptimal;

    vk::ImageCreateInfo img_info{
        {},
        vk::ImageType::e2D,
        image_format,
        {(uint32_t)opt.size.x, (uint32_t)opt.size.y, (uint32_t)1},
        1,
        image_array_layers,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferSrc |
        vk::ImageUsageFlagBits::eStorage |
        vk::ImageUsageFlagBits::eColorAttachment,
        vk::SharingMode::eExclusive
    };

    images.clear();
    for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        images.emplace_back(
            sync_create_gpu_image(
                dev_data,
                img_info,
                vk::ImageLayout::eTransferSrcOptimal,
                0, nullptr
            )
        );

        per_image_data id;
        vk::BufferCreateInfo staging_info(
            {}, opt.size.x*opt.size.y*sizeof(float)*4*image_array_layers,
            vk::BufferUsageFlagBits::eTransferDst,
            vk::SharingMode::eExclusive
        );
        id.staging_buffer = create_buffer(
            dev_data,
            staging_info,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
        );

        id.copy_cb = create_graphics_command_buffer(dev_data);
        id.copy_cb->begin(vk::CommandBufferBeginInfo{});

        vk::BufferImageCopy region(
            0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, image_array_layers},
            {0,0,0}, img_info.extent
        );
        id.copy_cb->copyImageToBuffer(
            images.back(),
            vk::ImageLayout::eTransferSrcOptimal,
            id.staging_buffer,
            1,
            &region
        );

        id.copy_cb->end();

        id.copy_fence = vkm(dev_data, dev_data.logical.createFence({}));

        per_image.emplace_back(std::move(id));
    }
    reset_image_views();
}

void frame_server::deinit_images()
{
    array_image_views.clear();
    images.clear();
    sync();
    per_image.clear();
}

void frame_server::init_sdl()
{
    uint32_t subsystems = SDL_INIT_EVENTS;
    putenv((char*)"SDL_VIDEODRIVER=dummy");
    if(SDL_Init(subsystems))
        throw std::runtime_error(SDL_GetError());
}

void frame_server::deinit_sdl()
{
    SDL_Quit();
}

void frame_server::read_image_worker(frame_server* s)
{
    for(;;)
    {
        uint32_t image_index = 0;
        {
            std::unique_lock lk(s->image_mutex);

            s->copy_start_cv.wait(
                lk,
                [&](){ return s->exit_streamer || s->image_read_queue.size() != 0; }
            );

            if(s->exit_streamer)
                break;

            image_index = s->image_read_queue.front();
            s->image_read_queue.erase(s->image_read_queue.begin());
        }
        device& d = s->get_display_device();
        auto& id = s->per_image[image_index];

        std::vector<uint8_t> latest_frame(s->opt.size.x * s->opt.size.y * 3);

        (void)d.logical.waitForFences(*id.copy_fence, true, UINT64_MAX);
        d.logical.resetFences(*id.copy_fence);

        uint8_t* mem = nullptr;
        vmaMapMemory(d.allocator, id.staging_buffer.get_allocation(), (void**)&mem);

        for(uint32_t i = 0; i < s->opt.size.x * s->opt.size.y; ++i)
        {
            latest_frame[i*3 + 0] = mem[i*4 + 0];
            latest_frame[i*3 + 1] = mem[i*4 + 1];
            latest_frame[i*3 + 2] = mem[i*4 + 2];
        }

        vmaUnmapMemory(d.allocator, id.staging_buffer.get_allocation());

        {
            std::unique_lock lk(s->frame_queue_mutex);
            s->frame_queue.emplace_back(std::move(latest_frame));
        }
        {
            std::unique_lock lk(s->image_mutex);
            id.copy_ongoing = false;
        }
        s->frame_queue_cv.notify_one();
        s->copy_finish_cv.notify_one();
    }
}

void frame_server::streamer_worker(frame_server* s)
{
    using namespace std::chrono_literals;

    nng_socket socket;
    nng_bus0_open(&socket);

    std::string address = "tcp://*:"+std::to_string(s->opt.port_number);
    nng_listener listener = NNG_LISTENER_INITIALIZER;
    nng_listen(socket, address.c_str(), &listener, NNG_FLAG_NONBLOCK);

    auto last_request_timestamp = std::chrono::steady_clock::now();

    nng_msg* msg = nullptr;

    for(;;)
    {
        std::vector<uint8_t> frame_data;
        {
            std::unique_lock<std::mutex> lk(s->frame_queue_mutex);
            while(!s->frame_queue_cv.wait_for(
                lk,
                std::chrono::milliseconds(10),
                [&]{
                    if(s->exit_streamer) return true;
                    int result = nng_recvmsg(socket, &msg, NNG_FLAG_NONBLOCK);
                    return result == 0 || !s->frame_queue.empty();
                }
            ));
            if(s->exit_streamer)
                break;

            if(!s->frame_queue.empty())
            {
                frame_data = std::move(s->frame_queue.front());
                s->frame_queue.erase(s->frame_queue.begin());
            }
        }

        if(msg != nullptr)
        { // Read input events from clients
            SDL_Event* events = (SDL_Event*)nng_msg_body(msg);
            size_t event_count = nng_msg_len(msg)/sizeof(SDL_Event);
            for(size_t i = 0; i < event_count; ++i)
                SDL_PushEvent(events+i);
            nng_msg_free(msg);
            msg = nullptr;

            last_request_timestamp = std::chrono::steady_clock::now();
            s->pause_rendering = false;
        }
        else
        {
            auto timestamp = std::chrono::steady_clock::now();
            auto duration_since_last_request = timestamp - last_request_timestamp;
            if(duration_since_last_request > 1s)
                s->pause_rendering = true;
        }

        if(frame_data.size())
        { // Send frame to clients
            nng_msg* msg = nullptr;
            nng_msg_alloc(&msg, 0);
            nng_msg_append_u32(msg, s->opt.size.x);
            nng_msg_append_u32(msg, s->opt.size.y);
            nng_msg_append_u32(msg, 3);
            nng_msg_append(msg, frame_data.data(), frame_data.size());
            if(nng_sendmsg(socket, msg, 0) != 0)
                nng_msg_free(msg);
        }
    }

    nng_close(socket);
}

}

