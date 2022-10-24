#include "headless.hh"
#include "misc.hh"
#include "tinyexr.h"
#include "stb_image_write.h"
#include <iostream>
#include <fstream>
#include <cstring>

namespace
{

using namespace tr;

int get_compression_type(headless::compression_type type)
{
    switch(type)
    {
    case headless::NONE: return TINYEXR_COMPRESSIONTYPE_NONE;
    case headless::RLE: return TINYEXR_COMPRESSIONTYPE_RLE;
    case headless::ZIPS: return TINYEXR_COMPRESSIONTYPE_ZIPS;
    case headless::ZIP: return TINYEXR_COMPRESSIONTYPE_ZIP;
    case headless::PIZ: return TINYEXR_COMPRESSIONTYPE_PIZ;
    }
    assert(false);
    return 0;
}

void parse_pixel_format(
    headless::pixel_format format,
    int& channels,
    int& pixel_type
){
    switch(format)
    {
    case headless::RGB16:
        channels = 3;
        pixel_type = TINYEXR_PIXELTYPE_HALF;
        break;
    case headless::RGB32:
        channels = 3;
        pixel_type = TINYEXR_PIXELTYPE_FLOAT;
        break;
    case headless::RGBA16:
        channels = 4;
        pixel_type = TINYEXR_PIXELTYPE_HALF;
        break;
    case headless::RGBA32:
        channels = 4;
        pixel_type = TINYEXR_PIXELTYPE_FLOAT;
        break;
    }
}

vk::Format sdl_to_vk_format(SDL_Surface* display_surface)
{
    SDL_PixelFormat* format = display_surface->format;
    if(format->BytesPerPixel != 4)
        throw std::runtime_error(
            "SDL does not have a 4-channel pixel format, direct memcpy does "
            "not suffice!"
        );
    if(format->Rmask == 0x00FF0000)
        return vk::Format::eB8G8R8A8Unorm;
    else if(format->Rmask == 0x000000FF)
        return vk::Format::eR8G8B8A8Unorm;
    else throw std::runtime_error("SDL has an incompatible pixel format!");
}

}

namespace tr
{

headless::headless(const options& opt)
: context(opt), opt(opt)
{
    if(opt.viewer && opt.display_count > 1)
        throw std::runtime_error(
            "More than one display is only allowed in fully headless mode"
        );

    if(opt.viewer) init_sdl();
    init_vulkan(vkGetInstanceProcAddr);
    init_devices();
    init_images();
    init_resources();
}

headless::~headless()
{
    if(!opt.viewer)
    {
        for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
            headless::save_image(i);
        reap_workers(false);
    }
    deinit_resources();
    deinit_images();
    deinit_devices();
    deinit_vulkan();
    if(opt.viewer) deinit_sdl();
}

uint32_t headless::prepare_next_image(uint32_t frame_index)
{
    device_data& d = get_display_device();
    d.graphics_queue.submit(
        vk::SubmitInfo(
            0, nullptr, nullptr, 0, nullptr,
            1, frame_available[frame_index].get()
        ),
        {}
    );
    return frame_index;
}

void headless::finish_image(
    uint32_t frame_index,
    uint32_t swapchain_index,
    bool display
){
    device_data& d = get_display_device();
    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTopOfPipe;

    if(!display || opt.output_file_type == EMPTY)
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

    if(opt.viewer)
    {
        view_image(swapchain_index);
    }
    else
    {
        // Save image from previous time this image was written to, since we're
        // going to overwrite that in the next operation.
        save_image(swapchain_index);
    }

    if(opt.output_file_type != EMPTY)
    {
        per_image_data& id = per_image[swapchain_index];
        d.graphics_queue.submit(
            vk::SubmitInfo(
                1, frame_finished[frame_index], &wait_stage,
                1, id.copy_cb,
                0, nullptr
            ),
            id.copy_fence
        );
        id.copy_ongoing = true;
        id.frame_number = opt.first_frame_index + get_displayed_frame_counter();
    }
}

bool headless::queue_can_present(
    const vk::PhysicalDevice&, uint32_t,
    const vk::QueueFamilyProperties&
){
    // Headless doesn't present.
    return false;
}

void headless::init_images()
{
    device_data& dev_data = get_display_device();
    opt.display_count = max(opt.display_count, 1u);

    image_size = opt.size;
    image_array_layers = opt.display_count;
    image_format = opt.viewer ?
        sdl_to_vk_format(display_surface) : vk::Format::eR32G32B32A32Sfloat;
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

        id.copy_fence = vkm(dev_data, dev_data.dev.createFence({}));

        per_image.emplace_back(std::move(id));
    }
    reset_image_views();
}

void headless::deinit_images()
{
    array_image_views.clear();
    images.clear();
    sync();
    per_image.clear();
}

void headless::init_sdl()
{
    uint32_t subsystems = SDL_INIT_VIDEO|SDL_INIT_JOYSTICK|
        SDL_INIT_GAMECONTROLLER|SDL_INIT_EVENTS;
    if(SDL_Init(subsystems))
        throw std::runtime_error(SDL_GetError());

    win = SDL_CreateWindow(
        "TauRay",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        opt.size.x,
        opt.size.y,
        (opt.viewer_fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0)
    );
    if(!win) throw std::runtime_error(SDL_GetError());
    SDL_GetWindowSize(win, (int*)&opt.size.x, (int*)&opt.size.y);
    SDL_SetWindowGrab(win, (SDL_bool)true);
    SDL_SetRelativeMouseMode((SDL_bool)true);

    display_surface = SDL_GetWindowSurface(win);
}

void headless::deinit_sdl()
{
    SDL_DestroyWindow(win);
    SDL_Quit();
}

void headless::save_image(uint32_t swapchain_index)
{
    device_data& d = get_display_device();
    per_image_data& id = per_image[swapchain_index];
    if(!id.copy_ongoing) return;

    (void)d.dev.waitForFences(*id.copy_fence, true, UINT64_MAX);
    d.dev.resetFences(*id.copy_fence);

    // Map memory, save images
    float* all_mem = nullptr;
    vmaMapMemory(d.allocator, id.staging_buffer.get_allocation(), (void**)&all_mem);

    for(size_t display_index = 0; display_index < opt.display_count; ++display_index)
    {
        std::string filename = opt.output_prefix;
        if(opt.display_count > 1) filename += std::to_string(display_index)+"_";
        std::string frame_num_string = std::to_string(id.frame_number);
        std::string padded_string  = std::string(6 - std::min((size_t)6, frame_num_string.length()), '0') + frame_num_string;
        //if(!opt.single_frame) filename += std::to_string(id.frame_number);
        if(!opt.single_frame) filename += padded_string;

        reap_workers(true);
        while(save_workers.size() >= std::thread::hardware_concurrency())
        {
            {
                std::unique_lock<std::mutex> lock(save_workers_mutex);
                save_workers_cv.wait(lock);
            }
            reap_workers(true);
        }

        size_t image_pixels = opt.size.x*opt.size.y;
        size_t pixel_offset = image_pixels * 4 * display_index;
        float* mem = all_mem + pixel_offset;

        if(!opt.skip_nan_check)
        {
            for(size_t j = 0; j < image_pixels; ++j)
                if(any(isnan(vec4(
                    mem[j*4+0],
                    mem[j*4+1],
                    mem[j*4+2],
                    mem[j*4+3]
                ))))
                    std::cout << "NaN pixel at: "
                        << (j%opt.size.x) << ", " << (j/opt.size.x) << std::endl;
        }

        if(opt.output_file_type == headless::EXR)
        {
            filename += ".exr";

            std::vector<float> channel_data(4*image_pixels);
            for(int i = 0; i < 4; ++i)
            for(size_t j = 0; j < image_pixels; ++j)
                channel_data[i*image_pixels+j] = mem[j*4+i];

            worker* w = new worker;
            save_workers.emplace_back(w);
            save_workers.back()->t = std::thread([
                this,
                filename,
                image_pixels,
                channel_data = std::move(channel_data),
                w
            ]() mutable {
                int num_channels = 0;
                int pixel_type = 0;
                parse_pixel_format(opt.output_format, num_channels, pixel_type);

                EXRHeader header;
                InitEXRHeader(&header);
                header.num_channels = num_channels;
                header.compression_type = get_compression_type(opt.output_compression);
                EXRChannelInfo channel_infos[4];
                header.channels = channel_infos;

                int pixel_types[4] = {
                    TINYEXR_PIXELTYPE_FLOAT,
                    TINYEXR_PIXELTYPE_FLOAT,
                    TINYEXR_PIXELTYPE_FLOAT,
                    TINYEXR_PIXELTYPE_FLOAT
                };
                header.pixel_types = pixel_types;
                int requested_pixel_types[4] = {
                    pixel_type, pixel_type, pixel_type, pixel_type
                };
                header.requested_pixel_types = requested_pixel_types;

                EXRImage image;
                InitEXRImage(&image);
                image.num_channels = num_channels;

                // BGRA order
                float* image_ptr[4];
                if(num_channels == 3)
                {
                    strncpy(channel_infos[0].name, "B", 2);
                    strncpy(channel_infos[1].name, "G", 2);
                    strncpy(channel_infos[2].name, "R", 2);
                    image_ptr[0] = channel_data.data() + 2*image_pixels;
                    image_ptr[1] = channel_data.data() + 1*image_pixels;
                    image_ptr[2] = channel_data.data() + 0*image_pixels;
                }
                else
                {
                    strncpy(channel_infos[0].name, "A", 2);
                    strncpy(channel_infos[1].name, "B", 2);
                    strncpy(channel_infos[2].name, "G", 2);
                    strncpy(channel_infos[3].name, "R", 2);
                    image_ptr[0] = channel_data.data() + 3*image_pixels;
                    image_ptr[1] = channel_data.data() + 2*image_pixels;
                    image_ptr[2] = channel_data.data() + 1*image_pixels;
                    image_ptr[3] = channel_data.data() + 0*image_pixels;
                }
                image.images = (uint8_t**)image_ptr;
                image.width = opt.size.x;
                image.height = opt.size.y;

                const char* err = nullptr;
                int ret = SaveEXRImageToFile(&image, &header, filename.c_str(), &err);
                if(ret != TINYEXR_SUCCESS)
                    throw std::runtime_error("Failed to write " + filename + ": " + err);

                {
                    std::lock_guard<std::mutex> lock(save_workers_mutex);
                    std::cout << "Saved " << filename << std::endl;
                    w->finished = true;
                }
                save_workers_cv.notify_one();
            });
        }
        else if(opt.output_file_type == headless::PNG)
        {
            filename += ".png";

            std::vector<uint8_t> pixel_data(4*image_pixels);
            for(size_t j = 0; j < image_pixels*4; ++j)
                pixel_data[j] = clamp((int)round(mem[j]*255.0f), 0, 255);

            worker* w = new worker;
            save_workers.emplace_back(w);
            save_workers.back()->t = std::thread([
                this,
                filename,
                pixel_data = std::move(pixel_data),
                w
            ]() mutable {
                int ret = stbi_write_png(
                    filename.c_str(), opt.size.x, opt.size.y, 4, pixel_data.data(),
                    opt.size.x*4
                );
                if(ret == 0)
                {
                    throw std::runtime_error("Failed to write " + filename);
                }

                {
                    std::lock_guard<std::mutex> lock(save_workers_mutex);
                    std::cout << "Saved " << filename << std::endl;
                    w->finished = true;
                }
                save_workers_cv.notify_one();
            });
        }
        else if(opt.output_file_type == headless::RAW)
        {
            filename += ".raw";

            std::vector<float> pixel_data(4*image_pixels);
            memcpy(pixel_data.data(), mem, sizeof(float)*pixel_data.size());

            worker* w = new worker;
            save_workers.emplace_back(w);
            save_workers.back()->t = std::thread([
                this,
                filename,
                pixel_data = std::move(pixel_data),
                w
            ]() mutable {
                std::fstream f(filename, std::ios::out | std::ios::binary);
                if(!f) throw std::runtime_error("Failed to write " + filename);
                f.write((char*)pixel_data.data(), pixel_data.size()*sizeof(float));
                f.close();

                {
                    std::lock_guard<std::mutex> lock(save_workers_mutex);
                    std::cout << "Saved " << filename << std::endl;
                    w->finished = true;
                }
                save_workers_cv.notify_one();
            });
        }
    }
    vmaUnmapMemory(d.allocator, id.staging_buffer.get_allocation());
}

void headless::view_image(uint32_t swapchain_index)
{
    device_data& d = get_display_device();
    per_image_data& id = per_image[swapchain_index];
    if(!id.copy_ongoing) return;

    (void)d.dev.waitForFences(*id.copy_fence, true, UINT64_MAX);
    d.dev.resetFences(*id.copy_fence);

    uint8_t* mem = nullptr;
    SDL_LockSurface(display_surface);
    vmaMapMemory(d.allocator, id.staging_buffer.get_allocation(), (void**)&mem);

    memcpy(display_surface->pixels, mem, 4*opt.size.x*opt.size.y);

    vmaUnmapMemory(d.allocator, id.staging_buffer.get_allocation());
    SDL_UnlockSurface(display_surface);

    SDL_UpdateWindowSurface(win);
}

void headless::reap_workers(bool finished_only)
{
    for(auto it = save_workers.begin(); it != save_workers.end();)
    {
        bool finished = true;
        if(finished_only)
        {
            std::lock_guard<std::mutex> lock(save_workers_mutex);
            finished = (*it)->finished;
        }
        if(finished)
        {
            (*it)->t.join();
            it = save_workers.erase(it);
        }
        else it++;
    }
}

}
