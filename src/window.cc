#include "window.hh"
#include "log.hh"
#include "misc.hh"
#include <iostream>

namespace tr
{

window::window(const options& opt)
: context(opt), opt(opt)
{
    init_sdl();
    init_vulkan((PFN_vkGetInstanceProcAddr)SDL_Vulkan_GetVkGetInstanceProcAddr());
    if(!SDL_Vulkan_CreateSurface(
        win, instance, (const VkAllocationCallbacks*)nullptr, &surface
    ))
        throw std::runtime_error(SDL_GetError());
    init_devices();
    init_swapchain();
    init_resources();
}

window::~window()
{
    sync();

    deinit_resources();
    deinit_swapchain();
    deinit_devices();
    vkDestroySurfaceKHR(instance, surface, nullptr);
    deinit_vulkan();
    deinit_sdl();
}

size_t window::get_swapchain_image_count() const
{
    return window_images.size();
}

std::vector<render_target> window::get_array_render_target()
{
    std::vector<render_target> frames;
    for(size_t i = 0; i < window_images.size(); ++i)
    {
        frames.emplace_back(
            image_size,
            0, image_array_layers,
            images[0],
            array_image_views[0],
            vk::ImageLayout::eUndefined,
            image_format,
            vk::SampleCountFlagBits::e1
        );
    }
    return frames;
}

void window::recreate_swapchains()
{
    device& dev_data = get_display_device();
    dev_data.logical.waitIdle();

    deinit_swapchain();
    init_swapchain();
}

uint32_t window::prepare_next_image(uint32_t frame_index)
{
    device& d = get_display_device();
    uint32_t swapchain_index = d.logical.acquireNextImageKHR(
        swapchain, UINT64_MAX, frame_available[frame_index], {}
    ).value;
    return swapchain_index;
}

dependencies window::fill_end_frame_dependencies(const dependencies& deps)
{
    if (opt.views.x == 1 && opt.views.y == 1)
        return deps;
    else
    {
        // Multi-view. We have to copy all of the views to the actual swapchain
        // image.
        return composition->run(deps);
    }
}

void window::finish_image(
    uint32_t /*frame_index*/,
    uint32_t swapchain_index,
    bool /*display*/
){
    device& d = get_display_device();
    // TODO: Honor display variable? Not really essential here since window
    // doesn't collect datasets.
    (void)d.present_queue.presentKHR({
        1, frame_finished[swapchain_index],
        1, &swapchain,
        &swapchain_index
    });
}

bool window::queue_can_present(
    const vk::PhysicalDevice& device,
    uint32_t queue_index,
    const vk::QueueFamilyProperties&
){
    return device.getSurfaceSupportKHR(queue_index, vk::SurfaceKHR(surface)) &&
        device.getSurfaceFormatsKHR(surface).size() > 0 &&
        device.getSurfacePresentModesKHR(surface).size() > 0;
}

void window::init_sdl()
{
    uint32_t subsystems = SDL_INIT_VIDEO|SDL_INIT_JOYSTICK|
        SDL_INIT_GAMEPAD|SDL_INIT_EVENTS;
    if(!SDL_Init(subsystems))
        throw std::runtime_error(SDL_GetError());

    uvec2 window_size = opt.size * opt.views;

    win = SDL_CreateWindow(
        "Tauray",
        window_size.x,
        window_size.y,
        SDL_WINDOW_VULKAN | (opt.fullscreen ? SDL_WINDOW_FULLSCREEN : SDL_WINDOW_ALWAYS_ON_TOP)
    );
    if(!win) throw std::runtime_error(SDL_GetError());
    SDL_GetWindowSize(win, (int*)&window_size.x, (int*)&window_size.y);
    opt.size = window_size / opt.views;
    //SDL_SetWindowKeyboardGrab(win, true);
    SDL_SetWindowMouseGrab(win, true);
    SDL_SetWindowRelativeMouseMode(win, true);
    image_size = opt.size;
    image_array_layers = opt.views.x * opt.views.y;

    unsigned count = 0;
    const char* const* exts = SDL_Vulkan_GetInstanceExtensions(&count);
    if(!exts)
        throw std::runtime_error(SDL_GetError());

    extensions.assign(exts, exts + count);
}

void window::deinit_sdl()
{
    SDL_DestroyWindow(win);
    SDL_Quit();
}

void window::init_swapchain()
{
    device& dev_data = get_display_device();
    std::vector<vk::SurfaceFormatKHR> formats =
        dev_data.physical.getSurfaceFormatsKHR(surface);

    uvec2 window_size = opt.size * opt.views;
    bool singleview = opt.views.x == 1 && opt.views.y == 1;

    // Find the format matching our desired format.
    bool found_format = false;
    vk::SurfaceFormatKHR swapchain_format = formats[0];
    for(vk::SurfaceFormatKHR& format: formats)
    {
        if(
            (!opt.hdr_display && format.format == vk::Format::eB8G8R8A8Unorm &&
            format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) ||
            (opt.hdr_display && format.format == vk::Format::eR16G16B16A16Sfloat)
        ){
            swapchain_format = format;
            found_format = true;
            break;
        }
    }
    if(!found_format)
        TR_WARN(
            "Could not find any suitable swap chain format!"
            "Using the first available format instead, results may look "
            "incorrect."
        );
    image_format = swapchain_format.format;
    expected_image_layout = singleview ?
        vk::ImageLayout::ePresentSrcKHR :
        vk::ImageLayout::eGeneral;

    // Find the present mode matching our vsync setting.
    std::vector<vk::PresentModeKHR> modes =
        dev_data.physical.getSurfacePresentModesKHR(surface);
    bool found_mode = false;
    vk::PresentModeKHR selected_mode = modes[0];
    if(opt.vsync)
    {
        if(
            std::find(
                modes.begin(),
                modes.end(),
                vk::PresentModeKHR::eMailbox
            ) != modes.end()
        ){
            selected_mode = vk::PresentModeKHR::eMailbox;
            found_mode = true;
        }
        else if(
            std::find(
                modes.begin(),
                modes.end(),
                vk::PresentModeKHR::eFifo
            ) != modes.end()
        ){
            selected_mode = vk::PresentModeKHR::eFifo;
            found_mode = true;
        }
    }
    else
    {
        if(
            std::find(
                modes.begin(),
                modes.end(),
                vk::PresentModeKHR::eImmediate
            ) != modes.end()
        ){
            selected_mode = vk::PresentModeKHR::eImmediate;
            found_mode = true;
        }
    }
    if(!found_mode)
        TR_WARN("Could not find desired present mode, falling back to first "
            "available mode.");

    // Find the size that matches our window size
    vk::SurfaceCapabilitiesKHR caps =
        dev_data.physical.getSurfaceCapabilitiesKHR(surface);
    vk::Extent2D selected_extent = caps.currentExtent;
    if(caps.currentExtent.width == UINT32_MAX)
    {
        uvec2 clamped_size = clamp(
            window_size,
            uvec2(caps.minImageExtent.width, caps.minImageExtent.height),
            uvec2(caps.maxImageExtent.width, caps.maxImageExtent.height)
        );
        selected_extent.width = clamped_size.x;
        selected_extent.height = clamped_size.y;
    }
    if(
        selected_extent.width != window_size.x ||
        selected_extent.height != window_size.y
    ) throw std::runtime_error(
        "Could not find swap chain extent matching the window size!"
    );

    // Create the actual swap chain!
    // + 1 avoids stalling when the previous image is used by the driver.
    uint32_t image_count = caps.minImageCount + 1;
    if(caps.maxImageCount != 0)
        image_count = min(image_count, caps.maxImageCount);

    vk::SharingMode sharing_mode;
    std::vector<uint32_t> queue_family_indices;
    if(dev_data.graphics_family_index == dev_data.present_family_index)
    {
        sharing_mode = vk::SharingMode::eExclusive;
        queue_family_indices = { dev_data.present_family_index };
    }
    else
    {
        sharing_mode = vk::SharingMode::eConcurrent;
        queue_family_indices = {
            dev_data.graphics_family_index,
            dev_data.present_family_index
        };
    }
    swapchain = dev_data.logical.createSwapchainKHR({
        {},
        surface,
        image_count,
        swapchain_format.format,
        swapchain_format.colorSpace,
        selected_extent,
        1,
        vk::ImageUsageFlagBits::eColorAttachment |
        vk::ImageUsageFlagBits::eStorage,
        sharing_mode,
        (uint32_t)queue_family_indices.size(),
        queue_family_indices.data(),
        caps.currentTransform,
        vk::CompositeAlphaFlagBitsKHR::eOpaque,
        selected_mode,
        true
    });

    auto swapchain_images = dev_data.logical.getSwapchainImagesKHR(swapchain);
    images.clear();

    // Get swap chain images & create image views
    if (opt.views.x == 1 && opt.views.y == 1)
    { // Single-view setup
        for(vk::Image img: swapchain_images)
            images.emplace_back(vkm(dev_data, img));
        reset_image_views();
    }
    else
    { // Multi-view setup
        window_images.clear();
        window_image_views.clear();
        std::vector<render_target> output_frames;
        for(vk::Image img: swapchain_images)
        {
            window_images.emplace_back(vkm(dev_data, img));
            window_image_views.emplace_back(dev_data,
                dev_data.logical.createImageView({
                    {},
                    img,
                    vk::ImageViewType::e2D,
                    swapchain_format.format,
                    {},
                    {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
                })
            );
            output_frames.emplace_back(
                opt.size * opt.views, 0, 1,
                window_images.back(),
                window_image_views.back(),
                vk::ImageLayout::eUndefined,
                image_format,
                vk::SampleCountFlagBits::e1
            );
        }
        vk::ImageCreateInfo info{
            {},
            vk::ImageType::e2D,
            swapchain_format.format,
            {opt.size.x, opt.size.y, 1},
            1,
            opt.views.x * opt.views.y,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eSampled|
            vk::ImageUsageFlagBits::eStorage|
            vk::ImageUsageFlagBits::eTransferDst|
            vk::ImageUsageFlagBits::eTransferSrc,
            vk::SharingMode::eExclusive
        };
        images.emplace_back(sync_create_gpu_image(
            dev_data, info, vk::ImageLayout::eGeneral
        ));
        reset_image_views();

        render_target input = get_array_render_target()[0];
        input.layout = expected_image_layout;
        composition.reset(new multiview_composition_stage(
            get_display_device(),
            input,
            output_frames,
            { opt.views }
        ));
    }
}

void window::deinit_swapchain()
{
    vk::Device& dev = get_display_device().logical;
    sync();
    composition.reset();
    array_image_views.clear();
    images.clear();
    window_image_views.clear();
    window_images.clear();
    sync();
    dev.destroySwapchainKHR(swapchain);
}

}
