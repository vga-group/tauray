#include "window.hh"
#include "log.hh"
#include <iostream>

namespace tr
{

window::window(const options& opt)
: context(opt), opt(opt)
{
    init_sdl();
    init_vulkan((PFN_vkGetInstanceProcAddr)SDL_Vulkan_GetVkGetInstanceProcAddr());
    if(!SDL_Vulkan_CreateSurface(win, instance, &surface))
        throw std::runtime_error(SDL_GetError());
    init_devices();
    init_swapchain();
    init_resources();
}

window::~window()
{
    deinit_resources();
    deinit_swapchain();
    deinit_devices();
    vkDestroySurfaceKHR(instance, surface, nullptr);
    deinit_vulkan();
    deinit_sdl();
}

void window::recreate_swapchains()
{
    device_data& dev_data = get_display_device();
    dev_data.dev.waitIdle();

    deinit_swapchain();
    init_swapchain();
}

uint32_t window::prepare_next_image(uint32_t frame_index)
{
    device_data& d = get_display_device();
    uint32_t swapchain_index = d.dev.acquireNextImageKHR(
        swapchain, UINT64_MAX, frame_available[frame_index], {}
    ).value;
    return swapchain_index;
}

void window::finish_image(
    uint32_t frame_index,
    uint32_t swapchain_index,
    bool /*display*/
){
    device_data& d = get_display_device();
    // TODO: Honor display variable? Not really essential here since window
    // doesn't collect datasets.
    (void)d.present_queue.presentKHR({
        1, frame_finished[frame_index],
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
        SDL_INIT_GAMECONTROLLER|SDL_INIT_EVENTS;
    if(SDL_Init(subsystems))
        throw std::runtime_error(SDL_GetError());

    win = SDL_CreateWindow(
        "Tauray",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        opt.size.x,
        opt.size.y,
        SDL_WINDOW_VULKAN | (opt.fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : SDL_WINDOW_ALWAYS_ON_TOP)
    );
    if(!win) throw std::runtime_error(SDL_GetError());
    SDL_GetWindowSize(win, (int*)&opt.size.x, (int*)&opt.size.y);
    SDL_SetWindowGrab(win, (SDL_bool)true);
    SDL_SetRelativeMouseMode((SDL_bool)true);
    image_size = opt.size;
    image_array_layers = 1;

    unsigned count = 0;
    if(!SDL_Vulkan_GetInstanceExtensions(win, &count, nullptr))
        throw std::runtime_error(SDL_GetError());

    extensions.resize(count);
    if(!SDL_Vulkan_GetInstanceExtensions(win, &count, extensions.data()))
        throw std::runtime_error(SDL_GetError());
}

void window::deinit_sdl()
{
    SDL_DestroyWindow(win);
    SDL_Quit();
}

void window::init_swapchain()
{
    device_data& dev_data = get_display_device();
    std::vector<vk::SurfaceFormatKHR> formats =
        dev_data.pdev.getSurfaceFormatsKHR(surface);

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
        TR_ERR(
            "Could not find any suitable swap chain format!"
            "Using the first available format instead, results may look "
            "incorrect."
        );
    image_format = swapchain_format.format;
    expected_image_layout = vk::ImageLayout::ePresentSrcKHR;

    // Find the present mode matching our vsync setting.
    std::vector<vk::PresentModeKHR> modes =
        dev_data.pdev.getSurfacePresentModesKHR(surface);
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
        TR_ERR("Could not find desired present mode, falling back to first "
            "available mode.");

    // Find the size that matches our window size
    vk::SurfaceCapabilitiesKHR caps =
        dev_data.pdev.getSurfaceCapabilitiesKHR(surface);
    vk::Extent2D selected_extent = caps.currentExtent;
    if(caps.currentExtent.width == UINT32_MAX)
    {
        uvec2 clamped_size = clamp(
            opt.size,
            uvec2(caps.minImageExtent.width, caps.minImageExtent.height),
            uvec2(caps.maxImageExtent.width, caps.maxImageExtent.height)
        );
        selected_extent.width = clamped_size.x;
        selected_extent.height = clamped_size.y;
    }
    if(
        selected_extent.width != opt.size.x ||
        selected_extent.height != opt.size.y
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
    swapchain = dev_data.dev.createSwapchainKHR({
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

    // Get swap chain images & create image views
    auto swapchain_images = dev_data.dev.getSwapchainImagesKHR(swapchain);
    images.clear();
    for(vk::Image img: swapchain_images)
        images.emplace_back(vkm(dev_data, img));
    reset_image_views();
}

void window::deinit_swapchain()
{
    vk::Device& dev = get_display_device().dev;
    array_image_views.clear();
    images.clear();
    sync();
    dev.destroySwapchainKHR(swapchain);
}

}
