#include "looking_glass.hh"
#include "misc.hh"
#include "log.hh"
#include "camera.hh"
#include <iostream>
#include <nng/nng.h>
#include <nng/protocol/reqrep0/req.h>
#include <cbor.h>

namespace
{

std::string get_cbor_string(cbor_item_t* item)
{
    return std::string((char*)cbor_string_handle(item), cbor_string_length(item));
}

float get_calibration_float(cbor_item_t* item)
{
    return cbor_float_get_float4(cbor_map_handle(item)[0].value);
}

};

namespace tr
{

looking_glass::looking_glass(const options& opt)
: context(opt), opt(opt)
{
    get_lkg_metadata();
    init_sdl();
    init_vulkan((PFN_vkGetInstanceProcAddr)SDL_Vulkan_GetVkGetInstanceProcAddr());
    if(!SDL_Vulkan_CreateSurface(win, instance, &surface))
        throw std::runtime_error(SDL_GetError());
    init_devices();
    init_swapchain();
    init_resources();
    init_render_target();
}

looking_glass::~looking_glass()
{
    deinit_render_target();
    deinit_resources();
    deinit_swapchain();
    deinit_devices();
    vkDestroySurfaceKHR(instance, surface, nullptr);
    deinit_vulkan();
    deinit_sdl();
}

void looking_glass::recreate_swapchains()
{
    device& dev_data = get_display_device();
    dev_data.logical.waitIdle();

    deinit_swapchain();
    init_swapchain();
}

void looking_glass::setup_cameras(
    scene& s,
    transformable_node* reference_frame
){
    s.foreach([&](entity id, camera&){s.remove<camera>(id);});

    float aspect = metadata.size.x/(float)metadata.size.y;
    float vfov = 2*atan(1/(2*opt.relative_view_distance))*180.0f/M_PI;
    for(size_t i = 0; i < opt.viewport_count; ++i)
    {
        camera cam;
        cam.perspective(vfov, aspect, 0.01f, 300.0f);

        float offset = ((i + 0.5f)/opt.viewport_count)*2.0f-1.0f;
        float angle = offset * metadata.view_cone * opt.depthiness;
        float ta = tan(glm::radians(angle));
        cam.set_fov(vfov);
        cam.set_pan(vec2(-ta, 0));
        vec4 dir = cam.get_projection_matrix() * vec4(0.0f, 0.0f, 1.0f, 1.0f);
        dir /= dir.z;
        cam.set_position(opt.mid_plane_dist * vec3(dir));
        cam.set_parent(reference_frame);
        s.add(std::move(cam), camera_metadata{true, int(i), true});
    }
}

uint32_t looking_glass::prepare_next_image(uint32_t frame_index)
{
    device& d = get_display_device();
    uint32_t swapchain_index = d.logical.acquireNextImageKHR(
        swapchain, UINT64_MAX, frame_available[frame_index], {}
    ).value;
    return swapchain_index;
}

dependencies looking_glass::fill_end_frame_dependencies(const dependencies& deps)
{
    return composition->run(deps);
}

void looking_glass::finish_image(
    uint32_t frame_index,
    uint32_t swapchain_index,
    bool /*display*/
){
    device& d = get_display_device();
    (void)d.present_queue.presentKHR({
        1, frame_finished[frame_index],
        1, &swapchain,
        &swapchain_index
    });
}

bool looking_glass::queue_can_present(
    const vk::PhysicalDevice& device,
    uint32_t queue_index,
    const vk::QueueFamilyProperties&
){
    return device.getSurfaceSupportKHR(queue_index, vk::SurfaceKHR(surface)) &&
        device.getSurfaceFormatsKHR(surface).size() > 0 &&
        device.getSurfacePresentModesKHR(surface).size() > 0;
}

void looking_glass::get_lkg_metadata()
{
    if(!opt.calibration_override)
    {
        // TODO: Use the official API instead, if it turns out to be available and
        // usable in Tauray.
        nng_socket sock;
        if(int err = nng_req0_open(&sock))
            throw std::runtime_error(nng_strerror(err));
        nng_dialer dialer;
        if(nng_dial(sock, "ipc:///tmp/holoplay-driver.ipc", &dialer, 0))
            throw std::runtime_error("HoloPlay service doesn't seem to be running.");

        // Initial handshake message
        cbor_item_t* cmd = cbor_new_definite_map(2);
        cbor_item_t* init = cbor_new_definite_map(1);
        cbor_item_t* appid = cbor_new_definite_map(1);
        cbor_map_add(appid, {
            cbor_move(cbor_build_string("appid")),
            cbor_move(cbor_build_string(""))
            });
        cbor_map_add(init, {
            cbor_move(cbor_build_string("init")),
            cbor_move(appid)
            });
        cbor_map_add(cmd, {
            cbor_move(cbor_build_string("cmd")),
            cbor_move(init)
            });
        cbor_map_add(cmd, {
            cbor_move(cbor_build_string("bin")),
            cbor_move(cbor_build_string(""))
            });
        uint8_t* buffer;
        size_t buffer_size;
        size_t length = cbor_serialize_alloc(cmd, &buffer, &buffer_size);
        nng_send(sock, buffer, length, 0);
        free(buffer);
        cbor_decref(&cmd);

        if (int err = nng_recv(sock, &buffer, &length, NNG_FLAG_ALLOC))
            throw std::runtime_error(nng_strerror(err));

        cbor_load_result res;
        cbor_item_t* response = cbor_load(buffer, length, &res);
        if (res.error.code != CBOR_ERR_NONE)
            throw std::runtime_error("CBOR load failure: " + std::to_string(res.error.code));
        nng_free(buffer, length);

        std::string error = "";
        size_t count = cbor_map_size(response);
        cbor_pair* response_pairs = cbor_map_handle(response);
        for (size_t i = 0; i < count; ++i)
        {
            cbor_pair pair = response_pairs[i];
            std::string key = get_cbor_string(pair.key);
            if (key == "error")
            {
                int err = cbor_get_uint8(pair.value);
                if (err != 0)
                    error = "HoloPlay Service refused us with error " + std::to_string(err);
            }
            else if (key == "version")
            {
                service_version = get_cbor_string(pair.value);
            }
            else if (key == "devices")
            {
                // TODO: How to deal with multiple looking glass devices?
                size_t count = cbor_array_size(pair.value);
                if (count == 0)
                    error = "Found zero Looking Glass devices!";
                else
                {
                    cbor_item_t* device = cbor_array_get(pair.value, 0);
                    metadata = get_lkg_device_metadata(device);
                }
            }
        }
        cbor_decref(&response);

        nng_dialer_close(dialer);
        nng_close(sock);

        TR_LOG("Using ", metadata.hardware_id, " (", metadata.hardware_version, ")");

        if (error.size() != 0)
            throw std::runtime_error(error);
    }
    else
    {
        TR_LOG("Using manually calibrated LF display");

        auto& md = metadata;
        auto& cd = *opt.calibration_override;

        md.dpi = cd.DPI;
        md.center = cd.center;
        md.flip_image.x = cd.flipImageX;
        md.flip_image.y = cd.flipImageY;
        md.flip_subpixel = cd.flipSubp;
        md.fringe = cd.fringe;
        md.invert = cd.invView;
        md.pitch = cd.pitch;
        md.size.x = cd.screenW;
        md.size.y = cd.screenH;
        md.slope = cd.slope;
        md.vertical_angle = cd.verticalAngle;
        md.view_cone = cd.viewCone;
        md.index = 0;
        md.window_coords.x = 0;
        md.window_coords.y = 0;

        metadata.corrected_pitch = metadata.size.x / metadata.dpi * metadata.pitch * sin(atan(fabs(metadata.slope)));
        metadata.tilt = metadata.size.y / (metadata.size.x * metadata.slope);
    }

    TR_LOG("dpi: ", metadata.dpi);
    TR_LOG("center: ", metadata.center);
    TR_LOG("pitch: ", metadata.pitch);
    TR_LOG("corrected_pitch: ", metadata.corrected_pitch);
    TR_LOG("size.x: ", metadata.size.x);
    TR_LOG("size.y: ", metadata.size.y);
    TR_LOG("slope: ", metadata.slope);
    TR_LOG("tilt: ", metadata.tilt);
    TR_LOG("vertical_angle: ", metadata.vertical_angle);
    TR_LOG("view_cone: ", metadata.view_cone);
    TR_LOG("window_coords.x: ", metadata.window_coords.x);
    TR_LOG("window_coords.y: ", metadata.window_coords.y);
}

looking_glass::device_metadata looking_glass::get_lkg_device_metadata(void* lkg_device)
{
    cbor_item_t* device = (cbor_item_t*)lkg_device;
    device_metadata md;
    size_t count = cbor_map_size(device);
    cbor_pair* device_pairs = cbor_map_handle(device);
    for(size_t i = 0; i < count; ++i)
    {
        cbor_pair pair = device_pairs[i];
        std::string key = get_cbor_string(pair.key);
        if(key == "calibration")
        {
            size_t calib_count = cbor_map_size(pair.value);
            cbor_pair* calib_pairs = cbor_map_handle(pair.value);
            for(size_t j = 0; j < calib_count; ++j)
            {
                cbor_pair pair = calib_pairs[j];
                std::string key = get_cbor_string(pair.key);
                TR_LOG(key);
                if(key == "DPI") md.dpi = get_calibration_float(pair.value);
                if(key == "center") md.center = get_calibration_float(pair.value);
                if(key == "configVersion") md.config_version = get_cbor_string(pair.value);
                if(key == "flipImageX") md.flip_image.x = get_calibration_float(pair.value) > 0.5f;
                if(key == "flipImageY") md.flip_image.y = get_calibration_float(pair.value) > 0.5f;
                if(key == "flipSubp") md.flip_subpixel = get_calibration_float(pair.value) > 0.5f;
                if(key == "fringe") md.fringe = get_calibration_float(pair.value);
                if(key == "invView") md.invert = get_calibration_float(pair.value) > 0.5f;
                if(key == "pitch") md.pitch = get_calibration_float(pair.value);
                if(key == "screenW") md.size.x = get_calibration_float(pair.value);
                if(key == "screenH") md.size.y = get_calibration_float(pair.value);
                if(key == "serial") md.serial = get_cbor_string(pair.value);
                if(key == "slope") md.slope = get_calibration_float(pair.value);
                if(key == "verticalAngle") md.vertical_angle = get_calibration_float(pair.value);
                if(key == "viewCone") md.view_cone = get_calibration_float(pair.value);
            }
        }

        if(key == "hardwareVersion")
            md.hardware_version = get_cbor_string(pair.value);
        if(key == "hwid")
            md.hardware_id = get_cbor_string(pair.value);
        if(key == "index")
            md.index = cbor_get_uint8(pair.value);
        if(key == "windowCoords")
        {
            md.window_coords = uvec2(
                cbor_get_uint16(cbor_array_get(pair.value, 0)),
                cbor_get_uint16(cbor_array_get(pair.value, 1))
            );
        }
    }
    md.corrected_pitch = md.size.x / md.dpi * md.pitch * sin(atan(fabs(md.slope)));
    md.tilt = md.size.y / (md.size.x * md.slope);
    return md;
}

void looking_glass::init_sdl()
{
    uint32_t subsystems = SDL_INIT_VIDEO|SDL_INIT_JOYSTICK|
        SDL_INIT_GAMECONTROLLER|SDL_INIT_EVENTS;
    if(SDL_Init(subsystems))
        throw std::runtime_error(SDL_GetError());

    if(opt.calibration_override)
    {
        SDL_Rect display_rect;
        SDL_GetDisplayBounds(opt.calibration_override->display_index, &display_rect);
        metadata.window_coords.x = display_rect.x;
        metadata.window_coords.y = display_rect.y;
    }

    win = SDL_CreateWindow(
        "Tauray",
        metadata.window_coords.x,
        metadata.window_coords.y,
        metadata.size.x,
        metadata.size.y,
        SDL_WINDOW_VULKAN | SDL_WINDOW_BORDERLESS
    );
    if(!win) throw std::runtime_error(SDL_GetError());
    SDL_SetWindowGrab(win, (SDL_bool)true);
    SDL_SetRelativeMouseMode((SDL_bool)true);
    image_size = opt.viewport_size;
    image_array_layers = opt.viewport_count;

    unsigned count = 0;
    if(!SDL_Vulkan_GetInstanceExtensions(win, &count, nullptr))
        throw std::runtime_error(SDL_GetError());

    extensions.resize(count);
    if(!SDL_Vulkan_GetInstanceExtensions(win, &count, extensions.data()))
        throw std::runtime_error(SDL_GetError());
}

void looking_glass::deinit_sdl()
{
    SDL_DestroyWindow(win);
    SDL_Quit();
}

void looking_glass::init_swapchain()
{
    device& dev_data = get_display_device();
    std::vector<vk::SurfaceFormatKHR> formats =
        dev_data.physical.getSurfaceFormatsKHR(surface);

    // Find the format matching our desired format.
    bool found_format = false;
    vk::SurfaceFormatKHR swapchain_format = formats[0];
    for(vk::SurfaceFormatKHR& format: formats)
    {
        if(
            format.format == vk::Format::eB8G8R8A8Unorm &&
            format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear
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
    expected_image_layout = vk::ImageLayout::eGeneral;

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
        TR_WARN(
            "Could not find desired present mode, falling back to first "
            "available mode."
        );

    // Find the size that matches our looking_glass size
    vk::SurfaceCapabilitiesKHR caps =
        dev_data.physical.getSurfaceCapabilitiesKHR(surface);
    vk::Extent2D selected_extent = caps.currentExtent;
    if(caps.currentExtent.width == UINT32_MAX)
    {
        uvec2 clamped_size = clamp(
            metadata.size,
            uvec2(caps.minImageExtent.width, caps.minImageExtent.height),
            uvec2(caps.maxImageExtent.width, caps.maxImageExtent.height)
        );
        selected_extent.width = clamped_size.x;
        selected_extent.height = clamped_size.y;
    }
    if(
        selected_extent.width != metadata.size.x ||
        selected_extent.height != metadata.size.y
    ) throw std::runtime_error(
        "Could not find swap chain extent matching the looking_glass size!"
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

    // Get swap chain images & create image views
    auto swapchain_images = dev_data.logical.getSwapchainImagesKHR(swapchain);
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
    }
    vk::ImageCreateInfo info{
        {},
        vk::ImageType::e2D,
        swapchain_format.format,
        {opt.viewport_size.x, opt.viewport_size.y, 1},
        1,
        opt.viewport_count,
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
}

void looking_glass::deinit_swapchain()
{
    vk::Device& dev = get_display_device().logical;
    array_image_views.clear();
    images.clear();
    window_images.clear();
    window_image_views.clear();
    sync();
    dev.destroySwapchainKHR(swapchain);
}

void looking_glass::init_render_target()
{
    render_target input = get_array_render_target()[0];
    input.layout = expected_image_layout;

    std::vector<render_target> output_frames;
    for(size_t i = 0; i < window_images.size(); ++i)
    {
        output_frames.emplace_back(
            metadata.size, 0, 1,
            window_images[i],
            window_image_views[i],
            vk::ImageLayout::eUndefined,
            image_format,
            vk::SampleCountFlagBits::e1
        );
    }

    try
    {
        composition.reset(new looking_glass_composition_stage(
            get_display_device(),
            input,
            output_frames,
            {
                opt.viewport_count,
                metadata.corrected_pitch,
                metadata.tilt,
                metadata.center,
                metadata.invert
            }
        ));
    } catch(std::runtime_error& err)
    {
        TR_ERR(err.what());
        std::abort();
    }
}

void looking_glass::deinit_render_target()
{
    composition.reset();
}

}


