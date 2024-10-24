#include "openxr.hh"
#include "camera.hh"
#include "misc.hh"
#include "log.hh"
#include <iostream>
#include <thread>

#define DEFINE_XR_EXT_FUNCTIONS\
    DEFINE_XR_EXT_FUNCTION(xrCreateDebugUtilsMessengerEXT) \
    DEFINE_XR_EXT_FUNCTION(xrDestroyDebugUtilsMessengerEXT) \
    DEFINE_XR_EXT_FUNCTION(xrGetVulkanGraphicsRequirements2KHR) \
    DEFINE_XR_EXT_FUNCTION(xrCreateVulkanInstanceKHR) \
    DEFINE_XR_EXT_FUNCTION(xrGetVulkanGraphicsDevice2KHR) \
    DEFINE_XR_EXT_FUNCTION(xrCreateVulkanDeviceKHR)

namespace
{

#define DEFINE_XR_EXT_FUNCTION(name) \
    PFN_##name name = nullptr;
DEFINE_XR_EXT_FUNCTIONS
#undef DEFINE_XR_EXT_FUNCTION

void load_extension_functions(XrInstance instance)
{
    // Damn I hate how Vulkan and OpenXR extension loading works... They
    // bothered to write a loader but then just don't load extensions so that
    // we have to do it manually like this??? As of writing, there's nothing
    // like Volk/GLEW for OpenXR and openxr.hpp is completely borked.

#define DEFINE_XR_EXT_FUNCTION(name) \
    xrGetInstanceProcAddr(instance, #name, (PFN_xrVoidFunction*)&name);
DEFINE_XR_EXT_FUNCTIONS
#undef DEFINE_XR_EXT_FUNCTION
}

bool has_extension(
    const std::vector<XrExtensionProperties>& extensions,
    const std::string& name
){
    for(XrExtensionProperties xr: extensions)
    {
        if(xr.extensionName == name)
            return true;
    }
    return false;
}


XRAPI_ATTR XrBool32 XRAPI_CALL debug_callback(
    XrDebugUtilsMessageSeverityFlagsEXT severity,
    XrDebugUtilsMessageTypeFlagsEXT type,
    const XrDebugUtilsMessengerCallbackDataEXT* data,
    void* pUserData
){
    // These are usually spammy and useless messages.
    if(type == VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT)
        return false;
    (void)severity;
    (void)type;
    (void)pUserData;
    TR_ERR(data->message);

    // Handy assert for debugging where validation errors happen
    //assert(false);
    return false;
}

bool session_is_ready(XrSessionState state)
{
    return state == XR_SESSION_STATE_READY ||
        state == XR_SESSION_STATE_FOCUSED ||
        state == XR_SESSION_STATE_SYNCHRONIZED ||
        state == XR_SESSION_STATE_VISIBLE;
}

}

namespace tr
{

openxr::openxr(const options& opt)
:   context(opt), opt(opt), xr_device(VK_NULL_HANDLE)
{
    if(opt.preview_window)
        init_sdl();
    init_xr();
    init_vulkan(vkGetInstanceProcAddr);
    if(opt.preview_window)
    {
        if(!SDL_Vulkan_CreateSurface(win, instance, &surface))
            throw std::runtime_error(SDL_GetError());
    }
    init_devices();
    init_session();
    init_xr_swapchain();
    if(opt.preview_window)
        init_window_swapchain();
    init_resources();
    init_local_resources();
}

openxr::~openxr()
{
    deinit_local_resources();
    deinit_resources();
    if(opt.preview_window)
        deinit_window_swapchain();
    deinit_xr_swapchain();
    deinit_session();
    deinit_xr();
    deinit_devices();
    if(opt.preview_window)
        vkDestroySurfaceKHR(instance, surface, nullptr);
    deinit_vulkan();
    if(opt.preview_window)
        deinit_sdl();
}

bool openxr::init_frame()
{
    // TODO: What else could we do?
    while(!session_is_ready(session_state))
    {
        if(poll())
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if(poll()) return true;

    XrFrameWaitInfo frame_info = {
        XR_TYPE_FRAME_WAIT_INFO,
        nullptr
    };
    frame_state.type = XR_TYPE_FRAME_STATE;
    frame_state.next = nullptr;
    xrWaitFrame(xr_session, &frame_info, &frame_state);

    XrFrameBeginInfo begin_info = {
        XR_TYPE_FRAME_BEGIN_INFO,
        nullptr
    };
    xrBeginFrame(xr_session, &begin_info);

    update_xr_views();
    update_xr_controllers();

    return false;
}

void openxr::recreate_swapchains()
{
    device& dev_data = get_display_device();
    dev_data.logical.waitIdle();

    if(opt.preview_window)
    {
        deinit_window_swapchain();
        init_window_swapchain();
    }
}

void openxr::setup_xr_surroundings(
    scene& s, transformable* reference_frame
){
    s.foreach([&](entity id, camera&){s.remove<camera>(id);});
    cameras.clear();
    for(size_t i = 0; i < view_states.size(); ++i)
    {
        camera cam;
        float aspect = image_size.x/(float)image_size.y;
        cam.perspective(90, aspect, 0.1f, 300.0f);
        entity id = s.add(
            std::move(cam),
            transformable(reference_frame),
            camera_metadata{true, int(i), true}
        );
        cameras.push_back(s.get<camera>(id));
        camera_transforms.push_back(s.get<transformable>(id));
    }

    for(size_t i = 0; i < 2; ++i)
    {
        entity id = s.add(
            transformable(reference_frame),
            openxr_controller{i == 0, false, false, false}
        );
        controllers.push_back(s.get<openxr_controller>(id));
        controller_transforms.push_back(s.get<transformable>(id));
    }
}

uint32_t openxr::prepare_next_image(uint32_t frame_index)
{
    device& d = get_display_device();

    uint32_t swapchain_index = 0;
    XrSwapchainImageAcquireInfo acquire_info = {
        XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO,
        nullptr
    };
    xrAcquireSwapchainImage(xr_swapchain, &acquire_info, &swapchain_index);

    // TODO: Why is this not a semaphore and how can we make it a semaphore?
    XrSwapchainImageWaitInfo wait_info = {
        XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO,
        nullptr,
        INT64_MAX
    };
    xrWaitSwapchainImage(xr_swapchain, &wait_info);

    // Signal the semaphore manually.
    d.graphics_queue.submit(
        vk::SubmitInfo(
            0, nullptr, nullptr, 0, nullptr,
            1, frame_available[frame_index]
        ),
        {}
    );

    if(opt.preview_window)
    {
        window_swapchain_index = d.logical.acquireNextImageKHR(
            window_swapchain, UINT64_MAX, window_frame_available[frame_index], {}
        ).value;
    }

    return swapchain_index;
}

void openxr::finish_image(uint32_t frame_index, uint32_t swapchain_index, bool)
{
    blit_images(frame_index, swapchain_index);

    device& d = get_display_device();
    if(opt.preview_window)
    {
        (void)d.present_queue.presentKHR({
            1, window_frame_finished[frame_index],
            1, &window_swapchain,
            &window_swapchain_index
        });
    }

    // TODO: Do we just wait for it orrr...??? Why can't OpenXR just take
    // semaphores...
    /*
    if(!opt.preview_window)
    {
        (void)d.logical.waitForFences(*finish_fence, true, UINT64_MAX);
        d.logical.resetFences(*finish_fence);
    }
    */

    XrSwapchainImageReleaseInfo release_info = {
        XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO,
        nullptr
    };
    xrReleaseSwapchainImage(xr_swapchain, &release_info);

    XrFrameEndInfo end_info = {
        XR_TYPE_FRAME_END_INFO,
        nullptr,
        frame_state.predictedDisplayTime,
        XR_ENVIRONMENT_BLEND_MODE_OPAQUE,
        (uint32_t)projection_layer_headers.size(),
        projection_layer_headers.data()
    };
    xrEndFrame(xr_session, &end_info);
}

bool openxr::queue_can_present(
    const vk::PhysicalDevice& device,
    uint32_t queue_index,
    const vk::QueueFamilyProperties& props
){
    if(props.queueFlags & vk::QueueFlagBits::eGraphics)
    {
        if(opt.preview_window)
        {
            return
                (VkPhysicalDevice)device == get_xr_device() &&
                device.getSurfaceSupportKHR(queue_index, vk::SurfaceKHR(surface)) &&
                device.getSurfaceFormatsKHR(surface).size() > 0 &&
                device.getSurfacePresentModesKHR(surface).size() > 0;
        }
        else return (VkPhysicalDevice)device == get_xr_device();
    }
    return false;
}

vk::Instance openxr::create_instance(
    const vk::InstanceCreateInfo& info,
    PFN_vkGetInstanceProcAddr getInstanceProcAddr
){
    XrVulkanInstanceCreateInfoKHR create_info = {
        XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR,
        nullptr,
        system_id,
        0,
        getInstanceProcAddr,
        &((const VkInstanceCreateInfo&)info),
        nullptr
    };
    VkInstance instance;
    VkResult vk_res;
    XrResult xr_res = xrCreateVulkanInstanceKHR(
        xr_instance, &create_info, &instance, &vk_res
    );
    if(vk_res != VK_SUCCESS || xr_res != XR_SUCCESS)
        throw std::runtime_error("Failed to create Vulkan instance for XR");
    return instance;
}

vk::Device openxr::create_device(
    const vk::PhysicalDevice& device,
    const vk::DeviceCreateInfo& info
){
    if((VkPhysicalDevice)device == xr_device)
    {
        XrVulkanDeviceCreateInfoKHR create_info = {
            XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR,
            nullptr,
            system_id,
            0,
            opt.preview_window ? (PFN_vkGetInstanceProcAddr)SDL_Vulkan_GetVkGetInstanceProcAddr() : vkGetInstanceProcAddr,
            xr_device,
            &(const VkDeviceCreateInfo&)info,
            nullptr
        };
        VkDevice dev;
        VkResult vk_res;
        XrResult xr_res = xrCreateVulkanDeviceKHR(
            xr_instance, &create_info, &dev, &vk_res
        );
        if(vk_res != VK_SUCCESS || xr_res != XR_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan device for XR");
        return dev;
    }
    else return device.createDevice(info);
}

void openxr::init_sdl()
{
    uint32_t subsystems = SDL_INIT_VIDEO|SDL_INIT_JOYSTICK|
        SDL_INIT_GAMECONTROLLER|SDL_INIT_EVENTS;
    if(SDL_Init(subsystems))
        throw std::runtime_error(SDL_GetError());

    if(opt.preview_window)
    {
        win = SDL_CreateWindow(
            "Tauray",
            SDL_WINDOWPOS_UNDEFINED,
            SDL_WINDOWPOS_UNDEFINED,
            opt.size.x,
            opt.size.y,
            SDL_WINDOW_VULKAN | (opt.fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0)
        );
        if(!win) throw std::runtime_error(SDL_GetError());
        SDL_GetWindowSize(win, (int*)&opt.size.x, (int*)&opt.size.y);
        SDL_SetWindowGrab(win, (SDL_bool)true);
        SDL_SetRelativeMouseMode((SDL_bool)true);

        unsigned count = 0;
        if(!SDL_Vulkan_GetInstanceExtensions(win, &count, nullptr))
            throw std::runtime_error(SDL_GetError());

        extensions.resize(count);
        if(!SDL_Vulkan_GetInstanceExtensions(win, &count, extensions.data()))
            throw std::runtime_error(SDL_GetError());
    }
}

void openxr::deinit_sdl()
{
    if(opt.preview_window)
        SDL_DestroyWindow(win);
    SDL_Quit();
}

void openxr::init_xr()
{
    uint32_t extension_count = 0;
    xrEnumerateInstanceExtensionProperties(
        nullptr, 0, &extension_count, nullptr
    );
    std::vector<XrExtensionProperties> available_extensions(
        extension_count,
        XrExtensionProperties{
            XR_TYPE_EXTENSION_PROPERTIES,
            nullptr,
            {},
            0
        }
    );
    xrEnumerateInstanceExtensionProperties(
        nullptr,
        available_extensions.size(),
        &extension_count,
        available_extensions.data()
    );

    if(!has_extension(available_extensions, XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME))
    {
        throw std::runtime_error(
            "XR_KHR_vulkan_enable2 not supported, but required for XR!"
        );
    }

    uint32_t layer_count = 0;
    xrEnumerateApiLayerProperties(0, &layer_count, nullptr);
    std::vector<XrApiLayerProperties> available_layers(
        layer_count,
        XrApiLayerProperties{
            XR_TYPE_API_LAYER_PROPERTIES, nullptr, {}, 0, 0, {}
        }
    );
    xrEnumerateApiLayerProperties(
        (uint32_t)available_layers.size(), &layer_count, available_layers.data()
    );
    std::vector<const char*> enabled_layers;

    std::vector<const char*> enabled_extensions = {
        XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME,
    };
    if(opt.enable_vulkan_validation)
    {
        enabled_extensions.push_back(XR_EXT_DEBUG_UTILS_EXTENSION_NAME);
        for(XrApiLayerProperties& props: available_layers)
        {
            if(strcmp(props.layerName, "XR_APILAYER_LUNARG_core_validation") == 0)
                enabled_layers.push_back("XR_APILAYER_LUNARG_core_validation");
        }
    }

    XrInstanceCreateInfo xr_info = {
        XR_TYPE_INSTANCE_CREATE_INFO,
        nullptr,
        0,
        {
            "Tauray",
            XR_MAKE_VERSION(0,0,1),
            "Tauray",
            XR_MAKE_VERSION(0,0,1),
            XR_CURRENT_API_VERSION
        },
        (uint32_t)enabled_layers.size(),
        enabled_layers.data(),
        (uint32_t)enabled_extensions.size(),
        enabled_extensions.data()
    };

    XrResult res = xrCreateInstance(&xr_info, &xr_instance);
    if(res != XR_SUCCESS)
        throw std::runtime_error("Failed to init XR");

    load_extension_functions(xr_instance);

    if(opt.enable_vulkan_validation)
    {
        XrDebugUtilsMessengerCreateInfoEXT messenger_info = {
            XR_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            nullptr,
            XR_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT|
            XR_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT|
            XR_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT|
            XR_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            XR_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT|
            XR_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT|
            XR_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT|
            XR_DEBUG_UTILS_MESSAGE_TYPE_CONFORMANCE_BIT_EXT,
            debug_callback,
            nullptr
        };
        xrCreateDebugUtilsMessengerEXT(xr_instance, &messenger_info, &messenger);
    }

    XrInstanceProperties xr_instance_props = {
        XR_TYPE_INSTANCE_PROPERTIES, nullptr, 0, {}
    };
    xrGetInstanceProperties(xr_instance, &xr_instance_props);

    TR_LOG("OpenXR runtime: ", xr_instance_props.runtimeName);

    XrSystemGetInfo system_info {
        XR_TYPE_SYSTEM_GET_INFO,
        nullptr,
        XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY
    };
    xrGetSystem(xr_instance, &system_info, &system_id);

    XrSystemProperties system_props;
    system_props.type = XR_TYPE_SYSTEM_PROPERTIES;
    system_props.next = nullptr;
    xrGetSystemProperties(xr_instance, system_id, &system_props);

    TR_LOG("OpenXR system: ", system_props.systemName);

    uint32_t view_config_count = 0;
    xrEnumerateViewConfigurations(xr_instance, system_id, 0, &view_config_count, nullptr);
    std::vector<XrViewConfigurationType> view_config_types(view_config_count);
    xrEnumerateViewConfigurations(
        xr_instance, system_id,
        view_config_count, &view_config_count, view_config_types.data()
    );

    bool found_view_config = false;
    for(XrViewConfigurationType type: view_config_types)
    {
        if(type == XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO)
        {
            view_config = type;
            found_view_config = true;
            break;
        }
    }
    if(!found_view_config)
        throw std::runtime_error(
            "Failed to find a suitable XR view configuration"
        );

    uint32_t view_config_view_count = 0;
    xrEnumerateViewConfigurationViews(
        xr_instance, system_id, view_config, 0, &view_config_view_count,
        nullptr
    );
    std::vector<XrViewConfigurationView> views(view_config_view_count, {
        XR_TYPE_VIEW_CONFIGURATION_VIEW, nullptr, 0, 0, 0, 0, 0, 0
    });
    xrEnumerateViewConfigurationViews(
        xr_instance, system_id, view_config,
        view_config_view_count, &view_config_view_count, views.data()
    );
    image_size = uvec2(0);

    for(XrViewConfigurationView& view: views)
    {
        if(
            view.recommendedImageRectWidth != views[0].recommendedImageRectWidth ||
            view.recommendedImageRectHeight != views[0].recommendedImageRectHeight
        ) throw std::runtime_error(
            "Currently, all views must have the same resolution in Tauray."
        );
        image_size.x = view.recommendedImageRectWidth;
        image_size.y = view.recommendedImageRectHeight;
    }

    image_array_layers = views.size();

    view_states.resize(views.size(), {XR_TYPE_VIEW, nullptr, {}, {}});

    XrGraphicsRequirementsVulkan2KHR graphics_requirements = {
        XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR, nullptr, 0, 0
    };
    xrGetVulkanGraphicsRequirements2KHR(
        xr_instance, system_id, &graphics_requirements
    );
    if(XR_MAKE_VERSION(1,2,0) < graphics_requirements.minApiVersionSupported)
        throw std::runtime_error("XR system requires newer Vulkan than 1.2!");
}

void openxr::deinit_xr()
{
    if(opt.enable_vulkan_validation)
    {
        xrDestroyDebugUtilsMessengerEXT(messenger);
    }
    xrDestroyInstance(xr_instance);
}

void openxr::init_session()
{
    device& dev_data = get_display_device();
    XrGraphicsBindingVulkan2KHR binding = {
        XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR,
        nullptr,
        instance,
        dev_data.physical,
        dev_data.logical,
        dev_data.present_family_index,
        0
    };
    XrSessionCreateInfo session_info = {
        XR_TYPE_SESSION_CREATE_INFO,
        &binding,
        0,
        system_id
    };
    xrCreateSession(xr_instance, &session_info, &xr_session);

    reference_space_type = XR_REFERENCE_SPACE_TYPE_LOCAL;
    uint32_t space_count = 0;
    xrEnumerateReferenceSpaces(xr_session, 0, &space_count, nullptr);
    std::vector<XrReferenceSpaceType> spaces(space_count);
    xrEnumerateReferenceSpaces(xr_session, space_count, &space_count, spaces.data());

    for(XrReferenceSpaceType type: spaces)
    {
        if(type == XR_REFERENCE_SPACE_TYPE_STAGE)
        {
            reference_space_type = type;
            break;
        }
    }

    XrReferenceSpaceCreateInfo space_info = {
        XR_TYPE_REFERENCE_SPACE_CREATE_INFO,
        nullptr,
        reference_space_type,
        {
            {0,0,0,1},
            {0,0,0}
        }
    };
    xrCreateReferenceSpace(xr_session, &space_info, &xr_reference_space);

    session_state = XR_SESSION_STATE_UNKNOWN;

    XrActionSetCreateInfo as_info{
        XR_TYPE_ACTION_SET_CREATE_INFO,
        nullptr,
        "gameplay",
        "Gameplay",
        0
    };
    xrCreateActionSet(xr_instance, &as_info, &action_set);

    XrActionCreateInfo grip_left_info{
        XR_TYPE_ACTION_CREATE_INFO,
        nullptr,
        "leftcontrollerorientation",
        XR_ACTION_TYPE_POSE_INPUT,
        0, nullptr,
        "Left controller orientation"
    };
    xrCreateAction(action_set, &grip_left_info, &grip_pose_action[0]);

    XrActionCreateInfo grip_right_info{
        XR_TYPE_ACTION_CREATE_INFO,
        nullptr,
        "rightcontrollerorientation",
        XR_ACTION_TYPE_POSE_INPUT,
        0, nullptr,
        "Right controller orientation"
    };
    xrCreateAction(action_set, &grip_right_info, &grip_pose_action[1]);

    XrActionCreateInfo click_left_info{
        XR_TYPE_ACTION_CREATE_INFO,
        nullptr,
        "leftcontrollerclick",
        XR_ACTION_TYPE_BOOLEAN_INPUT,
        0, nullptr,
        "Left controller click"
    };
    xrCreateAction(action_set, &click_left_info, &click_action[0]);

    XrActionCreateInfo click_right_info{
        XR_TYPE_ACTION_CREATE_INFO,
        nullptr,
        "rightcontrollerclick",
        XR_ACTION_TYPE_BOOLEAN_INPUT,
        0, nullptr,
        "Right controller click"
    };
    xrCreateAction(action_set, &click_right_info, &click_action[1]);

    XrPath left_grip_path;
    xrStringToPath(xr_instance, "/user/hand/left/input/grip/pose", &left_grip_path);
    XrPath right_grip_path;
    xrStringToPath(xr_instance, "/user/hand/right/input/grip/pose", &right_grip_path);
    XrPath left_click_path;
    xrStringToPath(xr_instance, "/user/hand/left/input/select/click", &left_click_path);
    XrPath right_click_path;
    xrStringToPath(xr_instance, "/user/hand/right/input/select/click", &right_click_path);

    XrPath interaction_profile_path;
    xrStringToPath(xr_instance, "/interaction_profiles/khr/simple_controller", &interaction_profile_path);

    XrActionSuggestedBinding bindings[4];
    bindings[0].action = grip_pose_action[0];
    bindings[0].binding = left_grip_path;
    bindings[1].action = grip_pose_action[1];
    bindings[1].binding = right_grip_path;
    bindings[2].action = click_action[0];
    bindings[2].binding = left_click_path;
    bindings[3].action = click_action[1];
    bindings[3].binding = right_click_path;

    XrInteractionProfileSuggestedBinding suggested_bindings{
        XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING,
        nullptr,
        interaction_profile_path,
        std::size(bindings),
        bindings
    };

    xrSuggestInteractionProfileBindings(xr_instance, &suggested_bindings);

    XrSessionActionSetsAttachInfo attach_info{
        XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO,
        nullptr,
        1,
        &action_set
    };
    xrAttachSessionActionSets(xr_session, &attach_info);

    const XrPosef identity = {{0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}};

    XrActionSpaceCreateInfo actionSpaceCI{};
    actionSpaceCI.type = XR_TYPE_ACTION_SPACE_CREATE_INFO;
    actionSpaceCI.action = grip_pose_action[0];
    actionSpaceCI.poseInActionSpace = identity;
    actionSpaceCI.subactionPath = XR_NULL_PATH;
    xrCreateActionSpace(xr_session, &actionSpaceCI, &grip_pose_space[0]);

    actionSpaceCI.action = grip_pose_action[1];
    xrCreateActionSpace(xr_session, &actionSpaceCI, &grip_pose_space[1]);
}

void openxr::deinit_session()
{
    xrDestroyAction(click_action[0]);
    xrDestroyAction(click_action[1]);
    xrDestroyAction(grip_pose_action[0]);
    xrDestroyAction(grip_pose_action[1]);
    xrDestroyActionSet(action_set);
    xrDestroySpace(xr_reference_space);
    xrDestroySession(xr_session);
}

void openxr::init_xr_swapchain()
{
    device& dev_data = get_display_device();

    uint32_t format_count = 0;
    xrEnumerateSwapchainFormats(xr_session, 0, &format_count, nullptr);
    std::vector<int64_t> formats(format_count);
    xrEnumerateSwapchainFormats(
        xr_session, format_count, &format_count, formats.data()
    );

    bool found_format = false;
    int64_t swapchain_format = formats[0];
    for(int64_t format: formats)
    {
        if(
            (!opt.hdr_display && format == VK_FORMAT_B8G8R8A8_UNORM) ||
            (opt.hdr_display && format == VK_FORMAT_R16G16B16A16_SFLOAT)
        ){
            swapchain_format = format;
            found_format = true;
            break;
        }
    }
    if(!found_format)
        TR_WARN(
            "Could not find any suitable swap chain format for XR!"
            "Using the first available format instead, results may look "
            "incorrect."
        );

    image_format = vk::Format((VkFormat)swapchain_format);
    expected_image_layout = vk::ImageLayout::eTransferSrcOptimal;

    XrSwapchainCreateInfo create_info = {
        XR_TYPE_SWAPCHAIN_CREATE_INFO,
        nullptr,
        0,
        XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT|XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT,
        swapchain_format,
        1,
        image_size.x,
        image_size.y,
        1,
        image_array_layers,
        1
    };

    xrCreateSwapchain(xr_session, &create_info, &xr_swapchain);

    uint32_t image_count = 0;
    xrEnumerateSwapchainImages(xr_swapchain, 0, &image_count, nullptr);
    std::vector<XrSwapchainImageVulkan2KHR> swapchain_images(
        image_count,
        {
            XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR,
            nullptr,
            VK_NULL_HANDLE
        }
    );
    xrEnumerateSwapchainImages(
        xr_swapchain,
        image_count,
        &image_count,
        (XrSwapchainImageBaseHeader*)swapchain_images.data()
    );

    images.clear();
    for(XrSwapchainImageVulkan2KHR img: swapchain_images)
    {
        vk::ImageCreateInfo info {
            {},
            vk::ImageType::e2D,
            vk::Format((VkFormat)swapchain_format),
            {image_size.x, image_size.y, 1},
            1,
            image_array_layers,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eSampled|
            vk::ImageUsageFlagBits::eStorage|
            vk::ImageUsageFlagBits::eTransferDst|
            vk::ImageUsageFlagBits::eTransferSrc,
            vk::SharingMode::eExclusive
        };
        images.emplace_back(sync_create_gpu_image(
            dev_data, info, vk::ImageLayout::eTransferDstOptimal
        ));

        xr_images.emplace_back(dev_data, vk::Image(img.image));
        xr_images.back().leak();
        xr_image_views.emplace_back(dev_data,
            dev_data.logical.createImageView({
                {},
                img.image,
                vk::ImageViewType::e2DArray,
                vk::Format((VkFormat)swapchain_format),
                {},
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, image_array_layers}
            })
        );
    }
    reset_image_views();

    projection_layer_views.resize(view_states.size());
    for(size_t i = 0; i < view_states.size(); ++i)
    {
        projection_layer_views[i] = {
            XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW,
            nullptr,
            view_states[i].pose,
            view_states[i].fov,
            {
                xr_swapchain,
                {
                    {0, 0},
                    {(int)image_size.x, (int)image_size.y}
                },
                (uint32_t)i
            }
        };
    }

    projection_layer = {
        XR_TYPE_COMPOSITION_LAYER_PROJECTION,
        nullptr,
        0,//XR_COMPOSITION_LAYER_CORRECT_CHROMATIC_ABERRATION_BIT,
        xr_reference_space,
        (uint32_t)view_states.size(),
        projection_layer_views.data()
    };
    projection_layer_headers.resize(1);
    projection_layer_headers[0] =
        (XrCompositionLayerBaseHeader*)&projection_layer;
}

void openxr::deinit_xr_swapchain()
{
    array_image_views.clear();
    images.clear();
    xr_image_views.clear();
    xr_images.clear();
    xrDestroySwapchain(xr_swapchain);
}

void openxr::init_window_swapchain()
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
            format.format == vk::Format::eB8G8R8A8Srgb &&
            format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear
        ){
            swapchain_format = format;
            found_format = true;
            break;
        }
    }
    if(!found_format)
        TR_WARN("Could not find any suitable swap chain format for preview window!");

    window_image_format = swapchain_format.format;

    std::vector<vk::PresentModeKHR> modes =
        dev_data.physical.getSurfacePresentModesKHR(surface);
    bool found_mode = false;
    vk::PresentModeKHR selected_mode = modes[0];
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
    if(!found_mode)
        TR_WARN(
            "Could not find desired present mode, falling back to first "
            "available mode."
        );

    // Find the size that matches our window size
    vk::SurfaceCapabilitiesKHR caps =
        dev_data.physical.getSurfaceCapabilitiesKHR(surface);
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
    window_swapchain = dev_data.logical.createSwapchainKHR({
        {},
        surface,
        image_count,
        swapchain_format.format,
        swapchain_format.colorSpace,
        selected_extent,
        1,
        vk::ImageUsageFlagBits::eTransferDst|
        vk::ImageUsageFlagBits::eColorAttachment,
        sharing_mode,
        (uint32_t)queue_family_indices.size(),
        queue_family_indices.data(),
        caps.currentTransform,
        vk::CompositeAlphaFlagBitsKHR::eOpaque,
        selected_mode,
        true
    });

    // Get swap chain images & create image views
    auto swapchain_images = dev_data.logical.getSwapchainImagesKHR(
        window_swapchain
    );
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
}

void openxr::deinit_window_swapchain()
{
    vk::Device& dev = get_display_device().logical;
    window_image_views.clear();
    window_images.clear();
    sync();
    dev.destroySwapchainKHR(window_swapchain);
}

void openxr::init_local_resources()
{
    device& dev_data = get_display_device();
    if(opt.preview_window)
    {
        window_frame_available.resize(MAX_FRAMES_IN_FLIGHT);
        window_frame_finished.resize(MAX_FRAMES_IN_FLIGHT);
        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            window_frame_available[i] = create_binary_semaphore(dev_data);
            window_frame_finished[i] = create_binary_semaphore(dev_data);
        }
    }

    finish_fence = vkm(dev_data, dev_data.logical.createFence({}));
}

void openxr::deinit_local_resources()
{
    finish_fence.drop();
    window_frame_available.clear();
    window_frame_finished.clear();
}

void openxr::blit_images(uint32_t frame_index, uint32_t swapchain_index)
{
    device& d = get_display_device();
    vkm<vk::CommandBuffer> cmd = create_graphics_command_buffer(d);
    cmd->begin(vk::CommandBufferBeginInfo{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    });

    if(opt.preview_window)
    {
        transition_image_layout(
            *cmd,
            *window_images[window_swapchain_index],
            window_image_format,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal
        );
        for(size_t i = 0; i < image_array_layers; ++i)
        {
            cmd->blitImage(
                images[swapchain_index],
                vk::ImageLayout::eTransferSrcOptimal,
                window_images[window_swapchain_index],
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageBlit{
                    {vk::ImageAspectFlagBits::eColor, 0, (uint32_t)i, 1},
                    {vk::Offset3D{0,0,0}, {(int)image_size.x, (int)image_size.y, 1}},
                    {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                    {
                        vk::Offset3D{(int)(opt.size.x/image_array_layers*i),0,0},
                        {(int)(opt.size.x/image_array_layers*(i+1)), (int)opt.size.y, 1}
                    }
                },
                vk::Filter::eLinear
            );
        }

        transition_image_layout(
            *cmd,
            *window_images[window_swapchain_index],
            window_image_format,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::ePresentSrcKHR
        );
    }

    transition_image_layout(
        *cmd,
        *xr_images[swapchain_index],
        image_format,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eTransferDstOptimal
    );

    cmd->blitImage(
        images[swapchain_index],
        vk::ImageLayout::eTransferSrcOptimal,
        xr_images[swapchain_index],
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageBlit{
            {vk::ImageAspectFlagBits::eColor, 0, 0, image_array_layers},
            {vk::Offset3D{0,0,0}, {(int)image_size.x, (int)image_size.y, 1}},
            {vk::ImageAspectFlagBits::eColor, 0, 0, image_array_layers},
            {vk::Offset3D{0,0,0}, {(int)image_size.x, (int)image_size.y, 1}}
        },
        vk::Filter::eNearest
    );

    transition_image_layout(
        *cmd,
        *xr_images[swapchain_index],
        image_format,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eColorAttachmentOptimal
    );
    cmd->end();

    vk::PipelineStageFlags wait_stages[2] = {
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTopOfPipe
    };
    vk::Semaphore wait_semaphores[2] = {
        frame_finished[frame_index],
        frame_finished[frame_index]
    };
    if(opt.preview_window)
    {
        wait_semaphores[1] = window_frame_available[frame_index];
        d.graphics_queue.submit(
            vk::SubmitInfo(
                2, wait_semaphores, wait_stages,
                1, cmd.get(),
                1, window_frame_finished[frame_index]
            ),
            {}
        );
    }
    else
    {
        d.graphics_queue.submit(
            vk::SubmitInfo(
                1, wait_semaphores, wait_stages,
                1, cmd.get(),
                0, {}
            ),
            {}//finish_fence
        );
    }
}

bool openxr::poll()
{
    XrEventDataBuffer event;
    for(;;)
    {
        event.type = XR_TYPE_EVENT_DATA_BUFFER;
        event.next = nullptr;
        XrResult res = xrPollEvent(xr_instance, &event);
        if(res == XR_EVENT_UNAVAILABLE)
            break;
        if(res != XR_SUCCESS)
            throw std::runtime_error(
                "xrPollEvent failed somehow " + std::to_string(res)
            );

        switch(event.type)
        {
        case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED:
            {
                XrEventDataSessionStateChanged state =
                    reinterpret_cast<XrEventDataSessionStateChanged&>(event);
                session_state = state.state;
                if(state.state == XR_SESSION_STATE_READY)
                {
                    XrSessionBeginInfo begin_info = {
                        XR_TYPE_SESSION_BEGIN_INFO,
                        nullptr,
                        view_config
                    };
                    xrBeginSession(xr_session, &begin_info);
                    TR_LOG("XR session begin");
                }
                else if(state.state == XR_SESSION_STATE_STOPPING)
                {
                    TR_LOG("XR session end");
                    xrEndSession(xr_session);
                    return true;
                }
            }
            break;
        default:
            break;
        }
    }
    return false;
}

void openxr::update_xr_views()
{
    uint32_t count = view_states.size();
    XrViewState vs = {
        XR_TYPE_VIEW_STATE,
        nullptr,
        0
    };
    XrViewLocateInfo li = {
        XR_TYPE_VIEW_LOCATE_INFO,
        nullptr,
        view_config,
        frame_state.predictedDisplayTime,
        xr_reference_space
    };
    xrLocateViews(xr_session, &li, &vs, count, &count, view_states.data());
    assert(count == view_states.size());

    for(size_t i = 0; i < view_states.size(); ++i)
    {
        if(i >= cameras.size())
            continue;

        camera& cam = *cameras[i];
        transformable& cam_transform = *camera_transforms[i];
        const XrView& v = view_states[i];

        cam.set_fov(
            glm::degrees(v.fov.angleLeft),
            glm::degrees(v.fov.angleRight),
            glm::degrees(v.fov.angleUp),
            glm::degrees(v.fov.angleDown)
        );

        if(vs.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT)
        {
            cam_transform.set_orientation(quat(
                v.pose.orientation.w,
                v.pose.orientation.x,
                v.pose.orientation.y,
                v.pose.orientation.z
            ));
            projection_layer_views[i].pose.orientation = v.pose.orientation;
        }

        if(vs.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT)
        {
            cam_transform.set_position(vec3(
                v.pose.position.x,
                v.pose.position.y,
                v.pose.position.z
            ));
            projection_layer_views[i].pose.position = v.pose.position;
        }

        projection_layer_views[i].fov = v.fov;
    }
}

void openxr::update_xr_controllers()
{
    XrActiveActionSet active_set{action_set, XR_NULL_PATH};
    XrActionsSyncInfo sync_info{
        XR_TYPE_ACTIONS_SYNC_INFO,
        nullptr, 1, &active_set
    };
    xrSyncActions(xr_session, &sync_info);

    XrActionStatePose grip_states[2] = {};
    grip_states[0].type = XR_TYPE_ACTION_STATE_POSE;
    grip_states[1].type = XR_TYPE_ACTION_STATE_POSE;

    XrActionStateBoolean click_states[2] = {};
    click_states[0].type = XR_TYPE_ACTION_STATE_BOOLEAN;
    click_states[1].type = XR_TYPE_ACTION_STATE_BOOLEAN;

    XrActionStateGetInfo get_info{
        XR_TYPE_ACTION_STATE_GET_INFO,
        nullptr,
        grip_pose_action[0],
        XR_NULL_PATH
    };
    xrGetActionStatePose(xr_session, &get_info, &grip_states[0]);
    get_info.action = grip_pose_action[1];
    xrGetActionStatePose(xr_session, &get_info, &grip_states[1]);
    get_info.action = click_action[0];
    xrGetActionStateBoolean(xr_session, &get_info, &click_states[0]);
    get_info.action = click_action[1];
    xrGetActionStateBoolean(xr_session, &get_info, &click_states[1]);

    for(int i = 0; i < 2; ++i)
    {
        controllers[i]->connected = grip_states[i].isActive;
        controllers[i]->clicked = click_states[i].currentState && click_states[i].isActive && click_states[i].changedSinceLastSync;
        controllers[i]->pressed = click_states[i].currentState && click_states[i].isActive;
        XrSpaceLocation space_location{};
        space_location.type = XR_TYPE_SPACE_LOCATION;
        XrResult res = xrLocateSpace(
            grip_pose_space[i],
            xr_reference_space,
            frame_state.predictedDisplayTime,
            &space_location
        );
        transformable& ct = *controller_transforms[i];
        if(
            XR_UNQUALIFIED_SUCCESS(res) &&
            (space_location.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0 &&
            (space_location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0
        ){
            ct.set_orientation(quat(
                space_location.pose.orientation.w,
                space_location.pose.orientation.x,
                space_location.pose.orientation.y,
                space_location.pose.orientation.z
            ));
            ct.set_position(vec3(
                space_location.pose.position.x,
                space_location.pose.position.y,
                space_location.pose.position.z
            ));
        }
        else controllers[i]->connected = false;
    }
}

VkPhysicalDevice openxr::get_xr_device()
{
    if(xr_device) return xr_device;
    XrVulkanGraphicsDeviceGetInfoKHR get_info = {
        XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR,
        nullptr,
        system_id,
        instance
    };
    xrGetVulkanGraphicsDevice2KHR(xr_instance, &get_info, &xr_device);
    return xr_device;
}

}
