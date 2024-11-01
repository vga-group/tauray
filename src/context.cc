#include "context.hh"
#include "placeholders.hh"
#include "misc.hh"
#include "log.hh"
#include "radix_sort/radix_sort_vk.h"
#include <iostream>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace
{

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* pUserData
){
    // These are usually spammy and useless messages.
    if(type == VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT)
        return false;
    if(uint32_t(data->messageIdNumber) == 0x912ddde2u) // FIXME: Timer ID error on windows
        return false;
    if(uint32_t(data->messageIdNumber) == 0x211e533bu) // Caused by Monado OpenXR driver
        return false;

    (void)severity;
    (void)type;
    (void)pUserData;
    TR_ERR(data->pMessage);

    // Handy assert for debugging where validation errors happen
    assert(severity != VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
    return false;
}

bool has_extension(
    const char* extension,
    const std::vector<const char*>& available_extensions
){
    for(auto& available_extension: available_extensions)
        if(!strcmp(available_extension, extension))
            return true;
    return false;
}

bool has_extension(
    const char* extension,
    const std::vector<vk::ExtensionProperties>& available_extensions
){
    for(auto& available_extension: available_extensions)
        if(!strcmp(available_extension.extensionName, extension))
            return true;
    return false;
}

bool has_extensions(
    const std::vector<const char*>& required_extensions,
    const std::vector<vk::ExtensionProperties>& available_extensions
){
    for(const char* required_extension: required_extensions)
        if(!has_extension(required_extension, available_extensions))
            return false;
    return true;
}

}

namespace tr
{

context::context(const options& opt)
:   image_array_layers(0), opt(opt), frame_counter(0),
    displayed_frame_counter(0), swapchain_index(0), frame_index(0),
    is_displaying(true), timing(this), tracker(this)
{}

context::~context() {}

bool context::init_frame() { return false; }

device& context::get_display_device()
{
    return devices[display_device_index];
}

std::vector<device>& context::get_devices()
{
    return devices;
}

uvec2 context::get_size() const
{
    return image_size;
}

vk::Format context::get_display_format() const
{
    return image_format;
}

vk::ImageLayout context::get_expected_display_layout() const
{
    return expected_image_layout;
}

void context::set_displaying(bool displaying)
{
    this->is_displaying = displaying;
}

bool context::get_displaying() const
{
    return is_displaying;
}

size_t context::get_display_count() const
{
    return image_array_layers;
}

size_t context::get_swapchain_image_count() const
{
    return images.size();
}

std::vector<render_target> context::get_array_render_target()
{
    std::vector<render_target> frames;
    for(size_t i = 0; i < get_swapchain_image_count(); ++i)
    {
        frames.emplace_back(
            image_size,
            0, image_array_layers,
            images[i],
            array_image_views[i],
            vk::ImageLayout::eUndefined,
            image_format,
            vk::SampleCountFlagBits::e1
        );
    }
    return frames;
}

placeholders& context::get_placeholders()
{
    return *placeholder_data;
}

bool context::is_ray_tracing_supported() const
{
    return !opt.disable_ray_tracing;
}

dependency context::begin_frame()
{
    frame_index = frame_counter % MAX_FRAMES_IN_FLIGHT;
    frame_counter++;

    timing.host_wait();
    device& d = get_display_device();
    (void)d.logical.waitForFences(*frame_fences[frame_index], true, UINT64_MAX);

    // Get new images
    swapchain_index = prepare_next_image(frame_index);

    // This annoying thing exists so that we can get the semaphore in the
    // position referenced by the image index (that makes things less
    // complicated when working with pipelines)
    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTopOfPipe;
    vk::SubmitInfo submit_info(
        image_array_layers != 0 ? 1 : 0,
        frame_available[frame_index].get(),
        &wait_stage,
        0, nullptr,
        1, image_available[swapchain_index]
    );
    vk::TimelineSemaphoreSubmitInfo timeline_submit_info(
        0, nullptr,
        1, &frame_counter
    );
    submit_info.pNext = (void*)&timeline_submit_info;
    d.graphics_queue.submit(submit_info, {});

    if(image_fences[swapchain_index])
        (void)d.logical.waitForFences(image_fences[swapchain_index], true, UINT64_MAX);
    image_fences[swapchain_index] = frame_fences[frame_index];

    d.logical.resetFences(*frame_fences[frame_index]);

    call_frame_end_actions(frame_index);

    if(frame_counter > MAX_FRAMES_IN_FLIGHT)
        timing.device_finish_frame();
    timing.begin_frame();

    return {d.id, *image_available[swapchain_index], frame_counter};
}

void context::end_frame(const dependencies& deps)
{
    dependencies local_deps = fill_end_frame_dependencies(deps);

    device& d = get_display_device();

    std::vector<vk::PipelineStageFlags> wait_stages(
        local_deps.size(d.id), vk::PipelineStageFlagBits::eTopOfPipe
    );

    vk::TimelineSemaphoreSubmitInfo timeline_info = local_deps.get_timeline_info(d.id);
    vk::SubmitInfo submit_info = local_deps.get_submit_info(d.id, timeline_info);
    submit_info.signalSemaphoreCount = image_array_layers != 0 ? 1 : 0;
    submit_info.pSignalSemaphores = frame_finished[frame_index];

    d.graphics_queue.submit(submit_info, frame_fences[frame_index]);

    finish_image(frame_index, swapchain_index, is_displaying);
    if(is_displaying) displayed_frame_counter++;
}

void context::get_indices(uint32_t& swapchain_index, uint32_t& frame_index) const
{
    swapchain_index = this->swapchain_index;
    frame_index = this->frame_index;
}

uint32_t context::get_frame_counter() const
{
    return frame_counter;
}

uint32_t context::get_displayed_frame_counter() const
{
    return displayed_frame_counter;
}

void context::sync()
{
    for(device& dev: devices)
        dev.logical.waitIdle();

    // No frames can be in flight anymore, so we can safely call all frame end
    // actions.
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        call_frame_end_actions(i);
}

tracing_record& context::get_timing()
{
    return timing;
}

progress_tracker& context::get_progress_tracker()
{
    return tracker;
}

void context::queue_frame_finish_callback(std::function<void()>&& func)
{
    frame_end_actions[frame_index].emplace_back(std::move(func));
}

vk::Instance context::create_instance(
    const vk::InstanceCreateInfo& info,
    PFN_vkGetInstanceProcAddr
){
    return vk::createInstance({info}, nullptr, vk::DispatchLoaderStatic());
}

vk::Device context::create_device(
    const vk::PhysicalDevice& device,
    const vk::DeviceCreateInfo& info
){
    return device.createDevice(info);
}

void context::init_vulkan(PFN_vkGetInstanceProcAddr getInstanceProcAddr)
{
    if(opt.enable_vulkan_validation)
    {
        validation_layers.push_back("VK_LAYER_KHRONOS_validation");
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        std::vector<vk::LayerProperties> available_layers =
            vk::enumerateInstanceLayerProperties(vk::DispatchLoaderStatic());

        for(
            auto it = validation_layers.begin();
            it != validation_layers.end();
        ){
            bool found = false;
            for(auto& available: available_layers)
            {
                if(!strcmp(available.layerName, *it))
                {
                    found = true;
                    break;
                }
            }
            if(found) ++it;
            else
            {
                TR_WARN("Unable to find validation layer ", *it, ", skipping.");
                validation_layers.erase(it);
            }
        }
    }

    vk::ApplicationInfo app_info(
        "Tauray",
        VK_MAKE_VERSION(0,0,1),
        "Tauray",
        VK_MAKE_VERSION(0,0,1),
        VK_API_VERSION_1_2
    );
    vk::InstanceCreateInfo instance_info(
        {}, &app_info,
        (uint32_t)validation_layers.size(), validation_layers.data(),
        (uint32_t)extensions.size(), extensions.data()
    );

    VkValidationFeatureEnableEXT debugPrintfFeature[] = {VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};
    VkValidationFeaturesEXT features = {};

    if(opt.enable_vulkan_validation)
    {
        features.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
        features.disabledValidationFeatureCount = 0;
        features.enabledValidationFeatureCount = std::size(debugPrintfFeature);
        features.pDisabledValidationFeatures = nullptr;
        features.pEnabledValidationFeatures = debugPrintfFeature;
        features.pNext = nullptr;
        instance_info.pNext = &features;
    }

    instance = create_instance(instance_info, getInstanceProcAddr);

    vk::defaultDispatchLoaderDynamic.init(instance, getInstanceProcAddr);

    if(opt.enable_vulkan_validation)
    {
        debug_messenger = instance.createDebugUtilsMessengerEXT(
            vk::DebugUtilsMessengerCreateInfoEXT(
                {},
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
                vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
                debug_callback
            ),
            nullptr
        );
    }
}

void context::deinit_vulkan()
{
    if(opt.enable_vulkan_validation)
    {
        instance.destroyDebugUtilsMessengerEXT(
            debug_messenger, nullptr
        );
        validation_layers.clear();
    }
    instance.destroy();
}

void context::init_devices()
{
    std::vector<vk::PhysicalDevice> physical_devices =
        instance.enumeratePhysicalDevices();

    std::vector<const char*> required_device_extensions = {
        VK_KHR_MAINTENANCE1_EXTENSION_NAME,
        VK_KHR_MULTIVIEW_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME
    };

    if(opt.enable_vulkan_validation)
        required_device_extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);

    if(!opt.disable_ray_tracing)
    {
        required_device_extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        required_device_extensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        required_device_extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        required_device_extensions.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
        required_device_extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    }

    bool use_distribution =
        opt.physical_device_indices.size() != 1 && physical_devices.size() > 1;
    if(use_distribution)
    {
        required_device_extensions.push_back(
            VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME
        );
#ifdef WIN32
        required_device_extensions.push_back("VK_KHR_external_semaphore_win32");
#else
        required_device_extensions.push_back(
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
        );
#endif
    }

    bool display_device_set = false;
    display_device_index = 0;

    for(unsigned duplicate = 0; duplicate < max(opt.fake_device_multiplier, 1u); ++duplicate)
    for(size_t pdev_index = 0; pdev_index < physical_devices.size(); ++pdev_index)
    {
        if(
            opt.physical_device_indices.size() != 0 &&
            !opt.physical_device_indices.count(pdev_index) &&
            (!opt.physical_device_indices.count(-1) || devices.size() != 0)
        ) continue;

        vk::PhysicalDevice& physical = physical_devices[pdev_index];
        auto props_pack = physical.getProperties2<
            vk::PhysicalDeviceProperties2,
            vk::PhysicalDeviceSubgroupProperties
        >();
        vk::PhysicalDeviceProperties props = props_pack.get<vk::PhysicalDeviceProperties2>().properties;
        vk::PhysicalDeviceSubgroupProperties subgroup_props = props_pack.get<vk::PhysicalDeviceSubgroupProperties>();
        if(props.apiVersion < VK_API_VERSION_1_2)
            continue;

        auto feats_pack = physical.getFeatures2<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan11Features,
            vk::PhysicalDeviceVulkan12Features,
            // The rest are needed for ray tracing
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
            vk::PhysicalDeviceRayQueryFeaturesKHR
        >();
        auto& feats = feats_pack.get<vk::PhysicalDeviceFeatures2>();
        auto& vulkan_11_feats = feats_pack.get<vk::PhysicalDeviceVulkan11Features>();
        auto& vulkan_12_feats = feats_pack.get<vk::PhysicalDeviceVulkan12Features>();
        auto& rt_feats =
            feats_pack.get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>();
        auto& rq_feats =
            feats_pack.get<vk::PhysicalDeviceRayQueryFeaturesKHR>();
        auto& as_feats =
            feats_pack.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();

        std::vector<vk::QueueFamilyProperties> queue_family_props =
            physical.getQueueFamilyProperties();
        std::vector<vk::ExtensionProperties> available_extensions =
            physical.enumerateDeviceExtensionProperties();
        std::vector<const char*> enabled_device_extensions =
            required_device_extensions;

        // Request anisotropic filtering support
        feats.features.samplerAnisotropy = true;
        vulkan_12_feats.timelineSemaphore = true;
        vulkan_12_feats.shaderSampledImageArrayNonUniformIndexing = true;
        vulkan_11_feats.multiview = true;
        vulkan_12_feats.bufferDeviceAddress = true;

        // If we're not ray tracing, cut off the ray tracing features.
        if(opt.disable_ray_tracing)
            vulkan_12_feats.pNext = nullptr;

        device dev_data;
        for(uint32_t i = 0; i < queue_family_props.size(); ++i)
        {
            auto flags = queue_family_props[i].queueFlags;
            bool cur_has_graphics = false;
            if(flags & vk::QueueFlagBits::eGraphics)
            {
                dev_data.graphics_family_index = i;
                dev_data.has_graphics = true;
                cur_has_graphics = true;
            }

            if(flags & vk::QueueFlagBits::eCompute)
            {
                dev_data.compute_family_index = i;
                dev_data.has_compute = true;
            }

            if(
                queue_can_present(physical, i, queue_family_props[i]) &&
                (!dev_data.has_present || cur_has_graphics)
            ){
                dev_data.present_family_index = i;
                dev_data.has_present = true;
            }

            // Look for a dedicated transfer queue!
            if(
                (flags & vk::QueueFlagBits::eTransfer) && (
                    !dev_data.has_transfer ||
                    !(flags & (vk::QueueFlagBits::eGraphics|
                        vk::QueueFlagBits::eCompute))
                )
            ){
                dev_data.transfer_family_index = i;
                dev_data.has_transfer = true;
            }
        }

        if(
            dev_data.has_present &&
            has_extension(VK_KHR_SURFACE_EXTENSION_NAME, extensions) &&
            has_extension(
                VK_KHR_SWAPCHAIN_EXTENSION_NAME, available_extensions
            )
        ){
            enabled_device_extensions.push_back(
                VK_KHR_SWAPCHAIN_EXTENSION_NAME
            );
        }

        radix_sort_vk_target_t* rs_target = radix_sort_vk_target_auto_detect(
            (VkPhysicalDeviceProperties*)&props,
            (VkPhysicalDeviceSubgroupProperties*)&subgroup_props,
            2
        );
        radix_sort_vk_target_requirements_t rs_requirements = {
            0, nullptr, (VkPhysicalDeviceFeatures*)&feats.features,
            (VkPhysicalDeviceVulkan11Features*)&vulkan_11_feats,
            (VkPhysicalDeviceVulkan12Features*)&vulkan_12_feats
        };
        radix_sort_vk_target_get_requirements(rs_target, &rs_requirements);
        size_t old_length = enabled_device_extensions.size();
        enabled_device_extensions.resize(old_length + rs_requirements.ext_name_count);
        rs_requirements.ext_names = enabled_device_extensions.data() + old_length;
        radix_sort_vk_target_get_requirements(rs_target, &rs_requirements);
        free(rs_target);

        if(
            has_extensions(required_device_extensions, available_extensions) &&
            dev_data.has_graphics && dev_data.has_compute
        ){
            TR_LOG("Using device: ", props.deviceName);

            float priority = 1.0f;
            std::vector<vk::DeviceQueueCreateInfo> queue_infos = {
                // Graphics queue
                {{}, dev_data.graphics_family_index, 1, &priority}
            };
            if(
                dev_data.graphics_family_index != dev_data.compute_family_index
            ){
                queue_infos.push_back({{}, dev_data.compute_family_index, 1, &priority});
            }
            if(
                dev_data.has_present &&
                dev_data.present_family_index != dev_data.graphics_family_index
            ) queue_infos.push_back(
                {{}, dev_data.present_family_index, 1, &priority}
            );
            if(
                dev_data.transfer_family_index != dev_data.graphics_family_index &&
                dev_data.transfer_family_index != dev_data.compute_family_index
            ) queue_infos.push_back(
                {{}, dev_data.transfer_family_index, 1, &priority}
            );

            auto props2 = physical.getProperties2<
                vk::PhysicalDeviceProperties2,
                vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
                vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
                vk::PhysicalDeviceExternalMemoryHostPropertiesEXT,
                vk::PhysicalDeviceMultiviewProperties
            >();

            dev_data.id = devices.size();
            dev_data.ctx = this;
            dev_data.physical = physical;
            vk::DeviceCreateInfo device_create_info(
                {},
                queue_infos.size(),
                queue_infos.data(),
                validation_layers.size(),
                validation_layers.data(),
                enabled_device_extensions.size(),
                enabled_device_extensions.data(),
                nullptr
            );
            device_create_info.pNext = &feats;

            dev_data.logical = create_device(physical, device_create_info);
            dev_data.props = props;
            dev_data.subgroup_props = subgroup_props;
            dev_data.feats = feats.features;
            if(use_distribution)
            {
                dev_data.ext_mem_props =
                    props2.get<vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>();
            }
            dev_data.vulkan_11_feats = vulkan_11_feats;
            dev_data.vulkan_12_feats = vulkan_12_feats;

            dev_data.rt_props =
                props2.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
            dev_data.rt_feats = rt_feats;
            dev_data.rq_feats = rq_feats;
            dev_data.as_props = props2.get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
            dev_data.as_feats = as_feats;
            dev_data.mv_props = props2.get<vk::PhysicalDeviceMultiviewProperties>();
            // Potential Nvidia driver bug as of 510.47.03: multiview rendering
            // starts having problems after 20 or so viewports, despite reporting
            // support for 32. So limit it to 16.
            dev_data.mv_props.maxMultiviewViewCount = std::min(
                16u,
                dev_data.mv_props.maxMultiviewViewCount
            );
            dev_data.logical.getQueue(
                dev_data.graphics_family_index, 0, &dev_data.graphics_queue
            );
            dev_data.graphics_pool = dev_data.logical.createCommandPool(
                {{}, dev_data.graphics_family_index}
            );
            dev_data.logical.getQueue(
                dev_data.compute_family_index, 0, &dev_data.compute_queue
            );
            dev_data.compute_pool = dev_data.logical.createCommandPool(
                {{}, dev_data.compute_family_index}
            );

            if(dev_data.has_present)
            {
                dev_data.logical.getQueue(
                    dev_data.present_family_index, 0, &dev_data.present_queue
                );
                dev_data.present_pool = dev_data.logical.createCommandPool(
                    {{}, dev_data.present_family_index}
                );
                if(!display_device_set)
                {
                    display_device_index = devices.size();
                    display_device_set = true;
                }
            }

            dev_data.logical.getQueue(
                dev_data.transfer_family_index, 0, &dev_data.transfer_queue
            );
            dev_data.transfer_pool = dev_data.logical.createCommandPool(
                {{}, dev_data.transfer_family_index}
            );
            dev_data.pp_cache = dev_data.logical.createPipelineCache({
            });

            VmaAllocatorCreateInfo allocator_info = {};
            allocator_info.physicalDevice = physical;
            allocator_info.device = dev_data.logical;
            allocator_info.instance = instance;
            allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
            vmaCreateAllocator(&allocator_info, &dev_data.allocator);

            devices.push_back(std::move(dev_data));
        }
    }

    if(devices.size() == 0)
        throw std::runtime_error("Failed to find any suitable devices!");
}

void context::deinit_devices()
{
    sync();
    for(device& dev_data: devices)
    {
        dev_data.logical.destroyPipelineCache(dev_data.pp_cache);
        dev_data.logical.destroyCommandPool(dev_data.graphics_pool);
        dev_data.logical.destroyCommandPool(dev_data.compute_pool);
        if(dev_data.has_present)
        {
            dev_data.logical.destroyCommandPool(dev_data.present_pool);
        }
        dev_data.logical.destroyCommandPool(dev_data.transfer_pool);
        vmaDestroyAllocator(dev_data.allocator);
        dev_data.logical.destroy();
    }
}

void context::init_resources()
{
    device& dev_data = get_display_device();

    // Create fences & semaphores
    frame_available.resize(MAX_FRAMES_IN_FLIGHT);
    frame_finished.resize(MAX_FRAMES_IN_FLIGHT);
    frame_fences.resize(MAX_FRAMES_IN_FLIGHT);
    image_fences.resize(get_swapchain_image_count());
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        frame_available[i] = create_binary_semaphore(dev_data);
        frame_finished[i] = create_binary_semaphore(dev_data);
        frame_fences[i] =
            vkm(dev_data, dev_data.logical.createFence({vk::FenceCreateFlagBits::eSignaled}));
    }

    for(size_t i = 0; i < get_swapchain_image_count(); ++i)
    {
        image_available.emplace_back(create_timeline_semaphore(dev_data));
    }

    if(get_swapchain_image_count() == 0)
    {
        image_available.emplace_back(create_timeline_semaphore(dev_data));
        image_fences.resize(1);
    }

    placeholder_data.reset(new placeholders(*this));

    timing.init(opt.max_timestamps);
}

void context::reset_image_views()
{
    device& dev_data = get_display_device();

    // Create image views
    array_image_views.clear();
    for(size_t i = 0; i < images.size(); ++i)
    {
        array_image_views.emplace_back(dev_data,
            dev_data.logical.createImageView({
                {},
                *images[i],
                vk::ImageViewType::e2DArray,
                vk::Format((VkFormat)image_format),
                {},
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, image_array_layers}
            })
        );
    }
}

bool context::has_validation() const
{
    return validation_layers.size() > 0;
}

void context::deinit_resources()
{
    sync();
    placeholder_data.reset();

    image_available.clear();
    frame_fences.clear();
    frame_available.clear();
    frame_finished.clear();
    image_fences.clear();
    timing.deinit();
    tracker.end();
}

void context::call_frame_end_actions(uint32_t frame_index)
{
    for(std::function<void()>& func: frame_end_actions[frame_index])
        func();
    frame_end_actions[frame_index].clear();
}

vk::Instance context::get_vulkan_instance() const
{
    return instance;
}

dependencies context::fill_end_frame_dependencies(
    const dependencies& deps
){ return deps; }

}
