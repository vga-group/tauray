#ifndef TAURAY_DEVICE_HH
#define TAURAY_DEVICE_HH

#include "math.hh"
#include "vkm.hh"
#include <unordered_map>

namespace tr
{

class context;
using device_id = unsigned;
struct device
{
    // TODO: rename to id
    device_id index = 0;
    context* ctx = nullptr;
    // TODO rename to physical
    vk::PhysicalDevice pdev;
    // TODO rename to logical
    vk::Device dev;
    vk::PhysicalDeviceProperties props;
    vk::PhysicalDeviceSubgroupProperties subgroup_props;
    vk::PhysicalDeviceFeatures feats;
    vk::PhysicalDeviceVulkan11Features vulkan_11_feats;
    vk::PhysicalDeviceVulkan12Features vulkan_12_feats;
    vk::PhysicalDeviceExternalMemoryHostPropertiesEXT ext_mem_props;
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rt_props;
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rt_feats;
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR as_props;
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR as_feats;
    vk::PhysicalDeviceMultiviewProperties mv_props;
    uint32_t graphics_family_index = 0;
    uint32_t compute_family_index = 0;
    uint32_t present_family_index = 0;
    uint32_t transfer_family_index = 0;
    bool has_graphics = false;
    bool has_compute = false;
    bool has_present = false;
    bool has_transfer = false;
    vk::Queue graphics_queue;
    vk::Queue compute_queue;
    vk::Queue present_queue;
    vk::Queue transfer_queue;
    vk::CommandPool graphics_pool;
    vk::CommandPool compute_pool;
    vk::CommandPool present_pool;
    vk::CommandPool transfer_pool;
    vk::PipelineCache pp_cache;
    VmaAllocator allocator;
};

// TODO: Remove near end of refactor, pointless to do now since most references
// will go away.
using device_data = device;

class device_mask
{
public:
    device_mask();
    // Single-device mask
    device_mask(device& dev);
    static device_mask all(context& ctx);
    static device_mask none(context& ctx);

    device_mask(const device_mask& other) = default;
    device_mask& operator=(const device_mask& other) = default;

    bool contains(device_id id) const;
    void erase(device_id id);
    void insert(device_id id);

    struct iterator
    {
        context* ctx;
        uint64_t bitmask;

        iterator& operator++();
        device& operator*() const;
        bool operator==(const iterator& other) const;
        bool operator!=(const iterator& other) const;
    };

    // Iterates keys in the order they were first added.
    iterator begin() const;
    iterator end() const;
    iterator cbegin() const;
    iterator cend() const;

    void clear();
    std::size_t size() const;
    context* get_context() const;
    device& get_device(device_id id) const;

private:
    context* ctx;
    uint64_t bitmask;
};

template<typename T>
struct per_device
{
public:
    per_device();

    per_device(device_mask mask);

    template<typename... Args>
    void emplace(device_mask mask, Args&&... args);

    template<typename F>
    void init(device_mask mask, F&& create_callback);

    T& operator[](device_id id);
    const T& operator[](device_id id) const;

    template<typename F>
    void operator()(F&& callback);

    void clear();

    device_mask get_mask() const;
    context* get_context() const;
    device& get_device(device_id id) const;

private:
    device_mask active_mask;
    std::unordered_map<unsigned, T> devices;
};

}

#include "device.tcc"

#endif
