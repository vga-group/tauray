#ifndef TAURAY_CONTEXT_HH
#define TAURAY_CONTEXT_HH

#include "math.hh"
#include "vkm.hh"
#include "dependency.hh"
#include "render_target.hh"
#include <set>
#include <map>
#include <chrono>
#include <functional>

namespace tr
{

class context;
struct device_data
{
    size_t index = 0;
    context* ctx = nullptr;
    vk::PhysicalDevice pdev;
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
    VmaAllocator allocator;
};

class placeholders;

// This should typically be _lower_ than the number of images in the display
// targets! In any case, there really cannot be more frames than the number
// of swap chain images going on at the same time, since their image views
// would clash.
static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

class context
{
public:
    struct options
    {
        bool disable_ray_tracing = false;
        // Empty vector: use all compatible devices
        // -1: First compatible device only
        // >= 0: Filter to only allow these indices.
        std::set<int> physical_device_indices = {};
        // If zero, timestamp() is a no-op. If non-zero, the number of
        // timestamps that can be measured during one frame.
        unsigned max_timestamps = 0;
        bool enable_vulkan_validation = false;
        unsigned fake_device_multiplier = 0;
    };

    context(const options& opt);
    context(const context& other) = delete;
    context(context&& other) = delete;
    virtual ~context();

    virtual bool init_frame();

    device_data& get_display_device();
    std::vector<device_data>& get_devices();
    uvec2 get_size() const;
    vk::Format get_display_format() const;
    vk::ImageLayout get_expected_display_layout() const;

    void set_displaying(bool displaying);
    bool get_displaying() const;

    size_t get_swapchain_image_count() const;
    // The default implementations of these functions assume that there is only
    // one display per image (and not that one image is divided into multiple
    // separate viewports).
    size_t get_display_count() const;
    render_target get_array_render_target();

    placeholders& get_placeholders();

    bool is_ray_tracing_supported() const;

    // The returned dependency is only for images[swapchain_index]. You can
    // start rendering into internal buffers immediately after the call.
    dependency begin_frame();
    void end_frame(const dependencies& deps);
    void get_indices(uint32_t& swapchain_index, uint32_t& frame_index) const;
    uint32_t get_frame_counter() const;
    // Ignore this unless you know what you are doing. Rendering algorithms
    // should only use the above function.
    uint32_t get_displayed_frame_counter() const;

    // Waits until all devices are idle. Calling this in destructors is
    // sometimes a good idea.
    void sync();

    // You can add functions to be called when the current frame is guaranteed
    // to be finished on the GPU side.
    void queue_frame_finish_callback(std::function<void()>&& func);

    int register_timer(size_t device_index, const std::string& name);
    void unregister_timer(size_t device_index, int timer_id);
    vk::QueryPool get_timestamp_pool(size_t device_index, uint32_t frame_index);

    float get_timing(size_t device_index, const std::string& name) const;
    void print_timing() const;
    // Slow function, call this at the end before destroying pipelines but
    // after issuing the last draw calls in order to print the remaining
    // in-flight times.
    void finish_print_timing();

    vk::Instance get_vulkan_instance() const;

protected:
    virtual uint32_t prepare_next_image(uint32_t frame_index) = 0;
    virtual dependencies fill_end_frame_dependencies(const dependencies& deps);
    virtual void finish_image(
        uint32_t frame_index,
        uint32_t swapchain_index,
        bool display // If false, the image shouldn't be output in any way.
    ) = 0;
    virtual bool queue_can_present(
        const vk::PhysicalDevice& device,
        uint32_t queue_index,
        const vk::QueueFamilyProperties& props
    ) = 0;
    virtual vk::Instance create_instance(
        const vk::InstanceCreateInfo& info,
        PFN_vkGetInstanceProcAddr getInstanceProcAddr
    );
    virtual vk::Device create_device(
        const vk::PhysicalDevice& device,
        const vk::DeviceCreateInfo& info
    );

    void init_vulkan(PFN_vkGetInstanceProcAddr getInstanceProcAddr);
    void deinit_vulkan();

    void init_devices();
    void deinit_devices();

    void init_resources();
    void deinit_resources();

    void reset_image_views();

    vk::Instance instance;
    std::vector<const char*> extensions;
    uvec2 image_size;
    unsigned image_array_layers;
    vk::Format image_format;
    vk::ImageLayout expected_image_layout;
    std::vector<vkm<vk::Image>> images;
    std::vector<vkm<vk::ImageView>> array_image_views;

    // These unfortunately have to be binary semaphores for presentKHR and
    // acquireNextImageKHR... :(
    std::vector<vkm<vk::Semaphore>> frame_available;
    std::vector<vkm<vk::Semaphore>> frame_finished;

private:
    void free_purgatory();
    void call_frame_end_actions(uint32_t frame_index);
    void step_timing();
    void save_timing(uint32_t frame_number);
    void print_timing_internal(uint32_t frame_number) const;

    options opt;
    std::vector<const char*> validation_layers;
    vk::DebugUtilsMessengerEXT debug_messenger;
    std::vector<device_data> devices;
    size_t display_device_index;

    std::vector<vkm<vk::Semaphore>> image_available;
    std::vector<vkm<vk::Fence>> frame_fences;
    std::vector<vk::Fence> image_fences;
    // This is the frame counter you can rely on for timing and rendering
    // duties.
    uint64_t frame_counter;
    // Not all frames are displayed due to is_displaying, so this only counts
    // those. Basically only useful for numbering actually rendered frames.
    uint32_t displayed_frame_counter;
    uint32_t swapchain_index;
    uint32_t frame_index;
    bool is_displaying;

    std::unique_ptr<placeholders> placeholder_data;

    struct time_info
    {
        uint64_t start;
        uint64_t end;
        std::string name;
    };

    struct timing_data
    {
        vkm<vk::QueryPool> timestamp_pool[MAX_FRAMES_IN_FLIGHT];

        std::set<int> available_queries;
        std::map<int, std::string> reserved_queries;

        std::vector<time_info> times;
    };
    std::vector<timing_data> timing;
    std::chrono::steady_clock::time_point host_timing[MAX_FRAMES_IN_FLIGHT][2];
    float last_host_time;

    // Callbacks for the end of each frame.
    std::vector<std::function<void()>> frame_end_actions[MAX_FRAMES_IN_FLIGHT];
};

}

#endif
