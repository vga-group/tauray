#ifndef TAURAY_MISC_HH
#define TAURAY_MISC_HH
#include "vkm.hh"

namespace tr
{

struct device;
class context;

template<typename T, size_t count>
inline std::string to_string(const vk::ArrayWrapper1D<T, count>& vkstring)
{
    return vkstring.data();
}

// These are for one-off command buffers.
vk::CommandBuffer begin_command_buffer(device& d);
void end_command_buffer(device& d, vk::CommandBuffer cb);

vkm<vk::CommandBuffer> create_compute_command_buffer(device& d);
vkm<vk::CommandBuffer> create_graphics_command_buffer(device& d);

vkm<vk::Semaphore> create_binary_semaphore(device& d);
vkm<vk::Semaphore> create_timeline_semaphore(device& d);

void transition_image_layout(
    vk::CommandBuffer cb,
    vk::Image img,
    vk::Format fmt,
    vk::ImageLayout src_layout,
    vk::ImageLayout dst_layout,
    uint32_t mip_level = 0,
    uint32_t mip_count = 1,
    uint32_t base_layer = 0,
    uint32_t layer_count = VK_REMAINING_ARRAY_LAYERS,
    bool ignore_src_stage_mask = false,
    bool ignore_dst_stage_mask = false
);

vkm<vk::Buffer> create_buffer(
    device& dev,
    vk::BufferCreateInfo info,
    VmaAllocationCreateFlagBits flags,
    const void* data = nullptr,
    vk::CommandBuffer shared_cb = {}
);

vkm<vk::Buffer> create_buffer_aligned(
    device& dev,
    vk::BufferCreateInfo info,
    VmaAllocationCreateFlagBits flags,
    size_t alignment,
    const void* data = nullptr
);

vkm<vk::Buffer> create_staging_buffer(
    device& dev,
    size_t size,
    const void* data = nullptr
);

// Staging but in reverse, GPU->CPU.
vkm<vk::Buffer> create_download_buffer(
    device& dev,
    size_t size
);

void* allocate_host_buffer(
    const std::vector<device*>& supported_devices,
    size_t size
);

void release_host_buffer(void* host_buffer);

void create_host_allocated_buffer(
    device& dev,
    vk::Buffer& res,
    vk::DeviceMemory& mem,
    size_t size,
    void* data
);

void destroy_host_allocated_buffer(
    device& dev,
    vk::Buffer& res,
    vk::DeviceMemory& mem
);

vk::ImageAspectFlags deduce_aspect_mask(vk::Format fmt);

void deduce_layout_access_stage(
    vk::ImageLayout layout,
    vk::AccessFlags& access,
    vk::PipelineStageFlags& stage
);

vkm<vk::Image> sync_create_gpu_image(
    device& dev,
    vk::ImageCreateInfo info,
    vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal,
    size_t data_size = 0,
    void* data = nullptr
);

// The hammer for all problems (if you don't care about performance at all)
void full_barrier(vk::CommandBuffer cb);
void bulk_upload_barrier(
    vk::CommandBuffer cb,
    vk::PipelineStageFlags usage_flags = vk::PipelineStageFlagBits::eAllCommands
);

vk::SampleCountFlagBits get_max_available_sample_count(context& ctx);

std::string get_resource_path(const std::string& path);
std::string load_text_file(const std::string& path);
bool nonblock_getline(std::string& line);

template<typename T>
void sorted_insert(
    std::vector<T>& vec,
    const T& value
){
    auto it = std::lower_bound(vec.begin(), vec.end(), value);
    if(it == vec.end() || *it != value) vec.insert(it, value);
}

template<typename T>
bool sorted_erase(
    std::vector<T>& vec,
    const T& value
){
    auto it = std::lower_bound(vec.begin(), vec.end(), value);
    if(it != vec.end() && *it == value)
    {
        vec.erase(it);
        return true;
    }
    return false;
}

template<typename T>
void unsorted_insert(
    std::vector<T>& vec,
    const T& value
){
    auto it = std::find(vec.begin(), vec.end(), value);
    if(it == vec.end()) vec.push_back(value);
}

template<typename T>
bool unsorted_erase(
    std::vector<T>& vec,
    const T& value
){
    auto it = std::find(vec.begin(), vec.end(), value);
    if(it != vec.end())
    {
        vec.erase(it);
        return true;
    }
    return false;
}

std::string to_uppercase(const std::string& str);

template<typename T>
size_t count_array_layers(const std::vector<T>& targets)
{
    size_t count = 0;
    for(const T& t: targets) count += t.layer_count;
    return count;
}

template<typename T>
size_t count_gbuffer_array_layers(const std::vector<T>& targets)
{
    size_t count = 0;
    for(const T& t: targets) count += t.get_layer_count();
    return count;
}

// For lazy CPU profiling ;)
void profile_tick();
void profile_tock(const char* message = "Tock: ");

template<typename T>
void set_debug_object_name(const tr::device& device, const T& vulkan_object, const char* name)
{
    assert(device.ctx);
    if (device.ctx->has_validation())
    {
        vk::DebugUtilsObjectNameInfoEXT info{};
        info.objectHandle = (uint64_t)static_cast<T::CType>(vulkan_object);
        info.objectType = vulkan_object.objectType;
        info.pObjectName = name;
        device.logical.setDebugUtilsObjectNameEXT(&info);
    }
}

}

#endif
