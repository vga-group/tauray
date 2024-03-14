#ifndef TAURAY_DESCRIPTOR_SET_HH
#define TAURAY_DESCRIPTOR_SET_HH

#include "context.hh"
#include "vkm.hh"
#include <vector>
#include <set>

namespace tr
{

struct shader_source;
struct raster_shader_sources;
struct rt_shader_sources;
class texture;
class gpu_buffer;
class sampler;

class descriptor_set_layout
{
public:
    descriptor_set_layout(device_mask dev, bool push_descriptor_set);
    descriptor_set_layout(descriptor_set_layout&& other) noexcept = default;
    ~descriptor_set_layout();

    void add(
        std::string_view name,
        const vk::DescriptorSetLayoutBinding& binding,
        vk::DescriptorBindingFlags flags = {}
    );
    void add(const shader_source& source, uint32_t target_set_index = 0);
    void add(const raster_shader_sources& source, uint32_t target_set_index = 0);
    void add(const rt_shader_sources& source, uint32_t target_set_index = 0);
    void set_binding_params(
        std::string_view name,
        uint32_t count,
        vk::DescriptorBindingFlags flags = {}
    );

    struct set_binding: vk::DescriptorSetLayoutBinding
    {
        vk::DescriptorBindingFlags flags;
    };

    set_binding find_binding(std::string_view name) const;
    vk::DescriptorSetLayout get_layout(device_id id) const;

    device_mask get_mask() const;

protected:
    void refresh(device_id id) const;

    bool push_descriptor_set;
    mutable std::vector<vk::DescriptorSetLayoutBinding> bindings;
    struct layout_data
    {
        bool dirty = true;
        uint32_t descriptor_pool_capacity = 0;
        vkm<vk::DescriptorSetLayout> layout;
    };
    mutable per_device<layout_data> layout;

    std::set<std::string> descriptor_names;
    std::unordered_map<std::string_view, set_binding> named_bindings;
};

class descriptor_set: public descriptor_set_layout
{
public:
    descriptor_set(device_mask dev);
    descriptor_set(descriptor_set&& other) noexcept = default;
    ~descriptor_set();

    void reset(device_mask devices, uint32_t count);
    void reset(device_id id, uint32_t count);

    void set_image(
        device_id id,
        uint32_t index,
        std::string_view name,
        std::vector<vk::DescriptorImageInfo>&& infos
    );

    void set_texture(
        uint32_t index,
        std::string_view name,
        const texture& tex,
        const sampler& s
    );

    void set_image(
        uint32_t index,
        std::string_view name,
        const texture& tex
    );

    void set_buffer(
        device_id id,
        uint32_t index,
        std::string_view name,
        std::vector<vk::DescriptorBufferInfo>&& buffers
    );

    void set_buffer(
        uint32_t index,
        std::string_view name,
        const gpu_buffer& buffer,
        uint32_t offset = 0
    );

    void set_acceleration_structure(
        device_id id,
        uint32_t index,
        std::string_view name,
        vk::AccelerationStructureKHR tlas
    );

    void bind(
        device_id id,
        vk::CommandBuffer buf,
        vk::PipelineLayout pipeline_layout,
        vk::PipelineBindPoint bind_point,
        uint32_t alternative_index,
        uint32_t set_index
    ) const;

protected:
    struct set_data
    {
        std::vector<vk::DescriptorSet> alternatives;
        vkm<vk::DescriptorPool> pool;
    };
    per_device<set_data> data;
};

class push_descriptor_set: public descriptor_set_layout
{
public:
    push_descriptor_set(device_mask dev);
    push_descriptor_set(push_descriptor_set&& other) noexcept = default;
    ~push_descriptor_set();

    void set_image(
        device_id id,
        std::string_view name,
        std::vector<vk::DescriptorImageInfo>&& infos
    );

    void set_texture(
        std::string_view name,
        const texture& tex,
        const sampler& s
    );

    void set_image(
        std::string_view name,
        const texture& tex
    );

    void set_buffer(
        device_id id,
        std::string_view name,
        std::vector<vk::DescriptorBufferInfo>&& buffers
    );

    void set_buffer(
        std::string_view name,
        const gpu_buffer& buffer,
        uint32_t offset = 0
    );

    void set_acceleration_structure(
        device_id id,
        std::string_view name,
        vk::AccelerationStructureKHR tlas
    );

    void push(
        device_id id,
        vk::CommandBuffer buf,
        vk::PipelineLayout pipeline_layout,
        vk::PipelineBindPoint bind_point,
        uint32_t set_index
    );

protected:
    struct set_data
    {
        uint32_t image_info_index = 0;
        std::vector<std::vector<vk::DescriptorImageInfo>> tmp_image_infos;
        uint32_t buffer_info_index = 0;
        std::vector<std::vector<vk::DescriptorBufferInfo>> tmp_buffer_infos;
        uint32_t as_info_index = 0;
        std::vector<vk::AccelerationStructureKHR> tmp_as;
        std::vector<vk::WriteDescriptorSetAccelerationStructureKHR> tmp_as_infos;
        std::vector<vk::WriteDescriptorSet> writes;
    };
    per_device<set_data> data;
};

}

#endif

