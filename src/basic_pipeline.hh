#ifndef TAURAY_BASIC_PIPELINE_HH
#define TAURAY_BASIC_PIPELINE_HH
#include "context.hh"
#include "shader_source.hh"
#include "descriptor_state.hh"

namespace tr
{

struct shader_sources;
class descriptor_set_layout;
class descriptor_set;
class push_descriptor_set;

// Pipelines are per-device. This is the base class for multiple pipeline types.
// It exists mostly for code reuse purposes, since all pipelines have some
// common amenities for buffer handling.
class basic_pipeline
{
public:
    basic_pipeline(device& dev, vk::PipelineBindPoint bind_point);
    basic_pipeline(const basic_pipeline& other) = delete;
    basic_pipeline(basic_pipeline&& other) = delete;
    virtual ~basic_pipeline() = default;

    void init(
        std::vector<std::vector<vk::DescriptorSetLayoutBinding>>&& sets,
        std::map<std::string, std::pair<uint32_t, uint32_t>>&& binding_names,
        std::vector<vk::PushConstantRange>&& push_constant_ranges,
        uint32_t max_descriptor_sets,
        bool use_push_descriptors,
        // REFACTOR to be the only option, removing max_descriptor_sets
        // and use_push_descriptors and sets and binding_names.
        const std::vector<descriptor_set_layout*>& layout
    );

    void reset_descriptor_sets();
    void update_descriptor_set(
        const std::vector<descriptor_state>& descriptor_states,
        int32_t descriptor_set_index = -1 // If -1, all frames use same descriptors.
    );

    void push_descriptors(
        vk::CommandBuffer cb,
        const std::vector<descriptor_state>& descriptor_states
    );

    template<typename T>
    void push_constants(
        vk::CommandBuffer cb,
        const T& pc,
        size_t pc_index = 0
    ){
        cb.pushConstants(
            pipeline_layout, push_constant_ranges[pc_index].stageFlags,
            0, sizeof(T), &pc
        );
    }

    device* get_device() const;

    void bind(vk::CommandBuffer cmd, uint32_t descriptor_set_index) const;
    void set_descriptors(vk::CommandBuffer cmd, descriptor_set& set, uint32_t index, uint32_t set_index) const;
    void push_descriptors(vk::CommandBuffer cmd, push_descriptor_set& set,  uint32_t set_index) const;
    void bind(vk::CommandBuffer cmd) const;

protected:
    void load_shader_module(
        shader_source src,
        vk::ShaderStageFlagBits stage,
        std::vector<vk::PipelineShaderStageCreateInfo>& stages,
        const vk::SpecializationInfo& specialization
    );
    device* dev;
    vk::PipelineBindPoint bind_point;
    vkm<vk::Pipeline> pipeline;
    vkm<vk::PipelineLayout> pipeline_layout;
    vkm<vk::DescriptorSetLayout> descriptor_set_layout;
    vkm<vk::DescriptorPool> descriptor_pool;
    std::vector<vk::DescriptorSet> descriptor_sets;

private:
    const vk::DescriptorSetLayoutBinding* find_descriptor_binding(
        const std::string& binding
    ) const;

    std::vector<std::vector<vk::DescriptorSetLayoutBinding>> sets;
    std::map<std::string, std::pair<uint32_t, uint32_t>> binding_names;
    std::vector<vk::PushConstantRange> push_constant_ranges;
};

}

#endif

