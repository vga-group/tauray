#ifndef TAURAY_BASIC_PIPELINE_HH
#define TAURAY_BASIC_PIPELINE_HH
#include "context.hh"
#include "shader_source.hh"

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
        std::vector<vk::PushConstantRange>&& push_constant_ranges,
        const std::vector<descriptor_set_layout*>& layout
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

private:
    std::vector<vk::PushConstantRange> push_constant_ranges;
};

}

#endif

