#include "basic_pipeline.hh"
#include "descriptor_set.hh"
#include "misc.hh"

namespace tr
{

basic_pipeline::basic_pipeline(
    device& dev,
    vk::PipelineBindPoint bind_point
):  dev(&dev),
    bind_point(bind_point)
{
}

void basic_pipeline::init(
    std::vector<vk::PushConstantRange>&& push_constant_ranges,
    const std::vector<tr::descriptor_set_layout*>& layout
){
    this->push_constant_ranges = std::move(push_constant_ranges);

    std::vector<vk::DescriptorSetLayout> descriptor_sets;

    for(tr::descriptor_set_layout* layout: layout)
        descriptor_sets.push_back(layout->get_layout(dev->id));

    vk::PipelineLayoutCreateInfo pipeline_layout_info(
        {}, descriptor_sets.size(), descriptor_sets.data(),
        this->push_constant_ranges.size(),
        this->push_constant_ranges.data()
    );

    pipeline_layout = vkm(*dev, dev->logical.createPipelineLayout(pipeline_layout_info));
}

device* basic_pipeline::get_device() const
{
    return dev;
}

void basic_pipeline::set_descriptors(vk::CommandBuffer cmd, descriptor_set& set, uint32_t index, uint32_t set_index) const
{
    set.bind(dev->id, cmd, pipeline_layout, bind_point, index, set_index);
}

void basic_pipeline::push_descriptors(vk::CommandBuffer cmd, push_descriptor_set& set,  uint32_t set_index) const
{
    set.push(dev->id, cmd, pipeline_layout, bind_point, set_index);
}

void basic_pipeline::bind(vk::CommandBuffer cmd) const
{
    cmd.bindPipeline(bind_point, *pipeline);
}

void basic_pipeline::load_shader_module(
    shader_source src,
    vk::ShaderStageFlagBits stage,
    std::vector<vk::PipelineShaderStageCreateInfo>& stages,
    const vk::SpecializationInfo& specialization
){
    if(src.data.empty()) return;

    vkm<vk::ShaderModule> mod = vkm(*dev, dev->logical.createShaderModule({
        {}, src.data.size()*sizeof(uint32_t), src.data.data()
    }));
    stages.push_back({{}, stage, mod, "main", specialization.pData != nullptr ? &specialization : nullptr});
}


}
