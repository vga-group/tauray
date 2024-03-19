#include "compute_pipeline.hh"
#include "misc.hh"
#include <map>

namespace tr
{

compute_pipeline::compute_pipeline(device& dev)
:   basic_pipeline(dev, vk::PipelineBindPoint::eCompute)
{
}

void compute_pipeline::init(
    shader_source src,
    std::vector<tr::descriptor_set_layout*> layout
){
    basic_pipeline::init(get_push_constant_ranges(src), layout);
    if(src.data.empty())
        throw std::runtime_error("The shader source code is missing!");

    vkm<vk::ShaderModule> comp(*dev, dev->logical.createShaderModule({
        {}, src.data.size() * sizeof(uint32_t), src.data.data()
    }));

    vk::ComputePipelineCreateInfo pipeline_info(
        {}, {{}, vk::ShaderStageFlagBits::eCompute, comp, "main"},
        pipeline_layout, {}, 0
    );

    pipeline = vkm(*dev, dev->logical.createComputePipeline(dev->pp_cache, pipeline_info).value);
}

}
