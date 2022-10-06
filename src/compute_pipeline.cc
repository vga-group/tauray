#include "compute_pipeline.hh"
#include "misc.hh"
#include <map>

namespace
{
using namespace tr;

shader_sources sources(const shader_source& src)
{
    shader_sources srcs;
    srcs.comp = src;
    return srcs;
}

}

namespace tr
{

compute_pipeline::compute_pipeline(device_data& dev, const params& p)
:   basic_pipeline(
        dev, sources(p.src), p.binding_array_lengths,
        p.max_descriptor_sets, vk::PipelineBindPoint::eCompute,
        p.use_push_descriptors
    )
{
    if(p.src.data.empty())
        throw std::runtime_error("The shader source code is missing!");

    vkm<vk::ShaderModule> comp(dev, dev.dev.createShaderModule({
        {}, p.src.data.size() * sizeof(uint32_t), p.src.data.data()
    }));

    vk::ComputePipelineCreateInfo pipeline_info(
        {}, {{}, vk::ShaderStageFlagBits::eCompute, comp, "main"},
        pipeline_layout, {}, 0
    );

    pipeline = vkm(dev, dev.dev.createComputePipeline({}, pipeline_info).value);
}

}
