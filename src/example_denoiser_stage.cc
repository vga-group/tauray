#include "example_denoiser_stage.hh"
#include "misc.hh"

namespace
{
using namespace tr;

shader_source load_source(const tr::example_denoiser_stage::options& opt)
{
    std::map<std::string, std::string> defines;
    if(opt.albedo_modulation)
        defines["USE_ALBEDO"];
    return {"shader/example_denoiser.comp", defines};
}

struct push_constant_buffer
{
    pivec2 size;
    int parity;
    int kernel_radius;
    int modulate_albedo;
};

static_assert(sizeof(push_constant_buffer) <= 128);

}

namespace tr
{

example_denoiser_stage::example_denoiser_stage(
    device_data& dev,
    gbuffer_target& input_features,
    render_target& tmp_color1,
    render_target& tmp_color2,
    const options& opt
):  stage(dev),
    comp(dev, compute_pipeline::params{load_source(opt), {}}),
    opt(opt),
    input_features(input_features),
    tmp_color{tmp_color1, tmp_color2},
    denoiser_timer(dev, "example denoiser (" + std::to_string(input_features.get_layer_count()) + " viewports)")
{
    init_resources();
    record_command_buffers();
}

bool example_denoiser_stage::need_pingpong_swap() const
{
    return opt.repeat_count % 2 == 1;
}

void example_denoiser_stage::init_resources()
{
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        comp.update_descriptor_set({
            {"in_color", {{}, input_features.color.view, vk::ImageLayout::eGeneral}},
            {"in_normal", {{}, input_features.normal.view, vk::ImageLayout::eGeneral}},
            {"in_pos", {{}, input_features.pos.view, vk::ImageLayout::eGeneral}},
            {"in_albedo", {{}, input_features.albedo.view, vk::ImageLayout::eGeneral}},
            {"inout_color", {
                {{}, tmp_color[1].view, vk::ImageLayout::eGeneral},
                {{}, tmp_color[0].view, vk::ImageLayout::eGeneral}
            }},
        }, i);
    }
}

void example_denoiser_stage::record_command_buffers()
{
    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();

        denoiser_timer.begin(cb, dev->index, i);

        comp.bind(cb, i);

        uvec2 wg = (input_features.get_size()+15u)/16u;
        push_constant_buffer control;
        control.size = input_features.get_size();
        control.kernel_radius = opt.kernel_radius;
        control.parity = -1;

        for(int j = 0; j < opt.repeat_count; ++j)
        {
            if(j != 0)
            {
                vk::MemoryBarrier barrier{
                    vk::AccessFlagBits::eShaderWrite,
                    vk::AccessFlagBits::eShaderRead
                };
                cb.pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader,
                    {}, barrier, {}, {}
                );
            }
            control.modulate_albedo = j == opt.repeat_count-1;
            control.parity = (j-1) % 2;

            comp.push_constants(cb, control);
            cb.dispatch(wg.x, wg.y, input_features.get_layer_count());
        }

        denoiser_timer.end(cb, dev->index, i);
        end_compute(cb, i);
    }
}

}
