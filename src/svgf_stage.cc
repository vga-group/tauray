#include "svgf_stage.hh"
#include "misc.hh"

namespace
{
using namespace tr;

struct push_constant_buffer_atrous
{
    pivec2 size;
    int parity;
    int iteration;
    int stride;
    int iteration_count;
};

struct push_constant_buffer_temporal
{
    pivec2 size;
};

static_assert(sizeof(push_constant_buffer_atrous) <= 128);
static_assert(sizeof(push_constant_buffer_temporal) <= 128);
}

namespace tr
{

svgf_stage::svgf_stage(
    device_data& dev,
    gbuffer_target& input_features,
    gbuffer_target& prev_features,
    render_target& tmp_color1,
    render_target& tmp_color2,
    const options& opt
) : stage(dev),
    atrous_comp(dev, compute_pipeline::params{shader_source("shader/svgf_atrous.comp"), {}}),
    temporal_comp(dev, compute_pipeline::params{shader_source("shader/svgf_temporal.comp"), {}}),
    opt(opt),
    input_features(input_features),
    prev_features(prev_features),
    tmp_color{tmp_color1, tmp_color2},
    svgf_timer(dev, "svgf (" + std::to_string(input_features.get_layer_count()) + " viewports)")
{
    init_resources();
    record_command_buffers();
}

void svgf_stage::init_resources()
{
    for (int i = 0; i < 6; ++i)
    {
        render_target_texture[i].reset(new texture(
            *dev,
            input_features.color.get_size(),
            input_features.get_layer_count(),
            vk::Format::eR16G16B16A16Sfloat,
            0, nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage,
            vk::ImageLayout::eGeneral,
            vk::SampleCountFlagBits::e1
        ));
    }
    atrous_specular_pingpong[0] = render_target_texture[0]->get_array_render_target(dev->index);
    atrous_specular_pingpong[1] = render_target_texture[1]->get_array_render_target(dev->index);
    moments_history[0]          = render_target_texture[2]->get_array_render_target(dev->index);
    moments_history[1]          = render_target_texture[3]->get_array_render_target(dev->index);
    svgf_color_hist             = render_target_texture[4]->get_array_render_target(dev->index);
    svgf_spec_hist              = render_target_texture[5]->get_array_render_target(dev->index);

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        atrous_comp.update_descriptor_set({
            {"in_color", {{}, input_features.color[i].view, vk::ImageLayout::eGeneral}},
            {"in_normal", {{}, input_features.normal[i].view, vk::ImageLayout::eGeneral}},
            {"in_pos", {{}, input_features.pos[i].view, vk::ImageLayout::eGeneral}},
            {"in_albedo", {{}, input_features.albedo[i].view, vk::ImageLayout::eGeneral}},
            {"inout_color", {
                {{}, tmp_color[1][i].view, vk::ImageLayout::eGeneral},
                {{}, tmp_color[0][i].view, vk::ImageLayout::eGeneral}
            }},
            {"inout_specular", {
                {{}, atrous_specular_pingpong[1][i].view, vk::ImageLayout::eGeneral},
                {{}, atrous_specular_pingpong[0][i].view, vk::ImageLayout::eGeneral}
            }},
            {"moments_history", {
                {{}, moments_history[1][i].view, vk::ImageLayout::eGeneral},
                {{}, moments_history[0][i].view, vk::ImageLayout::eGeneral}
            }},
            {"svgf_color_hist", {{}, svgf_color_hist[i].view, vk::ImageLayout::eGeneral}},
            {"svgf_hist_specular", {{}, svgf_spec_hist[i].view, vk::ImageLayout::eGeneral}},
        }, i);
        temporal_comp.update_descriptor_set({
            {"in_color", {{}, input_features.color[i].view, vk::ImageLayout::eGeneral}},
            {"in_normal", {{}, input_features.normal[i].view, vk::ImageLayout::eGeneral}},
            {"in_pos", {{}, input_features.pos[i].view, vk::ImageLayout::eGeneral}},
            {"in_screen_motion", {{}, input_features.screen_motion[i].view, vk::ImageLayout::eGeneral}},
            {"previous_normal", {{}, prev_features.normal[i].view, vk::ImageLayout::eGeneral}},
            {"previous_pos", {{}, prev_features.pos[i].view, vk::ImageLayout::eGeneral}},
            {"in_albedo", {{}, input_features.albedo[i].view, vk::ImageLayout::eGeneral}},
            {"in_diffuse", {{}, input_features.diffuse[i].view, vk::ImageLayout::eGeneral}},
            {"inout_color", {
                {{}, tmp_color[1][i].view, vk::ImageLayout::eGeneral},
                {{}, tmp_color[0][i].view, vk::ImageLayout::eGeneral}
            }},
            {"inout_specular", {
                {{}, atrous_specular_pingpong[1][i].view, vk::ImageLayout::eGeneral},
                {{}, atrous_specular_pingpong[0][i].view, vk::ImageLayout::eGeneral}
            }},
            {"moments_history", {
                {{}, moments_history[1][i].view, vk::ImageLayout::eGeneral},
                {{}, moments_history[0][i].view, vk::ImageLayout::eGeneral}
            }},
            {"svgf_color_hist", {{}, svgf_color_hist[i].view, vk::ImageLayout::eGeneral}},
            {"svgf_hist_specular", {{}, svgf_spec_hist[i].view, vk::ImageLayout::eGeneral}},
        }, i);
    }
}

void svgf_stage::record_command_buffers()
{
    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();

        svgf_timer.begin(cb, i);

        uvec2 wg = (input_features.get_size()+15u)/16u;
        push_constant_buffer_temporal control_temporal;
        control_temporal.size = input_features.get_size();
        temporal_comp.bind(cb, i);
        temporal_comp.push_constants(cb, control_temporal);
        cb.dispatch(wg.x, wg.y, input_features.get_layer_count());

        vk::MemoryBarrier barrier{
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead
        };
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        push_constant_buffer_atrous control_atrous;
        control_atrous.size = input_features.get_size();
        control_atrous.iteration_count = opt.repeat_count;

        for (int j = 0; j < opt.repeat_count; ++j)
        {
            if (j != 0)
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
            control_atrous.parity = (j-1) % 2;
            control_atrous.iteration = j;
            control_atrous.stride = j;//opt.repeat_count-1-j;
            atrous_comp.bind(cb, i);
            atrous_comp.push_constants(cb, control_atrous);
            cb.dispatch(wg.x, wg.y, input_features.get_layer_count());
        }

        svgf_timer.end(cb, i);
        end_compute(cb, i);
    }
}

}
