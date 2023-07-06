#include "svgf_stage.hh"
#include "misc.hh"
#include "camera.hh"
#include "placeholders.hh"

namespace
{
using namespace tr;

struct push_constants
{
    pivec2 size;
    int iteration;
    int diffuse_iteration_count;
    int specular_iteration_count;
    int atrous_kernel_radius;
    float sigma_n;
    float sigma_z;
    float sigma_l;
    float temporal_alpha_color;
    float temporal_alpha_moments;
};

static_assert(sizeof(push_constants) <= 128);
}

namespace tr
{

svgf_stage::svgf_stage(
    device& dev,
    gbuffer_target& input_features,
    gbuffer_target& prev_features,
    const options& opt
):  single_device_stage(dev),
    atrous_comp(dev, compute_pipeline::params{shader_source("shader/svgf_atrous.comp"), {}, 0, true}),
    temporal_comp(dev, compute_pipeline::params{shader_source("shader/svgf_temporal.comp"), {}}),
    estimate_variance_comp(dev, compute_pipeline::params{ shader_source("shader/svgf_estimate_variance.comp"), {} }),
    opt(opt),
    input_features(input_features),
    prev_features(prev_features),
    svgf_timer(dev, "svgf (" + std::to_string(input_features.get_layer_count()) + " viewports)"),
    jitter_buffer(dev, sizeof(pvec4)* 1, vk::BufferUsageFlagBits::eStorageBuffer)
{
}

void svgf_stage::set_scene(scene* cur_scene)
{
    this->cur_scene = cur_scene;
    init_resources();
    record_command_buffers();
}

void svgf_stage::update(uint32_t frame_index)
{
    bool existing = jitter_history.size() != 0;
    size_t viewport_count = opt.active_viewport_count;
    jitter_history.resize(viewport_count);

    for (size_t i = 0; i < viewport_count; ++i)
    {
        vec4& v = jitter_history[i];
        vec2 cur_jitter = cur_scene->get_camera(i)->get_jitter();
        vec2 prev_jitter = v;
        if (!existing) prev_jitter = cur_jitter;
        v = vec4(cur_jitter, prev_jitter);
    }

    jitter_buffer.update(frame_index, jitter_history.data());
}

void svgf_stage::init_resources()
{
    for (int i = 0; i < 8; ++i)
    {
        render_target_texture[i].reset(new texture(
            *dev,
            input_features.color.size,
            input_features.get_layer_count(),
            vk::Format::eR16G16B16A16Sfloat,
            0, nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage,
            vk::ImageLayout::eGeneral,
            vk::SampleCountFlagBits::e1
        ));
    }

    int rt_index = 0;
    atrous_specular_pingpong[0] = render_target_texture[rt_index++]->get_array_render_target(dev->index);
    atrous_specular_pingpong[1] = render_target_texture[rt_index++]->get_array_render_target(dev->index);
    moments_history[0]          = render_target_texture[rt_index++]->get_array_render_target(dev->index);
    moments_history[1]          = render_target_texture[rt_index++]->get_array_render_target(dev->index);
    svgf_color_hist             = render_target_texture[rt_index++]->get_array_render_target(dev->index);
    svgf_spec_hist              = render_target_texture[rt_index++]->get_array_render_target(dev->index);
    atrous_diffuse_pingpong[0]  = render_target_texture[rt_index++]->get_array_render_target(dev->index);
    atrous_diffuse_pingpong[1]  = render_target_texture[rt_index++]->get_array_render_target(dev->index);

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        cur_scene->bind(temporal_comp, i);
        temporal_comp.update_descriptor_set({
            {"in_color", {{}, input_features.color.view, vk::ImageLayout::eGeneral}},
            {"in_diffuse", {{}, input_features.diffuse.view, vk::ImageLayout::eGeneral}},
            {"previous_color", {{}, svgf_color_hist.view, vk::ImageLayout::eGeneral}},
            {"in_normal", {{}, input_features.normal.view, vk::ImageLayout::eGeneral}},
            {"in_screen_motion", {{}, input_features.screen_motion.view, vk::ImageLayout::eGeneral}},
            {"previous_normal", {{}, prev_features.normal.view, vk::ImageLayout::eGeneral}},
            {"in_albedo", {{}, input_features.albedo.view, vk::ImageLayout::eGeneral}},
            {"previous_moments", {{}, moments_history[0].view, vk::ImageLayout::eGeneral}},
            {"out_moments", {{}, moments_history[1].view, vk::ImageLayout::eGeneral}},
            {"out_color", {{}, atrous_diffuse_pingpong[0].view, vk::ImageLayout::eGeneral}},
            {"out_specular", {{}, atrous_specular_pingpong[0].view, vk::ImageLayout::eGeneral} },
            {"in_linear_depth", {{}, input_features.linear_depth.view, vk::ImageLayout::eGeneral}},
            {"previous_linear_depth", {{}, prev_features.linear_depth.view, vk::ImageLayout::eGeneral}},
            {"jitter_info", {jitter_buffer[dev->index], 0, VK_WHOLE_SIZE}},
            {"previous_specular", {{}, svgf_spec_hist.view, vk::ImageLayout::eGeneral}},
        }, i);
        estimate_variance_comp.update_descriptor_set({
            {"in_diffuse", {{}, atrous_diffuse_pingpong[0].view, vk::ImageLayout::eGeneral}},
            {"out_diffuse", {{}, atrous_diffuse_pingpong[1].view, vk::ImageLayout::eGeneral}},
            {"in_specular", {{}, atrous_specular_pingpong[0].view, vk::ImageLayout::eGeneral} },
            {"out_specular", {{}, atrous_specular_pingpong[1].view, vk::ImageLayout::eGeneral} },
            {"in_linear_depth", {{}, input_features.linear_depth.view, vk::ImageLayout::eGeneral}},
            {"current_moments", {{}, moments_history[1].view, vk::ImageLayout::eGeneral}},
            {"moments_hist", {{}, moments_history[0].view, vk::ImageLayout::eGeneral}},
            {"in_normal", {{}, input_features.normal.view, vk::ImageLayout::eGeneral}},
        }, i);
    }
}

void svgf_stage::record_command_buffers()
{
    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();

        svgf_timer.begin(cb, dev->index, i);

        jitter_buffer.upload(dev->index, i, cb);

        uvec2 wg = (input_features.get_size()+15u) / 16u;
        push_constants control{};
        control.size = input_features.get_size();
        control.diffuse_iteration_count = opt.atrous_diffuse_iters;
        control.specular_iteration_count = opt.atrous_spec_iters;
        control.atrous_kernel_radius = opt.atrous_kernel_radius;
        control.sigma_l = opt.sigma_l;
        control.sigma_z = opt.sigma_z;
        control.sigma_n = opt.sigma_n;
        control.temporal_alpha_color = opt.temporal_alpha_color;
        control.temporal_alpha_moments = opt.temporal_alpha_moments;

        temporal_comp.bind(cb, i);
        temporal_comp.push_constants(cb, control);
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

        estimate_variance_comp.bind(cb, i);
        estimate_variance_comp.push_constants(cb, control);
        cb.dispatch(wg.x, wg.y, input_features.get_layer_count());

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        atrous_comp.bind(cb);
        atrous_comp.push_descriptors(cb,
            {
                {"final_output", {{}, input_features.color.view, vk::ImageLayout::eGeneral}},
                {"diffuse_hist", {{}, svgf_color_hist.view, vk::ImageLayout::eGeneral}},
                {"spec_hist", {{}, svgf_spec_hist.view, vk::ImageLayout::eGeneral}},
                {"in_linear_depth", {{}, input_features.linear_depth.view, vk::ImageLayout::eGeneral}},
                {"in_normal", {{}, input_features.normal.view, vk::ImageLayout::eGeneral}},
                {"in_albedo", {{}, input_features.albedo.view, vk::ImageLayout::eGeneral}},
                {"in_material", {{}, input_features.material.view, vk::ImageLayout::eGeneral}},
                {"in_moments", {{}, moments_history[1].view, vk::ImageLayout::eGeneral}}
            }
        );
        const int iteration_count = opt.atrous_diffuse_iters;
        for (int j = 0; j < iteration_count; ++j)
        {
            if (j != 0)
            {
                cb.pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader,
                    {}, barrier, {}, {}
                );
            }
            int out_index = j & 1;
            int in_index = (j + 1) & 1;
            atrous_comp.push_descriptors(cb,
                {
                    {"diffuse_in", {{}, atrous_diffuse_pingpong[in_index].view, vk::ImageLayout::eGeneral}},
                    {"diffuse_out", {{}, atrous_diffuse_pingpong[out_index].view, vk::ImageLayout::eGeneral}},
                    {"specular_in", {{}, atrous_specular_pingpong[in_index].view, vk::ImageLayout::eGeneral}},
                    {"specular_out", {{}, atrous_specular_pingpong[out_index].view, vk::ImageLayout::eGeneral}}
                }
            );

            control.iteration = j;
            atrous_comp.push_constants(cb, control);
            cb.dispatch(wg.x, wg.y, input_features.get_layer_count());
        }

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        svgf_timer.end(cb, dev->index, i);
        end_compute(cb, i);
    }
}

}
