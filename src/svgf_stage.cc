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
    scene_stage& ss,
    gbuffer_target& input_features,
    gbuffer_target& prev_features,
    const options& opt
):  single_device_stage(dev),
    atrous_desc(dev), atrous_comp(dev),
    temporal_desc(dev), temporal_comp(dev),
    estimate_variance_desc(dev), estimate_variance_comp(dev),
    opt(opt),
    input_features(input_features),
    prev_features(prev_features),
    svgf_timer(dev, "svgf (" + std::to_string(input_features.get_layer_count()) + " viewports)"),
    jitter_buffer(dev, sizeof(pvec4)* 1, vk::BufferUsageFlagBits::eStorageBuffer),
    ss(&ss),
    scene_state_counter(0)
{
    {
        shader_source src("shader/svgf_atrous.comp");
        atrous_desc.add(src);
        estimate_variance_comp.init(src, {&estimate_variance_desc});
    }
    {
        shader_source src("shader/svgf_temporal.comp");
        temporal_desc.add(src);
        temporal_comp.init(src, {&temporal_desc, &ss.get_descriptors()});
    }
    {
        shader_source src("shader/svgf_estimate_variance.comp");
        estimate_variance_desc.add(src);
        estimate_variance_comp.init(src, {&estimate_variance_desc});
    }
}

void svgf_stage::update(uint32_t frame_index)
{
    if(ss->check_update(scene_stage::ENVMAP, scene_state_counter))
    {
        init_resources();
        record_command_buffers();
    }

    bool existing = jitter_history.size() != 0;
    size_t viewport_count = opt.active_viewport_count;
    jitter_history.resize(viewport_count);

    scene* cur_scene = ss->get_scene();
    std::vector<entity> cameras = get_sorted_cameras(*cur_scene);
    for (size_t i = 0; i < viewport_count; ++i)
    {
        vec4& v = jitter_history[i];
        vec2 cur_jitter = cur_scene->get<camera>(cameras[i])->get_jitter();
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
    atrous_specular_pingpong[0] = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    atrous_specular_pingpong[1] = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    moments_history[0]          = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    moments_history[1]          = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    svgf_color_hist             = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    svgf_spec_hist              = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    atrous_diffuse_pingpong[0]  = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    atrous_diffuse_pingpong[1]  = render_target_texture[rt_index++]->get_array_render_target(dev->id);
}

void svgf_stage::record_command_buffers()
{
    clear_commands();
    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();

        svgf_timer.begin(cb, dev->id, i);

        jitter_buffer.upload(dev->id, i, cb);

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

        temporal_comp.bind(cb);
        temporal_desc.set_image(dev->id, "in_color", {{{}, input_features.color.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "in_diffuse", {{{}, input_features.diffuse.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "previous_color", {{{}, svgf_color_hist.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "in_normal", {{{}, input_features.normal.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "in_screen_motion", {{{}, input_features.screen_motion.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "previous_normal", {{{}, prev_features.normal.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "in_albedo", {{{}, input_features.albedo.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "previous_moments", {{{}, moments_history[0].view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "out_moments", {{{}, moments_history[1].view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "out_color", {{{}, atrous_diffuse_pingpong[0].view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "out_specular", {{{}, atrous_specular_pingpong[0].view, vk::ImageLayout::eGeneral} });
        temporal_desc.set_image(dev->id, "in_linear_depth", {{{}, input_features.linear_depth.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "previous_linear_depth", {{{}, prev_features.linear_depth.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_buffer("jitter_info", jitter_buffer);
        temporal_desc.set_image(dev->id, "previous_specular", {{{}, svgf_spec_hist.view, vk::ImageLayout::eGeneral}});
        temporal_comp.push_descriptors(cb, temporal_desc, 0);
        temporal_comp.set_descriptors(cb, ss->get_descriptors(), 0, 1);
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

        estimate_variance_comp.bind(cb);
        estimate_variance_desc.set_image(dev->id, "in_diffuse", {{{}, atrous_diffuse_pingpong[0].view, vk::ImageLayout::eGeneral}});
        estimate_variance_desc.set_image(dev->id, "out_diffuse", {{{}, atrous_diffuse_pingpong[1].view, vk::ImageLayout::eGeneral}});
        estimate_variance_desc.set_image(dev->id, "in_specular", {{{}, atrous_specular_pingpong[0].view, vk::ImageLayout::eGeneral} });
        estimate_variance_desc.set_image(dev->id, "out_specular", {{{}, atrous_specular_pingpong[1].view, vk::ImageLayout::eGeneral} });
        estimate_variance_desc.set_image(dev->id, "in_linear_depth", {{{}, input_features.linear_depth.view, vk::ImageLayout::eGeneral}});
        estimate_variance_desc.set_image(dev->id, "current_moments", {{{}, moments_history[1].view, vk::ImageLayout::eGeneral}});
        estimate_variance_desc.set_image(dev->id, "moments_hist", {{{}, moments_history[0].view, vk::ImageLayout::eGeneral}});
        estimate_variance_desc.set_image(dev->id, "in_normal", {{{}, input_features.normal.view, vk::ImageLayout::eGeneral}});
        estimate_variance_comp.push_descriptors(cb, estimate_variance_desc, 0);
        estimate_variance_comp.push_constants(cb, control);
        cb.dispatch(wg.x, wg.y, input_features.get_layer_count());

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        atrous_comp.bind(cb);

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

            atrous_desc.set_image(dev->id, "final_output", {{{}, input_features.color.view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "diffuse_hist", {{{}, svgf_color_hist.view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "spec_hist", {{{}, svgf_spec_hist.view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "in_linear_depth", {{{}, input_features.linear_depth.view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "in_normal", {{{}, input_features.normal.view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "in_albedo", {{{}, input_features.albedo.view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "in_material", {{{}, input_features.material.view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "in_moments", {{{}, moments_history[1].view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "diffuse_in", {{{}, atrous_diffuse_pingpong[in_index].view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "diffuse_out", {{{}, atrous_diffuse_pingpong[out_index].view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "specular_in", {{{}, atrous_specular_pingpong[in_index].view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "specular_out", {{{}, atrous_specular_pingpong[out_index].view, vk::ImageLayout::eGeneral}});

            atrous_comp.push_descriptors(cb, atrous_desc, 0);

            control.iteration = j;
            atrous_comp.push_constants(cb, control);
            cb.dispatch(wg.x, wg.y, input_features.get_layer_count());
        }

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        svgf_timer.end(cb, dev->id, i);
        end_compute(cb, i);
    }
}

}
