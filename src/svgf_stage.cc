#include "svgf_stage.hh"
#include "misc.hh"
#include "camera.hh"
#include "placeholders.hh"
#include "log.hh"

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
    float min_rect_dim_mul_unproject;
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
    firefly_suppression_desc(dev), firefly_suppression_comp(dev),
    disocclusion_fix_desc(dev), disocclusion_fix_comp(dev),
    prefilter_variance_desc(dev), prefilter_variance_comp(dev),
    preblur_desc(dev), preblur_comp(dev),
    demodulate_inputs_desc(dev), demodulate_inputs_comp(dev),
    hit_dist_reconstruction_desc(dev), hit_dist_reconstruction_comp(dev),
    opt(opt),
    input_features(input_features),
    prev_features(prev_features),
    svgf_timer(dev, "svgf (" + std::to_string(input_features.get_layer_count()) + " viewports)"),
    jitter_buffer(dev, sizeof(pvec4) * 1, vk::BufferUsageFlagBits::eStorageBuffer),
    uniforms(dev, sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer),
    ss(&ss),
    scene_state_counter(0),
    my_sampler(
        dev,
        vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eLinear,
        1,
        true,
        false,
        false,
        0.0f
    )
{
    {
        shader_source src("shader/svgf_atrous.comp");
        atrous_desc.add(src);
        atrous_comp.init(src, { &atrous_desc,  &ss.get_descriptors() });
    }
    {
        shader_source src("shader/svgf_temporal.comp");
        temporal_desc.add(src);
        temporal_comp.init(src, {&temporal_desc, &ss.get_descriptors()});
    }
    {
        shader_source src("shader/svgf_spatial_variance.comp");
        estimate_variance_desc.add(src);
        estimate_variance_comp.init(src, {&estimate_variance_desc, &ss.get_descriptors() });
    }
    {
        shader_source src("shader/svgf_firefly_suppression.comp");
        firefly_suppression_desc.add(src);
        firefly_suppression_comp.init(src, {&firefly_suppression_desc});
    }
    {
        shader_source src("shader/svgf_disocclusion_fix.comp");
        disocclusion_fix_desc.add(src);
        disocclusion_fix_comp.init(src, { &disocclusion_fix_desc, &ss.get_descriptors() });
    }
    {
        shader_source src("shader/svgf_prefilter_variance.comp");
        prefilter_variance_desc.add(src);
        prefilter_variance_comp.init(src, { &prefilter_variance_desc, &ss.get_descriptors() });
    } 
    {
        shader_source src("shader/svgf_preblur.comp");
        preblur_desc.add(src);
        preblur_comp.init(src, { &preblur_desc, &ss.get_descriptors() });
    }
    {
        shader_source src("shader/svgf_demodulate_inputs.comp");
        demodulate_inputs_desc.add(src);
        demodulate_inputs_comp.init(src, { &demodulate_inputs_desc, &ss.get_descriptors() });
    }
    {
        shader_source src("shader/svgf_hit_dist_reconstruction.comp");
        hit_dist_reconstruction_desc.add(src);
        hit_dist_reconstruction_comp.init(src, { &hit_dist_reconstruction_desc, &ss.get_descriptors() });
    }

    init_resources();
}

void svgf_stage::update(uint32_t frame_index)
{
    if(ss->check_update(scene_stage::ENVMAP, scene_state_counter))
        record_command_buffers();

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

    uint32_t frame_counter = dev->ctx->get_frame_counter();
    uniforms.update(frame_index, &frame_counter, 0, sizeof(uint32_t));
}

void svgf_stage::init_resources()
{
    for (int i = 0; i < svgf_stage::render_target_count; ++i)
    {
        vk::Format format = i == render_target_count - 1 ? vk::Format::eR16G16B16A16Sfloat : vk::Format::eR32G32B32A32Sfloat;
        render_target_texture[i].reset(new texture(
            *dev,
            input_features.color.size,
            input_features.get_layer_count(),
            //vk::Format::eR16G16B16A16Sfloat,
            format,
            0, nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
            vk::ImageLayout::eGeneral,
            vk::SampleCountFlagBits::e1
        ));
    }

    int rt_index = 0;
    atrous_specular_pingpong[0]     = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    atrous_specular_pingpong[1]     = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    moments_history[0]              = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    moments_history[1]              = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    svgf_color_hist                 = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    svgf_spec_hist                  = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    atrous_diffuse_pingpong[0]      = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    atrous_diffuse_pingpong[1]      = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    specular_hit_distance_history   = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    accumulated_specular_hit_distance = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    emissive                        = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    moments_history_specular[0]     = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    moments_history_specular[1] = render_target_texture[rt_index++]->get_array_render_target(dev->id);
    hit_dist_reconst_temp           = render_target_texture[rt_index++]->get_array_render_target(dev->id);
}

void svgf_stage::record_command_buffers()
{
    clear_commands();
    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();

        svgf_timer.begin(cb, dev->id, i);

        jitter_buffer.upload(dev->id, i, cb);
        uniforms.upload(dev->id, i, cb);

        scene* cur_scene = ss->get_scene();
        std::vector<entity> cameras = get_sorted_cameras(*cur_scene);
        camera* cam = cur_scene->get<camera>(cameras[0]);
        float tan_half_fovy = tanf(cam->get_vfov() * 0.5f);
        float unproject = 2.0 * tan_half_fovy / input_features.get_size().y;
        float min_rect_dim_mul_unproject = min((float)input_features.get_size().x, (float)input_features.get_size().y) * unproject;

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
        control.min_rect_dim_mul_unproject = min_rect_dim_mul_unproject;

        vk::MemoryBarrier barrier{
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead
        };

        demodulate_inputs_comp.bind(cb);
        demodulate_inputs_desc.set_image(dev->id, "in_color", { {{}, input_features.color.view, vk::ImageLayout::eGeneral} });
        demodulate_inputs_desc.set_image(dev->id, "in_diffuse", { {{}, input_features.diffuse.view, vk::ImageLayout::eGeneral} });
        demodulate_inputs_desc.set_image(dev->id, "in_specular", { {{}, input_features.reflection.view, vk::ImageLayout::eGeneral} });
        demodulate_inputs_desc.set_image(dev->id, "in_material", { {{}, input_features.material.view, vk::ImageLayout::eGeneral} });
        demodulate_inputs_desc.set_image(dev->id, "in_albedo", { {{}, input_features.albedo.view, vk::ImageLayout::eGeneral} });
        demodulate_inputs_desc.set_image(dev->id, "in_depth", { {my_sampler.get_sampler(dev->id), input_features.depth.view, vk::ImageLayout::eGeneral} });
        demodulate_inputs_desc.set_image(dev->id, "in_normal", { {{}, input_features.normal.view, vk::ImageLayout::eGeneral} });
        demodulate_inputs_desc.set_image(dev->id, "out_diffuse", { {{}, input_features.diffuse.view, vk::ImageLayout::eGeneral} });
        demodulate_inputs_desc.set_image(dev->id, "out_specular", { {{}, input_features.reflection.view, vk::ImageLayout::eGeneral} });
        demodulate_inputs_desc.set_image(dev->id, "out_emissive", { {{}, emissive.view, vk::ImageLayout::eGeneral} });
        demodulate_inputs_comp.push_descriptors(cb, demodulate_inputs_desc, 0);
        demodulate_inputs_comp.set_descriptors(cb, ss->get_descriptors(), 0, 1);
        demodulate_inputs_comp.push_constants(cb, control);
        cb.dispatch(wg.x, wg.y, input_features.get_layer_count());


        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        // Hit dist reconstruction
        hit_dist_reconstruction_comp.bind(cb);
        hit_dist_reconstruction_desc.set_image(dev->id, "in_specular", { {{}, input_features.reflection.view, vk::ImageLayout::eGeneral} });
        hit_dist_reconstruction_desc.set_image(dev->id, "out_specular", { {{}, hit_dist_reconst_temp.view,  vk::ImageLayout::eGeneral} }); hit_dist_reconstruction_desc.set_image(dev->id, "normal", { {{}, input_features.normal.view, vk::ImageLayout::eGeneral} });
        hit_dist_reconstruction_desc.set_image(dev->id, "in_material", { {{}, input_features.material.view,  vk::ImageLayout::eGeneral} });
        hit_dist_reconstruction_desc.set_image(dev->id, "in_normal", { {{}, input_features.normal.view, vk::ImageLayout::eGeneral} });
        hit_dist_reconstruction_desc.set_image(dev->id, "in_depth", { {my_sampler.get_sampler(dev->id), input_features.depth.view, vk::ImageLayout::eGeneral} });
        hit_dist_reconstruction_comp.push_descriptors(cb, hit_dist_reconstruction_desc, 0);
        hit_dist_reconstruction_comp.set_descriptors(cb, ss->get_descriptors(), 0, 1);
        hit_dist_reconstruction_comp.push_constants(cb, control);
        cb.dispatch(wg.x, wg.y, input_features.get_layer_count());

        // Copy image
        {
            uvec3 size = uvec3(input_features.get_size(), 1);

            render_target& src = hit_dist_reconst_temp;
            render_target& dst = input_features.reflection;
            src.transition_layout_temporary(cb, vk::ImageLayout::eTransferSrcOptimal);
            dst.transition_layout_temporary(
                cb, vk::ImageLayout::eTransferDstOptimal, true, true
            );

            cb.copyImage(
                src.image,
                vk::ImageLayout::eTransferSrcOptimal,
                dst.image,
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageCopy(
                    src.get_layers(),
                    { 0,0,0 },
                    dst.get_layers(),
                    { 0,0,0 },
                    { size.x, size.y, size.z }
                )
            );

            vk::ImageLayout old_layout = src.layout;
            src.layout = vk::ImageLayout::eTransferSrcOptimal;
            src.transition_layout_temporary(cb, vk::ImageLayout::eGeneral);
            src.layout = old_layout;

            old_layout = dst.layout;
            dst.layout = vk::ImageLayout::eTransferDstOptimal;
            dst.transition_layout_temporary(
                cb, vk::ImageLayout::eGeneral, true, true
            );
            dst.layout = old_layout;
        }

        preblur_comp.bind(cb);
        preblur_desc.set_image(dev->id, "in_diffuse", { {{}, input_features.diffuse.view, vk::ImageLayout::eGeneral} });
        preblur_desc.set_image(dev->id, "normal", { {{}, input_features.normal.view, vk::ImageLayout::eGeneral} });
        preblur_desc.set_image(dev->id, "in_depth", { {my_sampler.get_sampler(dev->id), input_features.depth.view, vk::ImageLayout::eGeneral} });
        preblur_desc.set_image(dev->id, "out_diffuse", { {{}, atrous_diffuse_pingpong[0].view, vk::ImageLayout::eGeneral} });
        preblur_desc.set_image(dev->id, "in_specular", { {{}, input_features.reflection.view, vk::ImageLayout::eGeneral} });
        preblur_desc.set_image(dev->id, "out_specular", { {{}, atrous_specular_pingpong[0].view,  vk::ImageLayout::eGeneral} });
        preblur_desc.set_buffer("uniforms_buffer", uniforms);
        preblur_desc.set_image(dev->id, "in_material", { {{}, input_features.material.view,  vk::ImageLayout::eGeneral} });
        preblur_comp.push_descriptors(cb, preblur_desc, 0);
        preblur_comp.set_descriptors(cb, ss->get_descriptors(), 0, 1);
        preblur_comp.push_constants(cb, control);
        cb.dispatch(wg.x, wg.y, input_features.get_layer_count());

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        temporal_comp.bind(cb);
        temporal_desc.set_image(dev->id, "in_color", {{{}, input_features.color.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "in_diffuse", {{{}, atrous_diffuse_pingpong[0].view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "in_specular", {{{},atrous_specular_pingpong[0].view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "previous_color", {{my_sampler.get_sampler(dev->id), svgf_color_hist.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "in_normal", {{{}, input_features.normal.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "in_screen_motion", {{{}, input_features.screen_motion.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "previous_normal", {{my_sampler.get_sampler(dev->id), prev_features.normal.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "in_albedo", {{{}, input_features.albedo.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "previous_moments", {{my_sampler.get_sampler(dev->id), moments_history[0].view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "out_moments", {{{}, moments_history[1].view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "out_color", {{{}, atrous_diffuse_pingpong[1].view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "out_specular", {{{}, atrous_specular_pingpong[1].view, vk::ImageLayout::eGeneral} });
        temporal_desc.set_image(dev->id, "in_prev_depth", {{my_sampler.get_sampler(dev->id), prev_features.depth.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_buffer("jitter_info", jitter_buffer);
        temporal_desc.set_image(dev->id, "previous_specular", {{my_sampler.get_sampler(dev->id), svgf_spec_hist.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "in_material", {{{}, input_features.material.view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "in_depth", { {my_sampler.get_sampler(dev->id), input_features.depth.view, vk::ImageLayout::eGeneral} });
        temporal_desc.set_image(dev->id, "specular_hit_distance_history", { {my_sampler.get_sampler(dev->id), specular_hit_distance_history.view, vk::ImageLayout::eGeneral} });
        temporal_desc.set_image(dev->id, "out_specular_hit_distance", { {{}, accumulated_specular_hit_distance.view, vk::ImageLayout::eGeneral} });
        temporal_desc.set_image(dev->id, "previous_material", { {my_sampler.get_sampler(dev->id), prev_features.material.view, vk::ImageLayout::eGeneral} });
        temporal_desc.set_image(dev->id, "previous_moments_specular", { {my_sampler.get_sampler(dev->id), moments_history_specular[0].view, vk::ImageLayout::eGeneral}});
        temporal_desc.set_image(dev->id, "out_moments_specular", {{{}, moments_history_specular[1].view, vk::ImageLayout::eGeneral}});
        temporal_comp.push_descriptors(cb, temporal_desc, 0);
        temporal_comp.set_descriptors(cb, ss->get_descriptors(), 0, 1);
        temporal_comp.push_constants(cb, control);
        cb.dispatch(wg.x, wg.y, input_features.get_layer_count());


        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        disocclusion_fix_comp.bind(cb);
        disocclusion_fix_desc.set_image(dev->id, "accumulated_diffuse", { {{}, atrous_diffuse_pingpong[1].view, vk::ImageLayout::eGeneral} });
        disocclusion_fix_desc.set_image(dev->id, "filtered_diffuse", { {{}, atrous_diffuse_pingpong[0].view, vk::ImageLayout::eGeneral} });
        disocclusion_fix_desc.set_image(dev->id, "normal", { {{}, input_features.normal.view, vk::ImageLayout::eGeneral} });
        disocclusion_fix_desc.set_image(dev->id, "in_depth", { {my_sampler.get_sampler(dev->id), input_features.depth.view, vk::ImageLayout::eGeneral} });
        disocclusion_fix_desc.set_image(dev->id, "moments_histlen", { {{}, moments_history[1].view, vk::ImageLayout::eGeneral} });
        disocclusion_fix_desc.set_image(dev->id, "diffuse_hist", { {{}, svgf_color_hist.view, vk::ImageLayout::eGeneral} });
        disocclusion_fix_desc.set_image(dev->id, "accumulated_specular", { {{}, atrous_specular_pingpong[1].view, vk::ImageLayout::eGeneral} });
        disocclusion_fix_desc.set_image(dev->id, "specular_hist", { {{}, svgf_spec_hist.view, vk::ImageLayout::eGeneral} });
        disocclusion_fix_desc.set_image(dev->id, "filtered_specular", { {{},  atrous_specular_pingpong[0].view, vk::ImageLayout::eGeneral} });
        disocclusion_fix_desc.set_image(dev->id, "accumulated_specular_hit_distance", { {{},  accumulated_specular_hit_distance.view, vk::ImageLayout::eGeneral} });
        disocclusion_fix_desc.set_image(dev->id, "specular_hit_distance_history", { {{},  specular_hit_distance_history.view, vk::ImageLayout::eGeneral} });
        disocclusion_fix_desc.set_image(dev->id, "in_material", { {{},  input_features.material.view, vk::ImageLayout::eGeneral} });

        disocclusion_fix_comp.push_descriptors(cb, disocclusion_fix_desc, 0);
        disocclusion_fix_comp.set_descriptors(cb, ss->get_descriptors(), 0, 1);
        disocclusion_fix_comp.push_constants(cb, control);
        cb.dispatch(wg.x, wg.y, input_features.get_layer_count());

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        firefly_suppression_comp.bind(cb);
        firefly_suppression_desc.set_image(dev->id, "accumulated_diffuse", { {{}, atrous_diffuse_pingpong[0].view, vk::ImageLayout::eGeneral} });
        firefly_suppression_desc.set_image(dev->id, "filtered_diffuse", { {{}, atrous_diffuse_pingpong[1].view, vk::ImageLayout::eGeneral} });
        firefly_suppression_desc.set_image(dev->id, "accumulated_specular", { {{}, atrous_specular_pingpong[0].view, vk::ImageLayout::eGeneral} });
        firefly_suppression_desc.set_image(dev->id, "filtered_specular", { {{}, atrous_specular_pingpong[1].view, vk::ImageLayout::eGeneral} });
        firefly_suppression_comp.push_descriptors(cb, firefly_suppression_desc, 0);
        firefly_suppression_comp.push_constants(cb, control);
        cb.dispatch(wg.x, wg.y, input_features.get_layer_count());

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );


        estimate_variance_comp.bind(cb);
        estimate_variance_desc.set_image(dev->id, "in_diffuse", {{{}, atrous_diffuse_pingpong[1].view, vk::ImageLayout::eGeneral}});
        estimate_variance_desc.set_image(dev->id, "out_diffuse", {{{}, atrous_diffuse_pingpong[0].view, vk::ImageLayout::eGeneral}});
        estimate_variance_desc.set_image(dev->id, "in_specular", {{{}, atrous_specular_pingpong[1].view, vk::ImageLayout::eGeneral} });
        estimate_variance_desc.set_image(dev->id, "out_specular", {{{}, atrous_specular_pingpong[0].view, vk::ImageLayout::eGeneral} });
        estimate_variance_desc.set_image(dev->id, "current_moments", {{{}, moments_history[1].view, vk::ImageLayout::eGeneral}});
        estimate_variance_desc.set_image(dev->id, "moments_hist", {{{}, moments_history[0].view, vk::ImageLayout::eGeneral}});
        estimate_variance_desc.set_image(dev->id, "in_normal", { {{}, input_features.normal.view, vk::ImageLayout::eGeneral} });
        estimate_variance_desc.set_image(dev->id, "in_depth", {{my_sampler.get_sampler(dev->id), input_features.depth.view, vk::ImageLayout::eGeneral}});
        estimate_variance_desc.set_image(dev->id, "current_moments_specular", { {{}, moments_history_specular[1].view, vk::ImageLayout::eGeneral} });
        estimate_variance_desc.set_image(dev->id, "moments_hist_specular", { {{}, moments_history_specular[0].view, vk::ImageLayout::eGeneral} });

        estimate_variance_comp.push_descriptors(cb, estimate_variance_desc, 0);
        estimate_variance_comp.set_descriptors(cb, ss->get_descriptors(), 0, 1);
        estimate_variance_comp.push_constants(cb, control);
        cb.dispatch(wg.x, wg.y, input_features.get_layer_count());

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        prefilter_variance_comp.bind(cb);
        prefilter_variance_desc.set_image(dev->id, "in_color", { {{}, atrous_diffuse_pingpong[0].view, vk::ImageLayout::eGeneral} });
        prefilter_variance_desc.set_image(dev->id, "out_color", { {{}, atrous_diffuse_pingpong[1].view, vk::ImageLayout::eGeneral} });
        prefilter_variance_desc.set_image(dev->id, "in_specular", { {{}, atrous_specular_pingpong[0].view, vk::ImageLayout::eGeneral} });
        prefilter_variance_desc.set_image(dev->id, "out_specular", { {{}, atrous_specular_pingpong[1].view, vk::ImageLayout::eGeneral} });
        prefilter_variance_comp.push_descriptors(cb, prefilter_variance_desc, 0);
        prefilter_variance_comp.push_constants(cb, control);
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
            atrous_desc.set_image(dev->id, "in_normal", {{{}, input_features.normal.view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "in_albedo", {{{}, input_features.albedo.view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "in_material", {{{}, input_features.material.view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "in_moments", {{{}, moments_history[1].view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "diffuse_in", {{{}, atrous_diffuse_pingpong[in_index].view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "diffuse_out", {{{}, atrous_diffuse_pingpong[out_index].view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "specular_in", {{{}, atrous_specular_pingpong[in_index].view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "specular_out", { {{}, atrous_specular_pingpong[out_index].view, vk::ImageLayout::eGeneral} });
            atrous_desc.set_image(dev->id, "in_depth", { {my_sampler.get_sampler(dev->id), input_features.depth.view, vk::ImageLayout::eGeneral} });
            atrous_desc.set_image(dev->id, "raw_diffuse", { {{}, input_features.diffuse.view, vk::ImageLayout::eGeneral} });
            atrous_desc.set_buffer("uniforms_buffer", uniforms);
            atrous_desc.set_image(dev->id, "in_moments_specular", { {{}, moments_history_specular[1].view, vk::ImageLayout::eGeneral} });
            atrous_desc.set_image(dev->id, "in_emissive", { {{}, emissive.view, vk::ImageLayout::eGeneral}});
            atrous_desc.set_image(dev->id, "specular_hit_dist", { {{}, specular_hit_distance_history.view, vk::ImageLayout::eGeneral}});
            
            atrous_comp.push_descriptors(cb, atrous_desc, 0);
            atrous_comp.set_descriptors(cb, ss->get_descriptors(), 0, 1);

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
