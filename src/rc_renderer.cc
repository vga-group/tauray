#include "rc_renderer.hh"
//#include "vulkan/vulkan_format_traits.hpp"
#include "log.hh"
#include "misc.hh"

namespace tr
{

rc_renderer::rc_renderer(context& ctx, const options& opt)
: ctx(&ctx), opt(opt)
{
    device& dev = ctx.get_display_device();

    this->opt.scene_options.alloc_sh_grids = false;
    this->opt.scene_options.track_prev_tlas = false;
    this->opt.scene_options.shadow_mapping = true;

    if(opt.restir_options && !opt.restir_options->assume_unchanged_acceleration_structures)
        this->opt.scene_options.track_prev_tlas = true;

    if(ctx.has_validation() && this->opt.scene_options.track_prev_tlas)
    {
        TR_WARN(
            "Vulkan validation layers crash when trying to copy old TLAS; "
            "setting restir.assume-unchanged-acceleration-structures = true."
        );
        this->opt.scene_options.track_prev_tlas = false;
    }

    if(this->opt.pt_options)
    {
        this->opt.pt_options->distribution.size = ctx.get_size();
        this->opt.pt_options->distribution.strategy = DISTRIBUTION_DUPLICATE;
        this->opt.pt_options->active_viewport_count = 1;
        this->opt.restir_options.reset();
    }
    else if(this->opt.restir_options)
    {
        this->opt.restir_options->max_bounces = max(this->opt.restir_options->max_bounces, 1u);
        this->opt.restir_options->demodulated_output = opt.svgf_options.has_value();
        this->opt.restir_options->camera_index = 0;
        this->opt.restir_options->expect_taa_jitter = opt.taa_options.has_value();
    }

    gbuffer_spec gs;
    gs.color_present = true;
    gs.color_format = vk::Format::eR32G32B32A32Sfloat;
    bool need_full_gbuffer = 
        opt.taa_options || opt.svgf_options || opt.restir_options;

    if(opt.enable_visualizer)
        need_full_gbuffer = false;

    if(need_full_gbuffer)
    {
        if(opt.svgf_options)
        {
            gs.diffuse_present = true;
            gs.reflection_present = true;
        }
        else gs.emission_present = true;
        gs.depth_present = true;
        gs.albedo_present = true;
        gs.material_present = true;
        gs.normal_present = true;

        if(opt.restir_options || opt.taa_options || opt.svgf_options)
        {
            gs.curvature_present = true;
            gs.screen_motion_present = true;
            gs.flat_normal_present = true;
        }

        if(opt.svgf_options && opt.restir_options)
        {
            gs.confidence_present = true;
            gs.temporal_gradient_present = true;
        }
    }

    vk::ImageUsageFlags img_usage =
        vk::ImageUsageFlagBits::eStorage|
        vk::ImageUsageFlagBits::eTransferSrc|
        vk::ImageUsageFlagBits::eTransferDst|
        vk::ImageUsageFlagBits::eSampled|
        vk::ImageUsageFlagBits::eColorAttachment;

    gs.set_all_usage(img_usage);
    gs.depth_usage =
        vk::ImageUsageFlagBits::eDepthStencilAttachment |
        vk::ImageUsageFlagBits::eTransferSrc |
        vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eSampled;

    scene_update.emplace(dev, this->opt.scene_options);

    if (opt.rc_options.use_raster_di)
    {
        sms.emplace(dev, *scene_update, shadow_map_stage::options{});
    }

    std::vector<uint8_t> distance_field_data = load_binary_file(opt.distance_field_path);
    uint8_t* dfdata = distance_field_data.data();
    memcpy(&this->opt.rc_options.volume.min, dfdata, sizeof(float)*3);
    dfdata += sizeof(float)*3;
    memcpy(&this->opt.rc_options.volume.max, dfdata, sizeof(float)*3);
    dfdata += sizeof(float)*3;
    vec3 resolution;
    memcpy(&resolution, dfdata, sizeof(float)*3);
    dfdata += sizeof(float)*3;

    printf("Distance field:\n");
    printf("    Resolution: %u x %u x %u\n", uint(resolution.x), uint(resolution.y), uint(resolution.z));
    printf("    Range: [%f, %f, %f] - [%f, %f, %f]\n",
        this->opt.rc_options.volume.min.x,
        this->opt.rc_options.volume.min.y,
        this->opt.rc_options.volume.min.z,
        this->opt.rc_options.volume.max.x,
        this->opt.rc_options.volume.max.y,
        this->opt.rc_options.volume.max.z
    );

    int max_res = round(log2(resolution.x));
    if (this->opt.rc_options.log2_resolution > max_res)
    {
        TR_WARN("Cannot have a higher c0 density than distance field!");
        this->opt.rc_options.log2_resolution = max_res;
    }

    distance_field.emplace(texture(
        dev,
        uvec3(resolution),
        vk::Format::eR32Sfloat,
        distance_field_data.data()+distance_field_data.size()-dfdata,
        dfdata,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
        vk::ImageLayout::eGeneral,
        true
    ));
    this->opt.rc_options.distance_field = &distance_field.value();

    voxelizer.emplace(dev, *scene_update, voxelizer_stage::options{int(resolution.x), this->opt.rc_options.volume});

    current_gbuffer.reset(dev, ctx.get_size(), 1);
    current_gbuffer.add(gs, vk::ImageLayout::eGeneral);

    if(opt.restir_options || opt.svgf_options)
    {
        prev_gbuffer.reset(dev, ctx.get_size(), 1);
        prev_gbuffer.add(gs, vk::ImageLayout::eGeneral);
    }

    rc.emplace(dev, *scene_update, this->opt.rc_options);

    rc_visualizer_stage::options rcv_opt;
    rcv_opt.distance_field = &distance_field.value();
    rcv_opt.occupancy_map = &voxelizer->get_map();
    gbuffer_target cur = current_gbuffer.get_layer_target(dev.id, 0);

    if(opt.enable_visualizer)
    {
        rcv.emplace(*rc, cur.color, rcv_opt);
        cur = current_gbuffer.get_array_target(dev.id);

        std::vector<render_target> display = ctx.get_array_render_target();
        this->opt.tonemap_options.limit_to_input_layer = 0;
        this->opt.tonemap_options.limit_to_output_layer = 0;
        this->opt.tonemap_options.transition_output_layout = true;
        tonemap.emplace(
            dev,
            cur.color,
            display,
            this->opt.tonemap_options
        );
    }
    else
    {
        if (need_full_gbuffer)
        {
            envmap.emplace(dev, *scene_update, cur.color, 0);
            raster_stage::options raster_opt;
            raster_opt.clear_color = false;
            raster_opt.clear_depth = true;
            raster_opt.sample_shading = false;
            raster_opt.use_probe_visibility = false;
            raster_opt.sh_order = 0;
            raster_opt.estimate_direct = false;
            raster_opt.estimate_indirect = false;
            raster_opt.force_alpha_to_coverage = true;
            raster_opt.base_camera_index = 0;
            raster_opt.output_layout = vk::ImageLayout::eGeneral;

            render_target diffuse = cur.diffuse;
            render_target reflection = cur.reflection;
            render_target temporal_gradient = cur.temporal_gradient;
            render_target confidence = cur.confidence;
            cur.diffuse = render_target();
            cur.reflection = render_target();
            cur.temporal_gradient = render_target();
            cur.confidence = render_target();

            gbuffer_rasterizer.emplace(dev, *scene_update, cur, raster_opt);

            cur.diffuse = diffuse;
            cur.reflection = reflection;
            cur.temporal_gradient = temporal_gradient;
            cur.confidence = confidence;
            cur.color.layout = vk::ImageLayout::eGeneral;
        }

        if(this->opt.pt_options)
        {
            cur = current_gbuffer.get_array_target(dev.id);
            gbuffer_target old = cur;
            if(need_full_gbuffer)
            {
                cur = gbuffer_target();
                cur.color = old.color;
                cur.diffuse = old.diffuse;
                cur.reflection = old.reflection;
            }
            if(opt.svgf_options)
                cur.color = render_target();
            this->opt.pt_options->rc_source = &*rc;
            pt.emplace(dev, *scene_update, cur, *this->opt.pt_options);
            cur = old;
        }
        else if(this->opt.restir_options)
        {
            cur = current_gbuffer.get_layer_target(dev.id, 0);
            gbuffer_target prev = prev_gbuffer.get_layer_target(dev.id, 0);
            this->opt.restir_options->rc_source = &*rc;
            restir.emplace(dev, *scene_update, cur, prev, *this->opt.restir_options);

            cur = current_gbuffer.get_array_target(dev.id);
        }

        gbuffer_target prev;
        if(opt.svgf_options || opt.restir_options)
            prev = prev_gbuffer.get_array_target(dev.id);

        if(opt.svgf_options)
        {
            //this->opt.svgf_options->atrous_kernel_radius = 1;
            //this->opt.svgf_options->atrous_diffuse_iters = 5;

            svgf.emplace(
                dev,
                *scene_update,
                cur,
                prev,
                *this->opt.svgf_options
            );
        }

        std::vector<render_target> display = ctx.get_array_render_target();
        this->opt.tonemap_options.limit_to_input_layer = 0;
        this->opt.tonemap_options.limit_to_output_layer = 0;
        this->opt.tonemap_options.transition_output_layout = true;
        if(this->opt.taa_options)
        {
            taa_input_target.emplace(
                dev,
                ctx.get_size(),
                1,
                vk::Format::eR16G16B16A16Sfloat,
                0, nullptr,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eSampled,
                vk::ImageLayout::eGeneral,
                vk::SampleCountFlagBits::e1
            );
            render_target taa_target = taa_input_target->get_array_render_target(dev.id);
            tonemap.emplace(dev, cur.color, taa_target, this->opt.tonemap_options);
            this->opt.taa_options->gamma = 2.2f;
            this->opt.taa_options->base_camera_index = 0;
            this->opt.taa_options->active_viewport_count = 1;
            this->opt.taa_options->output_layer = 0;

            taa.emplace(
                dev,
                *scene_update,
                taa_target,
                cur.screen_motion,
                cur.depth,
                display,
                *this->opt.taa_options
            );
        }
        else
        {
            tonemap.emplace(dev, cur.color, display, this->opt.tonemap_options);
        }

        if(prev.entry_count() != 0)
        {
            cur.color = render_target();
            cur.screen_motion = render_target();
            cur.temporal_gradient = render_target();
            cur.emission = render_target();
            prev.color = render_target();
            prev.screen_motion = render_target();
            prev.temporal_gradient = render_target();
            prev.emission = render_target();

            copy.emplace(dev, cur, prev, 0, 0);
        }
    }
}

void rc_renderer::set_scene(scene* s)
{
    scene_update->set_scene(s);
}

void rc_renderer::set_visualizer_pos(int cascade, int layer)
{
    rcv->set_position(cascade, layer);
}

void rc_renderer::render()
{
    dependencies display_deps(ctx->begin_frame());
    uint32_t swapchain_index, frame_index;
    ctx->get_indices(swapchain_index, frame_index);

    // Disable random seed for PT for DEBUGGING
    //pt->reset_sample_counter();

    dependencies deps = scene_update->run(display_deps);
    if (opt.rc_options.use_raster_di)
    {
        deps = sms->run(deps);
    }
    deps = voxelizer->run(deps);
    deps = rc->run(deps);

    pt->force_command_buffer_refresh();
    if(rcv) deps = rcv->run(deps);
    else
    {
        if(envmap) deps = envmap->run(deps);
        if(gbuffer_rasterizer) deps = gbuffer_rasterizer->run(deps);
        if(restir) deps = restir->run(deps);
        if(pt) deps = pt->run(deps);
        if(svgf) deps = svgf->run(deps);
    }
    deps = tonemap->run(deps);
    if(taa) deps = taa->run(deps);
    if(copy) deps = copy->run(deps);

    ctx->end_frame(deps);
}

void rc_renderer::reset_accumulation(bool)
{
    pt->reset_accumulated_samples();
}

}

