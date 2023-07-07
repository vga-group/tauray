#include "rt_renderer.hh"
#include "scene.hh"
#include "misc.hh"
#ifdef WIN32
#include <vulkan/vulkan_win32.h>
#endif

namespace
{
using namespace tr;

template<typename Opt>
post_processing_renderer::options get_pp_opt(
    const Opt& opt
){
    post_processing_renderer::options pp_opt = opt.post_process;
    pp_opt.active_viewport_count = opt.active_viewport_count;
    return pp_opt;
}

}

namespace tr
{

template<typename Pipeline>
rt_renderer<Pipeline>::rt_renderer(context& ctx, const options& opt)
:   ctx(&ctx), opt(opt),
    post_processing(ctx.get_display_device(), ctx.get_size(), get_pp_opt(opt))
{
    this->opt.distribution.size = ctx.get_size();

    if(
        (opt.projection == camera::PERSPECTIVE ||
         opt.projection == camera::ORTHOGRAPHIC)
    ) use_raster_gbuffer = true;

    std::vector<device>& devices = ctx.get_devices();
    per_device.resize(devices.size());
    init_resources();
}

template<typename Pipeline>
rt_renderer<Pipeline>::~rt_renderer()
{
    ctx->sync();

    // Ensure each pipeline is deleted before the assets they may use
    stitch.reset();

    for(size_t i = 0; i < per_device.size(); ++i)
    {
        per_device_data& d = per_device[i];
        d.ray_tracer.reset();
        d.transfer.reset();
    }
    ctx->sync();
}

template<typename Pipeline>
void rt_renderer<Pipeline>::set_scene(scene* s)
{
    opt.projection = s->get_camera(0)->get_projection_type();
    if(s)
    {
        s->refresh_instance_cache(true);
        scene_update->set_scene(s);
        for(size_t i = 0; i < per_device.size(); ++i)
        {
            if(per_device[i].ray_tracer)
                per_device[i].ray_tracer->set_scene(s);
        }

        if(gbuffer_rasterizer)
            gbuffer_rasterizer->set_scene(s);

        post_processing.set_scene(s);
    }
}

template<typename Pipeline>
void rt_renderer<Pipeline>::reset_accumulation(bool reset_sample_counter)
{
    for(size_t i = 0; i < per_device.size(); ++i)
    {
        if(per_device[i].ray_tracer)
        {
            per_device[i].ray_tracer->reset_accumulated_samples();
            if(reset_sample_counter)
                per_device[i].ray_tracer->reset_sample_counter();
        }
    }
    accumulated_frames = 0;
    if(stitch)
        stitch->set_blend_ratio(1.0f);
}

template<typename Pipeline>
void rt_renderer<Pipeline>::render()
{
    dependencies display_deps(ctx->begin_frame());
    uint32_t swapchain_index, frame_index;
    ctx->get_indices(swapchain_index, frame_index);

    device& display_device = ctx->get_display_device();
    std::vector<device>& devices = ctx->get_devices();

    dependencies common_deps = scene_update->run(last_frame_deps);
    last_frame_deps.clear();

    for(size_t i = 0; i < devices.size(); ++i)
    {
        dependencies device_deps = common_deps;
        if(i == ctx->get_display_device().index)
            device_deps.concat(post_processing.get_gbuffer_write_dependencies());
        device_deps = per_device[i].ray_tracer->run(device_deps);
        last_frame_deps.concat(device_deps);

        if(i != display_device.index)
            display_deps.add(per_device[i].transfer->run(device_deps, frame_index));
        else
        {
            if(gbuffer_rasterizer)
                device_deps = gbuffer_rasterizer->run(device_deps);
            display_deps.concat(device_deps);
        }
    }

    display_deps.concat(post_processing.get_gbuffer_write_dependencies());

    if(stitch)
    {
        stitch->refresh_params();
        display_deps = stitch->run(display_deps);
        // Reset temporary blending from stitching
        stitch->set_blend_ratio(1.0f);
    }

    display_deps = post_processing.render(display_deps);
    ctx->end_frame(display_deps);
    accumulated_frames++;
}

template<typename Pipeline>
void rt_renderer<Pipeline>::set_device_workloads(const std::vector<double>& ratios)
{
    assert(ratios.size() == per_device.size());
    if(
        opt.distribution.strategy == DISTRIBUTION_SCANLINE ||
        opt.distribution.strategy == DISTRIBUTION_DUPLICATE
    ) return;

    device& display_device = ctx->get_display_device();

    double cumulative = 0;
    for(size_t i = 0; i < per_device.size(); ++i)
    {
        per_device_data& r = per_device[i];

        double ratio = clamp(ratios[i], 0.0, 1.0 - cumulative);
        r.dist = get_device_distribution_params(
            ctx->get_size(),
            opt.distribution.strategy,
            cumulative,
            ratio,
            i,
            ctx->get_devices().size(),
            i == ctx->get_display_device().index
        );
        cumulative += ratio;
        per_device[i].ray_tracer->reset_distribution_params(r.dist);
        if(i != display_device.index)
        {
            // Only the primary device renders in-place, so it's the only device
            // that can actually accumulate samples normally when workload ratio
            // changes.
            per_device[i].ray_tracer->reset_accumulated_samples();
            for(size_t j = 0; j < MAX_FRAMES_IN_FLIGHT; ++j)
                prepare_transfers(false);
        }
    }

    std::vector<distribution_params> dist;
    for(per_device_data& r: per_device)
        dist.push_back(r.dist);

    // Temporarily blend non-primary GPU accumulation from stitching stage
    // instead.
    if(opt.accumulate)
        stitch->set_blend_ratio(1.0f/(accumulated_frames+1));
    stitch->set_distribution_params(dist);
}

template<typename Pipeline>
void rt_renderer<Pipeline>::init_resources()
{
    gbuffer_spec spec, copy_spec;
    spec.color_present = true;
    spec.color_format = vk::Format::eR32G32B32A32Sfloat;
    post_processing.set_gbuffer_spec(spec);

    // Disable raster G-Buffer when nothing rasterizable is needed.
    if(
        spec.present_count()
        - spec.color_present
        - spec.direct_present
        - spec.diffuse_present == 0
    ) use_raster_gbuffer = false;

    vk::ImageUsageFlags img_usage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc;
    if(use_raster_gbuffer) img_usage |= vk::ImageUsageFlagBits::eColorAttachment;

    spec.set_all_usage(img_usage);
    spec.color_usage = vk::ImageUsageFlagBits::eStorage|
        vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc;

    if(use_raster_gbuffer)
    {
        spec.depth_present = true;
        spec.depth_usage = vk::ImageUsageFlagBits::eDepthStencilAttachment|
            vk::ImageUsageFlagBits::eTransferSrc;
        copy_spec.color_present = spec.color_present;
        copy_spec.color_format = spec.color_format;
        copy_spec.diffuse_present = spec.diffuse_present;
        copy_spec.diffuse_format = spec.diffuse_format;
        copy_spec.direct_present = spec.direct_present;
        copy_spec.direct_format = spec.direct_format;
    }
    else
        copy_spec = spec;

    copy_spec.set_all_usage(
        vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eStorage
    );

    gbuffer.reset(device_mask::all(*ctx), ctx->get_size(), ctx->get_display_count());
    gbuffer.add(spec);

    double even_workload_ratio = 1.0/ctx->get_devices().size();
    device& display_device = ctx->get_display_device();
    scene_update.reset(new scene_stage(device_mask::all(*ctx), opt.scene_options));

    for(device_id id = 0; id < per_device.size(); ++id)
    {
        device& d = ctx->get_devices()[id];
        bool is_display_device = id == ctx->get_display_device().index;
        per_device_data& r = per_device[id];
        r.dist = get_device_distribution_params(
            ctx->get_size(),
            opt.distribution.strategy,
            even_workload_ratio * id,
            even_workload_ratio,
            id,
            ctx->get_devices().size(),
            is_display_device
        );

        typename Pipeline::options rt_opt = opt;
        rt_opt.distribution = r.dist;
        rt_opt.active_viewport_count = opt.active_viewport_count;
        uvec2 max_target_size = get_distribution_target_max_size(rt_opt.distribution);

        if(!is_display_device)
        {
            r.gbuffer_copy.reset(display_device, max_target_size, ctx->get_display_count());
            r.gbuffer_copy.add(copy_spec);
        }

        gbuffer_target transfer_target = gbuffer.get_array_target(d.index);
        if(use_raster_gbuffer)
        {
            gbuffer_target limited_target;
            limited_target.color = transfer_target.color;
            limited_target.diffuse = transfer_target.diffuse;
            limited_target.direct = transfer_target.direct;
            transfer_target = limited_target;
        }
        transfer_target.set_layout(is_display_device ?
            vk::ImageLayout::eGeneral : vk::ImageLayout::eTransferSrcOptimal
        );

        r.ray_tracer.reset(new Pipeline(d, transfer_target, rt_opt));

        prepare_transfers(true);
    }

    size_t device_count = ctx->get_devices().size();
    if(device_count > 1)
    { // If multi-device, use parallel implementation
        std::vector<gbuffer_target> dimgs;
        for(size_t i = 0; i < per_device.size(); ++i)
        {
            gbuffer_target dimg;
            if(i == display_device.index)
                dimg = gbuffer.get_array_target(display_device.index);
            else
                dimg = per_device[i].gbuffer_copy.get_array_target(display_device.index);

            if(use_raster_gbuffer)
            {
                gbuffer_target limited_target;
                limited_target.color = dimg.color;
                limited_target.diffuse = dimg.diffuse;
                limited_target.direct = dimg.direct;
                dimg = limited_target;
            }
            dimgs.push_back(dimg);
        }

        std::vector<distribution_params> dist;
        for(per_device_data& r: per_device)
            dist.push_back(r.dist);

        stitch.reset(new stitch_stage(
            ctx->get_display_device(),
            ctx->get_size(),
            dimgs,
            dist,
            {
                opt.distribution.strategy,
                this->opt.active_viewport_count
            }
        ));
    }

    if(use_raster_gbuffer)
    {
        raster_stage::options raster_opt;
        raster_opt.max_samplers = opt.max_samplers;
        raster_opt.max_3d_samplers = 0;
        raster_opt.pcf_samples = 0;
        raster_opt.pcss_samples = 0;
        raster_opt.output_layout = vk::ImageLayout::eGeneral;
        raster_opt.force_alpha_to_coverage = opt.post_process.bmfr || opt.post_process.svgf_denoiser ? true : false;

        // We take out the things that the path tracer should produce.
        std::vector<gbuffer_target> gbuffer_block_targets;
        for(size_t i = 0; i < gbuffer.get_multiview_block_count(); ++i)
        {
            gbuffer_target mv_target = gbuffer.get_multiview_block_target(display_device.index, i);
            mv_target.color = render_target();
            mv_target.diffuse = render_target();
            mv_target.direct = render_target();
            gbuffer_block_targets.push_back(mv_target);
        }

        if(gbuffer_block_targets[0].entry_count() != 0)
        {
            gbuffer_rasterizer.reset(new raster_stage(
                display_device, gbuffer_block_targets, raster_opt
            ));
        }
    }

    gbuffer_target pp_target = gbuffer.get_array_target(display_device.index);
    pp_target.set_layout(vk::ImageLayout::eGeneral);
    post_processing.set_display(pp_target);
}

template<typename Pipeline>
void rt_renderer<Pipeline>::prepare_transfers(bool reserve)
{
    device& display_device = ctx->get_display_device();
    std::vector<device>& devices = ctx->get_devices();

    for(size_t i = 0; i < per_device.size(); ++i)
    {
        if(i == display_device.index)
            continue;
        auto& r = per_device[i];

        std::vector<device_transfer_interface::image_transfer> images;
        gbuffer_target target = gbuffer.get_array_target(i);
        gbuffer_target target_copy = r.gbuffer_copy.get_array_target(display_device.index);

        for(size_t i = 0; i < MAX_GBUFFER_ENTRIES; ++i)
        {
            if(!target_copy[i])
                continue;

            uvec2 max_target_size = get_distribution_target_max_size(r.dist);
            uvec2 target_size = get_distribution_target_size(r.dist);
            uvec2 transfer_size = reserve ? max_target_size : target_size;

            vk::ImageCopy region(
                {vk::ImageAspectFlagBits::eColor, 0, 0, (uint32_t)opt.active_viewport_count},
                {0,0,0},
                {vk::ImageAspectFlagBits::eColor, 0, 0, (uint32_t)opt.active_viewport_count},
                {0,0,0},
                {transfer_size.x, transfer_size.y, 1}
            );
            // TODO: This may be wasteful with some buffers if they're smaller
            // than this.
            unsigned color_format_size = sizeof(uint32_t)*4;
            images.push_back(device_transfer_interface::image_transfer{
                target[i].image,
                target_copy[i].image,
                color_format_size,
                region
            });
        }

        if(!r.transfer)
            r.transfer = create_device_transfer_interface(devices[i], display_device);

        if(reserve) r.transfer->reserve(images, {});
        else r.transfer->build(images, {});
    }

    // ... but also record the actual command buffers too.
    if(reserve) prepare_transfers(false);
}

template class rt_renderer<path_tracer_stage>;
template class rt_renderer<whitted_stage>;
template class rt_renderer<feature_stage>;
template class rt_renderer<direct_stage>;

}
