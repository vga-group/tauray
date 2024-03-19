#include "restir_renderer.hh"

namespace tr
{

restir_renderer::restir_renderer(context& ctx, const options& opt)
: ctx(&ctx), opt(opt)
{
    device& display_device = ctx.get_display_device();

    gbuffer_spec gs;
    gs.color_present = true;
    gs.depth_present = true;
    gs.albedo_present = true;
    gs.material_present = true;
    gs.normal_present = true;
    gs.screen_motion_present = true;
    gs.flat_normal_present = true;
    gs.emission_present = true;

    current_gbuffer.reset(device_mask::all(ctx), ctx.get_size(), ctx.get_display_count());
    prev_gbuffer.reset(device_mask::all(ctx), ctx.get_size(), ctx.get_display_count());

    scene_update.emplace(device_mask::all(ctx), opt.scene_options);

    raster_stage::options raster_opt;
    raster_opt.clear_color = true;
    raster_opt.clear_depth = true;
    raster_opt.sample_shading = false;
    raster_opt.pcf_samples = 0;
    raster_opt.omni_pcf_samples = 0;
    raster_opt.pcss_samples = 0;
    raster_opt.use_probe_visibility = false;
    raster_opt.sh_order = 0;
    raster_opt.force_alpha_to_coverage = true;
    raster_opt.output_layout = vk::ImageLayout::eGeneral;

    std::vector<gbuffer_target> gbuffer_block_targets;
    for(size_t i = 0; i < current_gbuffer.get_multiview_block_count(); ++i)
    {
        gbuffer_target mv_target = current_gbuffer.get_multiview_block_target(display_device.id, i);
        mv_target.color = render_target();
        gbuffer_block_targets.push_back(mv_target);
    }

    gbuffer_rasterizer.emplace(
        display_device, *scene_update,
        gbuffer_block_targets, raster_opt
    );

    gbuffer_target cur = current_gbuffer.get_array_target(display_device.id);
    gbuffer_target prev = prev_gbuffer.get_array_target(display_device.id);
    cur.set_layout(vk::ImageLayout::eGeneral);
    prev.set_layout(vk::ImageLayout::eGeneral);
    restir.emplace(display_device, *scene_update, cur, prev, opt.restir_options);

    std::vector<render_target> display = ctx.get_array_render_target();
    tonemap.emplace(
        display_device,
        cur.color,
        display,
        opt.tonemap_options
    );

    // Take out the things we don't need in the previous G-Buffer before
    // copying stuff over.
    cur.color = render_target();
    cur.screen_motion = render_target();
    copy.emplace(display_device, cur, prev);
}

void restir_renderer::set_scene(scene* s)
{
    scene_update->set_scene(s);
}

void restir_renderer::render()
{
    dependencies display_deps(ctx->begin_frame());
    uint32_t swapchain_index, frame_index;
    ctx->get_indices(swapchain_index, frame_index);

    dependencies deps = scene_update->run(last_frame_deps);
    last_frame_deps.clear();

    deps = gbuffer_rasterizer->run(deps);
    deps = restir->run(deps);
    deps = tonemap->run(deps);

    deps.concat(display_deps);
    deps = copy->run(deps);

    ctx->end_frame(deps);
    last_frame_deps = deps;
}

}
