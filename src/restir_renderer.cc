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

    current_gbuffer.reset(display_device, ctx.get_size(), ctx.get_display_count());
    current_gbuffer.add(gs, vk::ImageLayout::eGeneral);
    prev_gbuffer.reset(display_device, ctx.get_size(), ctx.get_display_count());
    prev_gbuffer.add(gs, vk::ImageLayout::eGeneral);

    if(this->opt.restir_options.shade_all_explicit_lights)
        this->opt.scene_options.shadow_mapping = true;

    scene_update.emplace(display_device, this->opt.scene_options);

    if(this->opt.restir_options.shade_all_explicit_lights)
        sms.emplace(display_device, *scene_update, shadow_map_stage::options{});

    gbuffer_target cur = current_gbuffer.get_layer_target(display_device.id, 0);
    gbuffer_target prev = prev_gbuffer.get_layer_target(display_device.id, 0);

    envmap.emplace(display_device, *scene_update, cur.color, 0);

    raster_stage::options raster_opt;
    raster_opt.clear_color = false;
    raster_opt.clear_depth = true;
    raster_opt.sample_shading = false;
    raster_opt.filter = opt.sm_filter;
    raster_opt.use_probe_visibility = false;
    raster_opt.sh_order = 0;
    raster_opt.estimate_indirect = false;
    raster_opt.force_alpha_to_coverage = true;
    raster_opt.output_layout = vk::ImageLayout::eGeneral;

    render_target color = cur.color;
    if(!this->opt.restir_options.shade_all_explicit_lights)
        cur.color = render_target();
    gbuffer_rasterizer.emplace(display_device, *scene_update, cur, raster_opt);
    if(!this->opt.restir_options.shade_all_explicit_lights)
        cur.color = color;

    cur.color.layout = vk::ImageLayout::eGeneral;

    this->opt.restir_options.max_bounces = max(this->opt.restir_options.max_bounces, 1u);
    this->opt.restir_options.temporal_reuse = true;
    this->opt.restir_options.spatial_sample_oriented_disk = false;
    this->opt.restir_options.spatial_samples = 4;
    this->opt.restir_options.shift_map = restir_stage::RECONNECTION_SHIFT;
    //this->opt.restir_options.shift_map = restir_stage::RANDOM_REPLAY_SHIFT;
    restir.emplace(display_device, *scene_update, cur, prev, this->opt.restir_options);

    cur = current_gbuffer.get_array_target(display_device.id);
    cur.set_layout(vk::ImageLayout::eShaderReadOnlyOptimal);
    cur.color.layout = this->opt.restir_options.shade_all_explicit_lights ?
        vk::ImageLayout::eGeneral :
        vk::ImageLayout::eColorAttachmentOptimal;
    cur.diffuse.layout = vk::ImageLayout::eGeneral;
    cur.reflection.layout = vk::ImageLayout::eGeneral;

    std::vector<render_target> display = ctx.get_array_render_target();
    tonemap.emplace(
        display_device,
        cur.color,
        display,
        opt.tonemap_options
    );

    // Take out the things we don't need in the previous G-Buffer before
    // copying stuff over.
    prev = prev_gbuffer.get_array_target(display_device.id);

    cur.color = render_target();
    cur.screen_motion = render_target();
    prev.color = render_target();
    prev.screen_motion = render_target();

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

    dependencies deps = scene_update->run(display_deps);
    if(opt.restir_options.shade_all_explicit_lights)
        deps = sms->run(deps);

    deps = envmap->run(deps);
    deps = gbuffer_rasterizer->run(deps);
    deps = restir->run(deps);
    deps = tonemap->run(deps);
    deps = copy->run(deps);

    ctx->end_frame(deps);
}

}
