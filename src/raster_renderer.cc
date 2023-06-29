#include "raster_renderer.hh"
#include "scene.hh"
#include "misc.hh"
#include "log.hh"
#include <iostream>

namespace tr
{

raster_renderer::raster_renderer(context& ctx, const options& opt)
:   ctx(&ctx), opt(opt), smr(ctx),
    post_processing(
        ctx.get_display_device(),
        ctx.get_size(),
        opt.post_process
    )
{
    skinning.reset(new skinning_stage(ctx.get_display_device(), opt.max_skinned_meshes));
    scene_update.reset(new scene_update_stage(ctx.get_display_device(), opt.scene_options));
    init_common_resources();
}

raster_renderer::~raster_renderer()
{
}

void raster_renderer::set_scene(scene* s)
{
    cur_scene = s;
    skinning->set_scene(s);

    // First, update without shadow maps
    s->set_shadow_map_renderer(nullptr);
    scene_update->set_scene(s);

    smr.set_scene(s);

    // Then, update scene for the shadow maps as well.
    s->set_shadow_map_renderer(&smr);
    scene_update->set_scene(s);

    if(envmap) envmap->set_scene(s);
    if(z_pass) z_pass->set_scene(s);
    if(raster) raster->set_scene(s);
    post_processing.set_scene(s);
}

void raster_renderer::render()
{
    dependencies deps(ctx->begin_frame());
    deps = skinning->run(deps);
    deps = scene_update->run(deps);

    deps = render_core(deps);

    ctx->end_frame(deps);
}

dependencies raster_renderer::render_core(dependencies deps)
{
    deps = smr.render(deps);
    deps.concat(post_processing.get_gbuffer_write_dependencies());

    deps = envmap->run(deps);
    if(z_pass) deps = z_pass->run(deps);
    deps = raster->run(deps);
    return post_processing.render(deps);
}

void raster_renderer::init_common_resources()
{
    device_data& d = ctx->get_display_device();

    int max_msaa = (int)get_max_available_sample_count(*ctx);
    int fixed_msaa = min((int)next_power_of_two(opt.msaa_samples), max_msaa);
    if(opt.msaa_samples != fixed_msaa)
    {
        TR_LOG("Sample count ", opt.msaa_samples,
            " is not available on this platform. Using ", fixed_msaa,
            " instead.");
        opt.msaa_samples = fixed_msaa;
    }

    gbuffer_spec spec;
    spec.color_present = true;
    spec.color_format = vk::Format::eR16G16B16A16Sfloat;
    spec.depth_present = true;
    spec.depth_format = vk::Format::eD32Sfloat;

    vk::ImageUsageFlags img_usage = vk::ImageUsageFlagBits::eStorage|
        vk::ImageUsageFlagBits::eTransferSrc|
        vk::ImageUsageFlagBits::eSampled|
        vk::ImageUsageFlagBits::eColorAttachment;

    spec.set_all_usage(img_usage);
    post_processing.set_gbuffer_spec(spec);
    spec.depth_usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;

    gbuffer = gbuffer_texture(
        d,
        ctx->get_size(),
        ctx->get_display_count(),
        (vk::SampleCountFlagBits)opt.msaa_samples
    );

    gbuffer.add(spec);

    std::vector<render_target> color_block_targets;
    std::vector<render_target> depth_block_targets;
    std::vector<gbuffer_target> gbuffer_block_targets;
    for(size_t i = 0; i < gbuffer.get_multiview_block_count(); ++i)
    {
        gbuffer_target mv_target = gbuffer.get_multiview_block_target(d.index, i);
        color_block_targets.push_back(mv_target.color);
        depth_block_targets.push_back(mv_target.depth);
        gbuffer_block_targets.push_back(mv_target);
    }
    envmap.reset(new envmap_stage(d, color_block_targets));
    if(opt.z_pre_pass)
    {
        z_pass.reset(new z_pass_stage(d, depth_block_targets));
    }

    raster_stage::options raster_opt = opt;
    raster_opt.clear_color = false;
    raster_opt.clear_depth = !opt.z_pre_pass;
    raster_opt.output_layout = vk::ImageLayout::eGeneral;
    raster.reset(new raster_stage(d, gbuffer_block_targets, raster_opt));

    gbuffer_target array_target = gbuffer.get_array_target(d.index);
    array_target.set_layout(vk::ImageLayout::eGeneral);

    post_processing.set_display(array_target);
}

}
