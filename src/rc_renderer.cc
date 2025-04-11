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

    gbuffer_spec gs;
    gs.color_present = true;
    gs.set_all_usage(vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled);

    this->opt.scene_options.shadow_mapping = false;
    this->opt.scene_options.alloc_sh_grids = false;
    this->opt.scene_options.track_prev_tlas = false;
    this->opt.scene_options.shadow_mapping = true;

    this->opt.pt_options.distribution.size = ctx.get_size();
    this->opt.pt_options.distribution.strategy = DISTRIBUTION_DUPLICATE;
    this->opt.pt_options.active_viewport_count = 1;

    scene_update.emplace(dev, this->opt.scene_options);

    radiance_cascades_stage::options rc_opt;
    // Hardcoded for test.glb. TODO: Use scene bounding volume.
    rc_opt.volume = {vec3(-2), vec3(2)};

    std::vector<uint8_t> distance_field_data = load_binary_file("test/test-distance-field.raw");
    distance_field.emplace(texture(
        dev,
        uvec3(256),
        vk::Format::eR32Sfloat,
        distance_field_data.size(),
        distance_field_data.data(),
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
        vk::ImageLayout::eGeneral,
        true
    ));
    rc_opt.distance_field = &distance_field.value();

    gbuffer.reset(dev, ctx.get_size(), 1);
    gbuffer.add(gs, vk::ImageLayout::eGeneral);

    rc.emplace(dev, *scene_update, rc_opt);

    rc_visualizer_stage::options rcv_opt;
    rcv_opt.distance_field = &distance_field.value();
    gbuffer_target cur = gbuffer.get_layer_target(dev.id, 0);

    //rcv.emplace(*rc, cur.color, rcv_opt);
    cur = gbuffer.get_array_target(dev.id);

    this->opt.pt_options.rc_source = &*rc;
    pt.emplace(dev, *scene_update, cur, this->opt.pt_options);

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

void rc_renderer::set_scene(scene* s)
{
    scene_update->set_scene(s);
}

void rc_renderer::set_visualizer_pos(int cascade, int layer)
{
    //rcv->set_position(cascade, layer);
}

void rc_renderer::render()
{
    dependencies display_deps(ctx->begin_frame());
    uint32_t swapchain_index, frame_index;
    ctx->get_indices(swapchain_index, frame_index);

    dependencies deps = scene_update->run(display_deps);
    deps = rc->run(deps);
    pt->force_command_buffer_refresh();
    deps = pt->run(deps);
    //deps = rcv->run(deps);
    deps = tonemap->run(deps);

    ctx->end_frame(deps);
}

void rc_renderer::reset_accumulation(bool)
{
    pt->reset_accumulated_samples();
}

}

