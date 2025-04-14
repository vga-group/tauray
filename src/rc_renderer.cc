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
    std::vector<uint8_t> distance_field_data = load_binary_file(opt.distance_field_path);
    uint8_t* dfdata = distance_field_data.data();
    memcpy(&rc_opt.volume.min, dfdata, sizeof(float)*3);
    dfdata += sizeof(float)*3;
    memcpy(&rc_opt.volume.max, dfdata, sizeof(float)*3);
    dfdata += sizeof(float)*3;
    vec3 resolution;
    memcpy(&resolution, dfdata, sizeof(float)*3);
    dfdata += sizeof(float)*3;

    printf("Distance field:\n");
    printf("    Resolution: %u x %u x %u\n", uint(resolution.x), uint(resolution.y), uint(resolution.z));
    printf("    Range: [%f, %f, %f] - [%f, %f, %f]\n",
        rc_opt.volume.min.x,
        rc_opt.volume.min.y,
        rc_opt.volume.min.z,
        rc_opt.volume.max.x,
        rc_opt.volume.max.y,
        rc_opt.volume.max.z
    );

    rc_opt.log2_resolution = round(log2(resolution.x));

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

