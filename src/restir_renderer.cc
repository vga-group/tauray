#include "restir_renderer.hh"
#include "vulkan/vulkan_format_traits.hpp"
#include "log.hh"

namespace tr
{

restir_renderer::restir_renderer(context& ctx, const options& opt)
: ctx(&ctx), opt(opt), per_view(ctx.get_display_count())
{
    device& display_device = ctx.get_display_device();
    std::vector<device>& devices = ctx.get_devices();
    size_t device_count = devices.size();

    if(device_count < ctx.get_display_count())
    {
        TR_WARN("Fewer GPUs than views; using only one GPU for all.");
        device_count = 1;
    }
    else device_count = std::min(device_count, ctx.get_display_count());

    if(!this->opt.restir_options.assume_unchanged_acceleration_structures)
        this->opt.scene_options.track_prev_tlas = true;

    gbuffer_spec gs;
    gs.color_present = true;
    gs.emission_present = true;
    gs.depth_present = true;
    gs.albedo_present = true;
    gs.material_present = true;
    gs.normal_present = true;
    gs.screen_motion_present = true;
    gs.flat_normal_present = true;
    gs.curvature_present = true;

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

    scene_update.emplace(
        device_count == 1 ? device_mask(display_device) : device_mask::all(ctx),
        this->opt.scene_options
    );

    per_device.resize(device_count);
    device_mask dev = device_mask::none(ctx);
    for(size_t i = 0; i < per_device.size(); ++i)
    {
        int view_count = device_count ==  1 ? ctx.get_display_count() : 1;
        dev.insert(devices[i].id);

        per_device[i].current_gbuffer.reset(devices[i], ctx.get_size(), view_count);
        per_device[i].current_gbuffer.add(gs, vk::ImageLayout::eGeneral);
        per_device[i].prev_gbuffer.reset(devices[i], ctx.get_size(), view_count);
        per_device[i].prev_gbuffer.add(gs, vk::ImageLayout::eGeneral);

    }

    for(size_t i = 0; i < ctx.get_display_count(); ++i)
    {
        int device_index = device_count == 1 ? 0 : i;
        int layer_index = device_count == 1 ? i : 0;
        per_device_data& data = per_device[device_index];
        bool is_display_device = devices[device_index].id == display_device.id;

        gbuffer_target cur = data.current_gbuffer.get_layer_target(devices[device_index].id, layer_index);
        gbuffer_target prev = data.prev_gbuffer.get_layer_target(devices[device_index].id, layer_index);

        per_view_stages& pv = per_view[i];
        pv.envmap.emplace(devices[device_index], *scene_update, cur.color, i);

        raster_stage::options raster_opt;
        raster_opt.clear_color = false;
        raster_opt.clear_depth = true;
        raster_opt.sample_shading = false;
        raster_opt.use_probe_visibility = false;
        raster_opt.sh_order = 0;
        raster_opt.estimate_indirect = false;
        raster_opt.force_alpha_to_coverage = true;
        raster_opt.base_camera_index = i;
        raster_opt.output_layout = vk::ImageLayout::eGeneral;
        raster_opt.estimate_direct = false;

        render_target diffuse = cur.diffuse;
        render_target reflection = cur.reflection;
        render_target temporal_gradient = cur.temporal_gradient;
        render_target confidence = cur.confidence;
        cur.diffuse = render_target();
        cur.reflection = render_target();
        cur.temporal_gradient = render_target();
        cur.confidence = render_target();

        pv.gbuffer_rasterizer.emplace(devices[device_index], *scene_update, cur, raster_opt);
        cur.diffuse = diffuse;
        cur.reflection = reflection;
        cur.temporal_gradient = temporal_gradient;
        cur.confidence = confidence;

        cur.color.layout = vk::ImageLayout::eGeneral;

        this->opt.restir_options.max_bounces = max(this->opt.restir_options.max_bounces, 1u);
        this->opt.restir_options.camera_index = i;
        //this->opt.restir_options.shift_map = restir_stage::RANDOM_REPLAY_SHIFT;
        pv.restir.emplace(devices[device_index], *scene_update, cur, prev, this->opt.restir_options);

        cur = data.current_gbuffer.get_array_target(devices[device_index].id);
        prev = data.prev_gbuffer.get_array_target(devices[device_index].id);

        std::vector<render_target> display = ctx.get_array_render_target();
        if(is_display_device)
        {
            this->opt.tonemap_options.limit_to_input_layer = layer_index;
            this->opt.tonemap_options.limit_to_output_layer = i;
            this->opt.tonemap_options.transition_output_layout = true;
            pv.tonemap.emplace(
                devices[device_index],
                cur.color,
                display,
                this->opt.tonemap_options
            );
        }
        else
        {
            pv.tmp_compressed_output_img.emplace(
                devices[i],
                ctx.get_size(),
                1,
                ctx.get_display_format(),
                0, nullptr,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc,
                vk::ImageLayout::eGeneral,
                vk::SampleCountFlagBits::e1
            );
            this->opt.tonemap_options.transition_output_layout = true;
            this->opt.tonemap_options.limit_to_input_layer = 0;
            this->opt.tonemap_options.limit_to_output_layer = 0;
            this->opt.tonemap_options.output_image_layout = vk::ImageLayout::eTransferSrcOptimal;
            render_target compressed_target = pv.tmp_compressed_output_img->get_array_render_target(devices[i].id);
            pv.tonemap.emplace(
                devices[device_index],
                cur.color,
                compressed_target,
                this->opt.tonemap_options
            );
        }

        // Take out the things we don't need in the previous G-Buffer before
        // copying stuff over.

        cur.color = render_target();
        cur.screen_motion = render_target();
        cur.temporal_gradient = render_target();
        cur.emission = render_target();
        prev.color = render_target();
        prev.screen_motion = render_target();
        prev.temporal_gradient = render_target();
        prev.emission = render_target();

        pv.copy.emplace(devices[device_index], cur, prev, layer_index, layer_index);

        if(!is_display_device)
        {
            vk::Image own_color = pv.tmp_compressed_output_img->get_image(devices[device_index].id);
            for(size_t j = 0; j < display.size(); ++j)
            {
                pv.transfer.push_back(create_device_transfer_interface(
                    devices[device_index],
                    display_device
                ));

                size_t pixel_format_size = vk::blockSize(ctx.get_display_format());
                std::vector<device_transfer_interface::image_transfer> images = {
                    device_transfer_interface::image_transfer{
                        own_color,
                        display[j].image,
                        pixel_format_size,
                        vk::ImageCopy{
                            {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                            {0,0,0},
                            {vk::ImageAspectFlagBits::eColor, 0, (uint32_t)i, 1},
                            {0,0,0},
                            {display[j].size.x, display[j].size.y, 1}
                        },
                        vk::ImageLayout::eTransferSrcOptimal,
                        ctx.get_expected_display_layout()
                    }
                };

                pv.transfer.back()->build(images, {});
            }
        }
    }
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

    for(auto& pv: per_view)
    {
        deps = pv.envmap->run(deps);
        deps = pv.gbuffer_rasterizer->run(deps);
        deps = pv.restir->run(deps);
        deps = pv.tonemap->run(deps);
        deps = pv.copy->run(deps);
    }

    for(auto& pv: per_view)
    {
        if(pv.transfer.size() != 0)
            deps = pv.transfer[swapchain_index]->run(deps, frame_index);
    }

    ctx->end_frame(deps);
}

}
