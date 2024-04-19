#include "post_processing_renderer.hh"

namespace tr
{

post_processing_renderer::post_processing_renderer(
    device& dev, scene_stage& ss, uvec2 output_size, const options& opt
): dev(&dev), opt(opt), output_size(output_size), ss(&ss)
{
    // It's easiest to test WIP post processing pipelines by forcing their
    // options from here

    // this->opt.example_denoiser = example_denoiser_stage::options{
    //     1,
    //     4,
    //     true
    // };
}

post_processing_renderer::~post_processing_renderer()
{
    deinit_pipelines();
}

void post_processing_renderer::set_gbuffer_spec(gbuffer_spec& spec) const
{
    if(opt.temporal_reprojection.has_value())
    {
        spec.normal_present = true;
        spec.pos_present = true;
        spec.screen_motion_present = true;
    }

    if(opt.spatial_reprojection.has_value())
    {
        spec.normal_present = true;
        spec.pos_present = true;
    }

    if(opt.svgf_denoiser.has_value())
    {
        spec.normal_present = true;
        spec.screen_motion_present = true;
        spec.pos_present = true;
        spec.albedo_present = true;
        spec.diffuse_present = true;
        spec.linear_depth_present = true;
        spec.material_present = true;
        spec.reflection_present = true;
        spec.linear_depth_usage |= vk::ImageUsageFlagBits::eSampled;
    }

    if(opt.bmfr.has_value())
    {
        spec.normal_present = true;
        spec.screen_motion_present = true;
        spec.pos_present = true;
        spec.albedo_present = true;
        spec.diffuse_present = true;
    }

    if(opt.taa.has_value())
        spec.screen_motion_present = true;
}

void post_processing_renderer::set_display(gbuffer_target input_gbuffer)
{
    this->input_gbuffer = input_gbuffer;
    init_pipelines();
}

dependencies post_processing_renderer::get_gbuffer_write_dependencies() const
{
    uint32_t swapchain_index, frame_index;
    dev->ctx->get_indices(swapchain_index, frame_index);
    return delay_deps[(frame_index + 1) % MAX_FRAMES_IN_FLIGHT];
}

dependencies post_processing_renderer::render(dependencies deps)
{
    uint32_t swapchain_index, frame_index;
    dev->ctx->get_indices(swapchain_index, frame_index);
    bool first_frame = dev->ctx->get_frame_counter() <= 1;

    if(temporal_reprojection && !first_frame)
        deps = temporal_reprojection->run(deps);

    if(spatial_reprojection)
        deps = spatial_reprojection->run(deps);

    if(svgf)
        deps = svgf->run(deps);

    if (bmfr)
        deps = bmfr->run(deps);

    if(taa)
        deps = taa->run(deps);

    dependencies out_deps = tonemap->run(deps);

    if(delay)
        delay_deps[frame_index] = delay->run(deps);

    return out_deps;
}

void post_processing_renderer::init_pipelines()
{
    gbuffer_target input_target = input_gbuffer;
    vk::SampleCountFlagBits msaa = input_target.color.msaa;

    if(opt.spatial_reprojection.has_value())
    {
        opt.spatial_reprojection->active_viewport_count = opt.active_viewport_count;

        spatial_reprojection.reset(new spatial_reprojection_stage(
            *dev,
            *ss,
            input_target,
            opt.spatial_reprojection.value()
        ));
    }

    render_target in_color = input_target.color;
    render_target out_color = input_target.color;

    bool need_temporal = opt.temporal_reprojection.has_value() || opt.svgf_denoiser.has_value() || opt.bmfr.has_value();
    gbuffer_target prev_gbuffer;
    if(need_temporal)
    {
        gbuffer_target simplified = input_target;
        simplified.depth = {};
        delay.reset(new frame_delay_stage(*dev, simplified));

        prev_gbuffer = delay->get_output();
    }

    if(opt.temporal_reprojection.has_value())
    {
        opt.temporal_reprojection->active_viewport_count =
            opt.active_viewport_count;
        temporal_reprojection.reset(new temporal_reprojection_stage(
            *dev,
            input_target,
            prev_gbuffer,
            opt.temporal_reprojection.value()
        ));
    }

    bool need_pingpong =
        opt.svgf_denoiser.has_value() ||
        /* Add your new post processing pipeline check here */need_temporal;
    int pingpong_index = 0;

    if(need_pingpong)
    {
        for(std::unique_ptr<texture>& tex: pingpong)
        {
            tex.reset(new texture(
                *dev,
                output_size,
                input_gbuffer.get_layer_count(),
                vk::Format::eR16G16B16A16Sfloat,
                0, nullptr,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
                vk::ImageLayout::eGeneral,
                msaa
            ));
            if(pingpong_index == 0)
                out_color = tex->get_array_render_target(dev->id);

            pingpong_index++;
        }
        pingpong_index = 0;
    }

    /*
    auto swap_pingpong = [&](){
        if(!need_pingpong)
            throw std::runtime_error("You did need pingpong...");

        std::swap(in_color, out_color);
        if(pingpong_index == 0)
        {
            // Swap the initial color to second pingpong target
            out_color = pingpong[1]->get_array_render_target(dev->id);
        }
        pingpong_index++;
    };
    */

    // Add the new post processing pipelines here, don't forget to set
    // need_pingpong above to true when the stage is present.
    if(opt.svgf_denoiser.has_value())
    {
        opt.svgf_denoiser.value().active_viewport_count = opt.active_viewport_count;
        svgf.reset(new svgf_stage(
            *dev,
            *ss,
            input_target,
            prev_gbuffer,
            opt.svgf_denoiser.value()
        ));
    }
    if(opt.bmfr.has_value())
    {
        bmfr.reset(new bmfr_stage(
            *dev,
            input_target,
            prev_gbuffer,
            opt.bmfr.value()
        ));
    }

    if(opt.taa.has_value())
    {
        opt.taa->active_viewport_count = dev->ctx->get_display_count();
        gbuffer_target tmp = input_target;
        tmp.color = in_color;
        taa.reset(new taa_stage(
            *dev,
            *ss,
            tmp,
            opt.taa.value()
        ));
    }

    opt.tonemap.input_msaa = (int)msaa;
    opt.tonemap.transition_output_layout = true;
    std::vector<render_target> display = dev->ctx->get_array_render_target();
    tonemap.reset(new tonemap_stage(
        *dev,
        in_color,
        display,
        opt.tonemap
    ));
}

void post_processing_renderer::deinit_pipelines()
{
    temporal_reprojection.reset();
    spatial_reprojection.reset();
    svgf.reset();
    taa.reset();
    tonemap.reset();

    pingpong[0].reset();
    pingpong[1].reset();
}

}
