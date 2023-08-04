#include "taa_stage.hh"
#include "misc.hh"
#include "camera.hh"

namespace
{
using namespace tr;

shader_source load_source(const tr::taa_stage::options&)
{
    std::map<std::string, std::string> defines;
    return {"shader/taa.comp", defines};
}

struct push_constant_buffer
{
    pivec2 size;
    float blending_ratio;
};

static_assert(sizeof(push_constant_buffer) <= 128);

}

namespace tr
{

taa_stage::taa_stage(
    device& dev,
    scene_stage& ss,
    gbuffer_target& current_features,
    const options& opt
):  single_device_stage(dev),
    ss(&ss),
    comp(dev, compute_pipeline::params{load_source(opt), {}}),
    opt(opt),
    current_features(current_features),
    previous_color(
        dev,
        current_features.color.size,
        current_features.get_layer_count(),
        current_features.color.format,
        0, nullptr,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::SampleCountFlagBits::e1
    ),
    jitter_buffer(dev, sizeof(pvec4)*opt.active_viewport_count, vk::BufferUsageFlagBits::eStorageBuffer),
    history_sampler(
        dev, vk::Filter::eLinear, vk::Filter::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest, 0, true, false
    ),
    stage_timer(dev, "temporal antialiasing (" + std::to_string(opt.active_viewport_count) + " viewports)")
{
    init_resources();
    record_command_buffers();
}

void taa_stage::update(uint32_t frame_index)
{
    bool existing = jitter_history.size() != 0;
    jitter_history.resize(opt.active_viewport_count);

    for(size_t i = 0; i < opt.active_viewport_count; ++i)
    {
        vec4& v = jitter_history[i];
        vec2 cur_jitter = ss->get_scene()->get_camera(i)->get_jitter();
        vec2 prev_jitter = v;
        if(!existing) prev_jitter = cur_jitter;
        v = vec4(cur_jitter, prev_jitter);
    }

    jitter_buffer.update(frame_index, jitter_history.data());
}

void taa_stage::init_resources()
{
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        comp.update_descriptor_set({
            {"current_color", {{}, current_features.color.view, vk::ImageLayout::eGeneral}},
            {"current_screen_motion", {{}, current_features.screen_motion.view, vk::ImageLayout::eGeneral}},
            {"previous_color", {history_sampler.get_sampler(dev->index), previous_color.get_array_image_view(dev->index), vk::ImageLayout::eShaderReadOnlyOptimal}},
            {"jitter_info", {jitter_buffer[dev->index], 0, VK_WHOLE_SIZE}}
        }, i);
    }
}

void taa_stage::record_command_buffers()
{
    render_target previous_color_rt = previous_color.get_array_render_target(dev->index);

    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();

        stage_timer.begin(cb, dev->index, i);

        // Run the actual TAA code
        jitter_buffer.upload(dev->index, i, cb);

        comp.bind(cb, i);

        uvec2 wg = (current_features.get_size()+15u)/16u;
        push_constant_buffer control;
        control.size = current_features.get_size();
        control.blending_ratio = opt.blending_ratio;

        comp.push_constants(cb, control);
        cb.dispatch(wg.x, wg.y, opt.active_viewport_count);

        // Store the previous color buffer.

        current_features.color.transition_layout_temporary(cb, vk::ImageLayout::eTransferSrcOptimal);
        previous_color_rt.transition_layout_temporary(
            cb, vk::ImageLayout::eTransferDstOptimal, true, true
        );

        uvec3 size = uvec3(current_features.get_size(), 1);
        cb.copyImage(
            current_features.color.image,
            vk::ImageLayout::eTransferSrcOptimal,
            previous_color_rt.image,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageCopy(
                current_features.color.get_layers(),
                {0,0,0},
                previous_color_rt.get_layers(),
                {0,0,0},
                {size.x, size.y, size.z}
            )
        );

        vk::ImageLayout old_layout = current_features.color.layout;
        current_features.color.layout = vk::ImageLayout::eTransferSrcOptimal;
        current_features.color.transition_layout_temporary(cb, vk::ImageLayout::eGeneral);
        current_features.color.layout = old_layout;

        old_layout = previous_color_rt.layout;
        previous_color_rt.layout = vk::ImageLayout::eTransferDstOptimal;
        previous_color_rt.transition_layout_temporary(
            cb, vk::ImageLayout::eShaderReadOnlyOptimal, true, true
        );
        previous_color_rt.layout = old_layout;

        stage_timer.end(cb, dev->index, i);
        end_compute(cb, i);
    }
}

}

