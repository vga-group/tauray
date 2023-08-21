#include "temporal_reprojection_stage.hh"
#include "misc.hh"

namespace
{
using namespace tr;

shader_source load_source(const tr::temporal_reprojection_stage::options&)
{
    std::map<std::string, std::string> defines;
    return {"shader/temporal_reprojection.comp", defines};
}

struct push_constant_buffer
{
    pivec2 size;
    float temporal_ratio;
};

static_assert(sizeof(push_constant_buffer) <= 128);

}

namespace tr
{

temporal_reprojection_stage::temporal_reprojection_stage(
    device& dev,
    gbuffer_target& current_features,
    gbuffer_target& previous_features,
    const options& opt
):  single_device_stage(dev),
    comp(dev, compute_pipeline::params{load_source(opt), {}}),
    opt(opt),
    current_features(current_features),
    previous_features(previous_features),
    stage_timer(dev, "temporal reprojection (" + std::to_string(opt.active_viewport_count) + " viewports)")
{
    init_resources();
    record_command_buffers();
}


void temporal_reprojection_stage::init_resources()
{
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        comp.update_descriptor_set({
            {"current_color", {{}, current_features.color.view, vk::ImageLayout::eGeneral}},
            {"current_normal", {{}, current_features.normal.view, vk::ImageLayout::eGeneral}},
            {"current_pos", {{}, current_features.pos.view, vk::ImageLayout::eGeneral}},
            {"current_screen_motion", {{}, current_features.screen_motion.view, vk::ImageLayout::eGeneral}},

            {"previous_color", {{}, previous_features.color.view, vk::ImageLayout::eGeneral}},
            {"previous_normal", {{}, previous_features.normal.view, vk::ImageLayout::eGeneral}},
            {"previous_pos", {{}, previous_features.pos.view, vk::ImageLayout::eGeneral}}
        }, i);
    }
}

void temporal_reprojection_stage::record_command_buffers()
{
    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();

        stage_timer.begin(cb, dev->id, i);

        comp.bind(cb, i);

        uvec2 wg = (current_features.get_size()+15u)/16u;
        push_constant_buffer control;
        control.size = current_features.get_size();
        control.temporal_ratio = opt.temporal_ratio;

        comp.push_constants(cb, control);
        cb.dispatch(wg.x, wg.y, opt.active_viewport_count);

        stage_timer.end(cb, dev->id, i);
        end_compute(cb, i);
    }
}

}
