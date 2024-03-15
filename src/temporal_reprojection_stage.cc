#include "temporal_reprojection_stage.hh"
#include "misc.hh"

namespace
{
using namespace tr;

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
    desc(dev),
    comp(dev),
    opt(opt),
    stage_timer(dev, "temporal reprojection (" + std::to_string(opt.active_viewport_count) + " viewports)")
{
    shader_source src("shader/temporal_reprojection.comp");
    desc.add(src);
    comp.init(src, {&desc});

    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();

        stage_timer.begin(cb, dev.id, i);

        comp.bind(cb);

        desc.set_image(dev.id, "current_color", {{{}, current_features.color.view, vk::ImageLayout::eGeneral}});
        desc.set_image(dev.id, "current_normal", {{{}, current_features.normal.view, vk::ImageLayout::eGeneral}});
        desc.set_image(dev.id, "current_pos", {{{}, current_features.pos.view, vk::ImageLayout::eGeneral}});
        desc.set_image(dev.id, "current_screen_motion", {{{}, current_features.screen_motion.view, vk::ImageLayout::eGeneral}});
        desc.set_image(dev.id, "previous_color", {{{}, previous_features.color.view, vk::ImageLayout::eGeneral}});
        desc.set_image(dev.id, "previous_normal", {{{}, previous_features.normal.view, vk::ImageLayout::eGeneral}});
        desc.set_image(dev.id, "previous_pos", {{{}, previous_features.pos.view, vk::ImageLayout::eGeneral}});

        comp.push_descriptors(cb, desc, 0);

        uvec2 wg = (current_features.get_size()+15u)/16u;
        push_constant_buffer control;
        control.size = current_features.get_size();
        control.temporal_ratio = opt.temporal_ratio;

        comp.push_constants(cb, control);
        cb.dispatch(wg.x, wg.y, opt.active_viewport_count);

        stage_timer.end(cb, dev.id, i);
        end_compute(cb, i);
    }
}

}
