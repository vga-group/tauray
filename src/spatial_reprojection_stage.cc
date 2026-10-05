#include "spatial_reprojection_stage.hh"
#include "misc.hh"
#include "camera.hh"
#include <algorithm>

namespace
{
using namespace tr;

struct camera_data_buffer
{
    pmat4 view_proj;
};

struct push_constant_buffer
{
    pvec4 default_value;
    pivec2 size;
    uint32_t source_count;
};

static_assert(sizeof(push_constant_buffer) <= 128);

}

namespace tr
{

spatial_reprojection_stage::spatial_reprojection_stage(
    device& dev,
    scene_stage& ss,
    gbuffer_target& target,
    const options& opt
):  single_device_stage(dev),
    ss(&ss),
    target_viewport(target),
    desc(dev),
    comp(dev),
    opt(opt),
    camera_data(
        dev,
        sizeof(camera_data_buffer) * opt.active_viewport_count,
        vk::BufferUsageFlagBits::eStorageBuffer
    ),
    stage_timer(
        dev,
        "spatial reprojection (from " +
        std::to_string(opt.active_viewport_count) + " to " +
        std::to_string(target.get_layer_count() - opt.active_viewport_count) +
        " viewports)"
    )
{
    shader_source src("shader/spatial_reprojection.comp");
    desc.add(src);
    comp.init(src, {&desc});

    target_viewport.set_layout(vk::ImageLayout::eGeneral);
    this->target_viewport.color.layout = vk::ImageLayout::eUndefined;

    clear_commands();
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();

        stage_timer.begin(cb, dev.id, i);

        target_viewport.color.transition_layout_temporary(
            cb, vk::ImageLayout::eGeneral, true
        );
        camera_data.upload(dev.id, i, cb);

        comp.bind(cb);
        desc.set_buffer("camera_data", camera_data);
        desc.set_image(dev.id, "color_tex", {{{}, target_viewport.color.view, vk::ImageLayout::eGeneral}});
        desc.set_image(dev.id, "normal_tex", {{{}, target_viewport.normal.view, vk::ImageLayout::eGeneral}});
        desc.set_image(dev.id, "position_tex", {{{}, target_viewport.pos.view, vk::ImageLayout::eGeneral}});
        comp.push_descriptors(cb, desc, 0);

        push_constant_buffer control;
        control.size = target_viewport.get_size();
        control.source_count = opt.active_viewport_count;
        control.default_value = vec4(NAN);

        uvec2 wg = (uvec2(control.size) + 15u)/16u;

        comp.push_constants(cb, control);
        cb.dispatch(wg.x, wg.y, target_viewport.get_layer_count() - control.source_count);

        stage_timer.end(cb, dev.id, i);
        end_compute(cb, i);
    }
}

void spatial_reprojection_stage::update(uint32_t frame_index)
{
    scene* cur_scene = ss->get_scene();
    std::vector<entity> cameras = get_sorted_cameras(*cur_scene);
    camera_data.foreach<camera_data_buffer>(
        frame_index,
        opt.active_viewport_count,
        [&](camera_data_buffer& data, size_t i){
            data.view_proj = cur_scene->get<camera>(cameras[i])->get_view_projection(
                *cur_scene->get<transformable>(cameras[i])
            );
        }
    );
}

}

