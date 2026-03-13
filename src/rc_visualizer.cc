#include "rc_visualizer.hh"

namespace
{
using namespace tr;

struct visualize_push_constant_buffer
{
    int display_w;
    int display_h;
    int cascade_w;
    int cascade_h;
    int cascade;
    int layer;
};

}

namespace tr
{

rc_visualizer_stage::rc_visualizer_stage(
    radiance_cascades_stage& rc,
    render_target& output,
    const options& opt
):  single_device_stage(*rc.dev),
    rc(&rc),
    output(output),
    opt(opt),
    desc(*rc.dev),
    visualize(*rc.dev),
    cascade(0),
    layer(0),
    stage_timer(*rc.dev, "radiance cascade visualize")
{
    std::map<std::string, std::string> defines;
    rc.add_defines(defines);

    shader_source src("shader/radiance_cascades_visualizer.comp", defines);
    desc.add(src);
    visualize.init(src, {&desc, &rc.ss->get_descriptors(), &rc.get_descriptors()});
}

void rc_visualizer_stage::set_position(int cascade, int layer)
{
    this->cascade = cascade;
    this->layer = layer;
}

void rc_visualizer_stage::update(uint32_t frame_index)
{
    clear_commands();

    vk::CommandBuffer cb = begin_compute();
    stage_timer.begin(cb, dev->id, frame_index);

    visualize.bind(cb);
    visualize.set_descriptors(cb, rc->ss->get_descriptors(), 0, 1);
    visualize.set_descriptors(cb, rc->get_descriptors(), 0, 2);
    desc.set_image(dev->id, "target", {{{}, output.view, vk::ImageLayout::eGeneral}});
    desc.set_image(dev->id, "distance_field", {{{}, opt.distance_field->get_image_view(dev->id), vk::ImageLayout::eGeneral}});
    desc.set_image(dev->id, "occupancy_map", {{{}, opt.occupancy_map->get_image_view(dev->id), vk::ImageLayout::eGeneral}});
    visualize.push_descriptors(cb, desc, 0);

    visualize_push_constant_buffer pc;
    pc.cascade = std::clamp(cascade, 0, int(rc->get_cascade_count())-1);

    uvec3 size = rc->get_cascade_size(pc.cascade);
    pc.display_w = output.size.x;
    pc.display_h = output.size.y;
    pc.cascade_w = size.x;
    pc.cascade_h = size.y;
    pc.layer = std::clamp(layer, 0, int(size.z)-1);
    visualize.push_constants(cb, pc);

    uvec2 wg = uvec2(output.size+7u)/8u;
    cb.dispatch(wg.x, wg.y, 1);

    stage_timer.end(cb, dev->id, frame_index);
    end_compute(cb, frame_index);
}

}
