#include "multiview_composition_stage.hh"

namespace
{
using namespace tr;

struct push_constant_buffer
{
    puvec2 output_size;
    puvec2 viewport_size;
    puvec2 viewport_count;
};

}

namespace tr
{

multiview_composition_stage::multiview_composition_stage(
    device& dev,
    render_target& input,
    std::vector<render_target>& output_frames,
    const options& opt
):  single_device_stage(dev, single_device_stage::COMMAND_BUFFER_PER_FRAME),
    desc(dev),
    comp(dev),
    stage_timer(dev, "multiview composition")
{
    shader_source src("shader/multiview_composition.comp");
    desc.add(src);
    comp.init(src, {&desc});

    for(uint32_t i = 0; i < output_frames.size(); ++i)
    {
        // Record command buffer
        vk::CommandBuffer cb = begin_graphics();

        output_frames[i].transition_layout_temporary(cb, vk::ImageLayout::eGeneral, true);
        output_frames[i].layout = vk::ImageLayout::eGeneral;

        //stage_timer.begin(cb, i);

        comp.bind(cb);
        desc.set_image(dev.id, "in_color", {{{}, input.view, vk::ImageLayout::eGeneral}});
        desc.set_image(dev.id, "out_color", {{{}, output_frames[i].view, vk::ImageLayout::eGeneral}});
        comp.push_descriptors(cb, desc, 0);

        push_constant_buffer control;
        control.output_size = output_frames[i].size;
        control.viewport_size = input.size;
        control.viewport_count = opt.views;;

        comp.push_constants(cb, control);

        uvec2 wg = (output_frames[i].size+15u)/16u;
        cb.dispatch(wg.x, wg.y, 1);

        //stage_timer.end(cb, i);
        output_frames[i].transition_layout_temporary(cb, vk::ImageLayout::ePresentSrcKHR);
        output_frames[i].layout = vk::ImageLayout::ePresentSrcKHR;
        end_graphics(cb, 0, i);
    }
}

}

