#include "looking_glass_composition_stage.hh"

namespace
{
using namespace tr;

struct push_constant_buffer
{
    pvec4 calibration_info;
    puvec2 output_size;
    puvec2 viewport_size;
    uint32_t viewport_count;
};

}

namespace tr
{

looking_glass_composition_stage::looking_glass_composition_stage(
    device_data& dev,
    render_target& input,
    std::vector<render_target>& output_frames,
    const options& opt
):  stage(dev, stage::COMMAND_BUFFER_PER_SWAPCHAIN_IMAGE),
    comp(dev, compute_pipeline::params{
        {"shader/looking_glass_composition.comp"},
        {},
        (uint32_t)output_frames.size()
    }),
    input_sampler(
        dev, vk::Filter::eLinear, vk::Filter::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest,
        0, true, false
    ),
    stage_timer(dev, "looking glass composition")
{
    for(uint32_t i = 0; i < output_frames.size(); ++i)
    {
        // Bind descriptors
        comp.update_descriptor_set({
            {"in_color", {input_sampler.get_sampler(dev.index), input.view, vk::ImageLayout::eShaderReadOnlyOptimal}},
            {"out_color", {{}, output_frames[i].view, vk::ImageLayout::eGeneral}}
        }, i);

        // Record command buffer
        vk::CommandBuffer cb = begin_graphics();

        input.transition_layout_temporary(cb, vk::ImageLayout::eShaderReadOnlyOptimal, true, true);
        output_frames[i].transition_layout_temporary(cb, vk::ImageLayout::eGeneral, true);
        output_frames[i].layout = vk::ImageLayout::eGeneral;

        //stage_timer.begin(cb, i);

        comp.bind(cb, i);

        push_constant_buffer control;
        control.output_size = output_frames[i].size;
        control.viewport_size = input.size;
        control.viewport_count = opt.viewport_count;
        control.calibration_info = vec4(
            opt.pitch,
            opt.tilt * opt.pitch,
            opt.pitch/(3.0f * control.output_size.x),
            -opt.center
        );
        if(opt.invert)
            control.calibration_info = -control.calibration_info;

        comp.push_constants(cb, control);

        uvec2 wg = (output_frames[i].size+15u)/16u;
        cb.dispatch(wg.x, wg.y, 1);

        //stage_timer.end(cb, i);
        output_frames[i].transition_layout_temporary(cb, vk::ImageLayout::ePresentSrcKHR);
        output_frames[i].layout = vk::ImageLayout::ePresentSrcKHR;
        end_graphics(cb, 0, i);
    }
    input.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
}

}

