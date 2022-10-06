#include "stitch_stage.hh"
#include "misc.hh"
#include <iostream>

namespace
{
using namespace tr;

namespace scanline
{
    shader_source load_source() { return {"shader/stitch_scanline.comp"}; }

    struct push_constant_buffer
    {
        puvec2 size;
        int device_count;
        int primary_index;
        int subimage_count;
        int subimage_index;
    };
}

namespace shuffled_strips
{
    shader_source load_source() { return {"shader/stitch_shuffled_strips.comp"}; }

    struct push_constant_buffer
    {
        puvec2 size;
        int start_p_offset;
        int count;
        unsigned int input_img_id;
        unsigned int output_img_id;
        unsigned int shuffled_strips_b;
    };
}

}

shader_source load_source(distribution_strategy s)
{
    switch(s)
    {
    case distribution_strategy::DISTRIBUTION_DUPLICATE:
        return scanline::load_source();
        break;
    case distribution_strategy::DISTRIBUTION_SCANLINE:
        return scanline::load_source();
        break;
    case distribution_strategy::DISTRIBUTION_SHUFFLED_STRIPS:
        return shuffled_strips::load_source();
        break;
    default:
        return scanline::load_source();
        break;
    }
}

void record_shuffled_strips_commands(
    uvec2 size,
    unsigned int primary_index,
    const std::vector<gbuffer_target>& images,
    vk::CommandBuffer cb,
    compute_pipeline &comp,
    const std::vector<distribution_params>& params,
    size_t active_viewports
){
    shuffled_strips::push_constant_buffer control;
    control.size = size;
    control.input_img_id = 0;
    control.output_img_id = 0;
    control.shuffled_strips_b = calculate_shuffled_strips_b(size);

    for(size_t img_idx = 0; img_idx < images.size(); ++img_idx)
    {
        control.output_img_id = 0;
        control.start_p_offset = params[img_idx].index;
        control.count = params[img_idx].count;

        if(img_idx != primary_index)
        {
            images[img_idx].visit([&](const render_target&){
                comp.push_constants(cb, control);
                unsigned int wg = (control.count+255)/256;
                cb.dispatch(wg, 1, active_viewports);
                control.input_img_id++;
                control.output_img_id++;
            });
        }
    }
}


namespace tr
{

stitch_stage::stitch_stage(
    device_data& dev,
    uvec2 size,
    const std::vector<gbuffer_target>& images,
    const std::vector<distribution_params>& params,
    const options& opt
):  stage(dev),
    comp(dev, compute_pipeline::params{
    load_source(opt.strategy),
        {
            {"input_images", (uint32_t)images[0].entry_count()*(images.size()-1)},
            {"output_images", (uint32_t)images[0].entry_count()},
        }
    }),
    opt(opt),
    size(size),
    images(images),
    params(params),
    stitch_timer(dev, "stitch (" + std::to_string(opt.active_viewport_count) + " viewports)")
{
    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        // Bind descriptors
        std::vector<vk::DescriptorImageInfo> input_images;
        std::vector<vk::DescriptorImageInfo> output_images;

        for(size_t j = 0; j < images.size(); ++j)
        {
            auto* target_vec = params[j].primary ?
                &output_images : &input_images;

            images[j].visit([&](const render_target& img){
                target_vec->push_back({{}, img[i].view, vk::ImageLayout::eGeneral});
            });
        }

        comp.update_descriptor_set({
            {"input_images", input_images},
            {"output_images", output_images}
        }, i);
    }
    record_commands();
}

void stitch_stage::set_distribution_params(
    const std::vector<distribution_params>& params
){
    this->params = params;
    record_commands();
}

void stitch_stage::record_commands()
{
    clear_commands();
    uint32_t primary_index = 0;
    while(!params[primary_index].primary)
        primary_index++;

    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        // Record command buffer
        vk::CommandBuffer cb = begin_compute();
        stitch_timer.begin(cb, i);

        comp.bind(cb, i);

        int subimage_index = 0;
        switch(opt.strategy)
        {
        case distribution_strategy::DISTRIBUTION_SHUFFLED_STRIPS:
            record_shuffled_strips_commands(
                size, primary_index,
                images, cb, comp, params,
                opt.active_viewport_count
            );
            break;
        default :
            images[0].visit([&](const render_target&){
                scanline::push_constant_buffer control;
                control.size = size;
                control.device_count = images.size();
                control.primary_index = primary_index;
                control.subimage_count = images[0].entry_count();
                control.subimage_index = subimage_index++;

                comp.push_constants(cb, control);

                uvec2 wg = (uvec2(size.x, (size.y+1)/images.size())+15u)/16u;
                cb.dispatch(wg.x, wg.y, (images.size()-1)*opt.active_viewport_count );
            });
            break;
        }

        stitch_timer.end(cb, i);
        end_compute(cb, i);
    }
}

}
