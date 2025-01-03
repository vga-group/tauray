#include "envmap_stage.hh"
#include "scene_stage.hh"
#include "environment_map.hh"
#include "scene.hh"
#include "placeholders.hh"
#include "camera.hh"
#include "misc.hh"

namespace
{
using namespace tr;

struct push_constant_buffer
{
    pvec4 environment_factor;
    pvec2 screen_size;
    // -1 for no environment map
    int environment_proj;
    int base_camera_index;
};

}

namespace tr
{

envmap_stage::envmap_stage(
    device& dev,
    scene_stage& ss,
    const std::vector<render_target>& color_arrays
):  single_device_stage(dev),
    envmap_timer(dev, "envmap ("+ std::to_string(count_array_layers(color_arrays)) +" viewports)"),
    scene_state_counter(0),
    base_camera_index(0),
    ss(&ss)
{
    for(const render_target& target: color_arrays)
    {
        array_pipelines.emplace_back(new raster_pipeline(dev));
        array_pipelines.back()->init(raster_pipeline::pipeline_state{
            target.size,
            uvec4(0, 0, target.size),
            {{"shader/envmap.vert"}, {"shader/envmap.frag"}},
            {&ss.get_descriptors()},
            {}, {},
            {{
                target,
                {
                    {},
                    target.format,
                    target.msaa,
                    vk::AttachmentLoadOp::eDontCare,
                    vk::AttachmentStoreOp::eStore,
                    vk::AttachmentLoadOp::eDontCare,
                    vk::AttachmentStoreOp::eDontCare,
                    vk::ImageLayout::eUndefined,
                    vk::ImageLayout::eColorAttachmentOptimal
                }
            }},
            {},
            false, false, true,
            {}, false
        });
    }
}

envmap_stage::envmap_stage(
    device& dev,
    scene_stage& ss,
    render_target& color_target,
    unsigned base_camera_index
):  single_device_stage(dev),
    envmap_timer(dev, "envmap"),
    scene_state_counter(0),
    base_camera_index(base_camera_index),
    ss(&ss)
{
    array_pipelines.emplace_back(new raster_pipeline(dev));
    array_pipelines.back()->init(raster_pipeline::pipeline_state{
        color_target.size,
        uvec4(0, 0, color_target.size),
        {{"shader/envmap.vert"}, {"shader/envmap.frag"}},
        {&ss.get_descriptors()},
        {}, {},
        {{
            color_target,
            {
                {},
                color_target.format,
                color_target.msaa,
                vk::AttachmentLoadOp::eDontCare,
                vk::AttachmentStoreOp::eStore,
                vk::AttachmentLoadOp::eDontCare,
                vk::AttachmentStoreOp::eDontCare,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal
            }
        }},
        {},
        false, false, true,
        {}, false
    });
    color_target.layout = vk::ImageLayout::eColorAttachmentOptimal;
}

void envmap_stage::update(uint32_t)
{
    if(!ss->check_update(scene_stage::ENVMAP, scene_state_counter))
        return;

    clear_commands();
    environment_map* envmap = ss->get_environment_map();

    push_constant_buffer control;
    if(envmap)
    {
        control.environment_factor = vec4(envmap->get_factor(), 1);
        control.environment_proj = (int)envmap->get_projection();
    }
    else
    {
        control.environment_factor = vec4(0,0,0,1);
        control.environment_proj = -1;
    }

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        // Record command buffer
        vk::CommandBuffer cb = begin_graphics();
        envmap_timer.begin(cb, dev->id, i);

        size_t j = base_camera_index;
        for(std::unique_ptr<raster_pipeline>& gfx: array_pipelines)
        {
            gfx->begin_render_pass(cb, i);
            gfx->bind(cb);
            gfx->set_descriptors(cb, ss->get_descriptors(), 0, 0);

            control.screen_size = vec2(gfx->get_state().output_size);
            control.base_camera_index = j;
            gfx->push_constants(cb, control);

            cb.draw(6, 1, 0, 0);

            gfx->end_render_pass(cb);
            j += gfx->get_multiview_layer_count();
        }
        envmap_timer.end(cb, dev->id, i);
        end_graphics(cb, i);
    }
}

}

