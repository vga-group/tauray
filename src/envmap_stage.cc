#include "envmap_stage.hh"
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
};

}

namespace tr
{

envmap_stage::envmap_stage(
    device_data& dev,
    const std::vector<render_target>& color_arrays
):  stage(dev),
    envmap_timer(dev, "envmap ("+ std::to_string(count_array_layers(color_arrays)) +" viewports)"),
    cur_scene(nullptr)
{
    set_scene(nullptr);
    for(const render_target& target: color_arrays)
    {
        array_pipelines.emplace_back(new raster_pipeline(dev, {
            target.size,
            uvec4(0, 0, target.size),
            {{"shader/envmap.vert"}, {"shader/envmap.frag"}},
            {},
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
            false, false, true
        }));
    }
}

void envmap_stage::set_scene(scene* s)
{
    cur_scene = s;
    clear_commands();
    if(!s) return;

    environment_map* envmap = cur_scene ? cur_scene->get_environment_map() : nullptr;

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
        envmap_timer.begin(cb, dev->index, i);

        size_t j = 0;
        for(std::unique_ptr<raster_pipeline>& gfx: array_pipelines)
        {
            // Bind descriptors
            cur_scene->bind(*gfx, i, j);
            j += gfx->get_multiview_layer_count();

            gfx->begin_render_pass(cb, i);
            gfx->bind(cb, i);

            control.screen_size = vec2(gfx->get_state().output_size);
            gfx->push_constants(cb, control);

            cb.draw(6, 1, 0, 0);

            gfx->end_render_pass(cb);
        }
        envmap_timer.end(cb, dev->index, i);
        end_graphics(cb, i);
    }
}

}

