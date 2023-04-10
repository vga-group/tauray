#include "z_pass_stage.hh"
#include "mesh.hh"
#include "shader_source.hh"
#include "scene.hh"
#include "camera.hh"
#include "misc.hh"

namespace
{
using namespace tr;

// This must match the push_constant_buffer in shader/z_pass.glsl
struct push_constant_buffer
{
    uint32_t instance_id;
};

}

namespace tr
{

z_pass_stage::z_pass_stage(
    device_data& dev,
    const std::vector<render_target>& depth_buffer_arrays
):  stage(dev),
    cur_scene(nullptr),
    z_pass_timer(dev, "Z-pass (" + std::to_string(count_array_layers(depth_buffer_arrays)) + " viewports)")
{
    for(const render_target& depth_buffer: depth_buffer_arrays)
    {
        array_pipelines.emplace_back(new raster_pipeline(dev, {
            depth_buffer.get_size(),
            uvec4(0,0,depth_buffer.get_size()),
            {
                {"shader/z_pass.vert"},
                {"shader/z_pass.frag"}
            },
            {},
            mesh::get_bindings(),
            {mesh::get_attributes()[0]},
            {},
            raster_pipeline::pipeline_state::depth_attachment_state{
                depth_buffer,
                {
                    {},
                    depth_buffer.get_format(),
                    (vk::SampleCountFlagBits)depth_buffer.get_msaa(),
                    vk::AttachmentLoadOp::eClear,
                    vk::AttachmentStoreOp::eStore,
                    vk::AttachmentLoadOp::eDontCare,
                    vk::AttachmentStoreOp::eDontCare,
                    vk::ImageLayout::eUndefined,
                    vk::ImageLayout::eDepthStencilAttachmentOptimal
                }
            },
            false, false, true
        }));
    }
}

void z_pass_stage::set_scene(scene* s)
{
    cur_scene = s;

    clear_commands();
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        // Record command buffer
        vk::CommandBuffer cb = begin_graphics();
        z_pass_timer.begin(cb, i);

        size_t j = 0;
        for(std::unique_ptr<raster_pipeline>& gfx: array_pipelines)
        {
            // Bind descriptors
            cur_scene->bind(*gfx, i, j);
            j += gfx->get_multiview_layer_count();

            gfx->begin_render_pass(cb, i);
            gfx->bind(cb, i);

            const std::vector<scene::instance>& instances = cur_scene->get_instances();

            push_constant_buffer control;
            for(size_t i = 0; i < instances.size(); ++i)
            {
                const scene::instance& inst = instances[i];
                // Only render opaque things.
                if(inst.mat->potentially_transparent())
                    continue;
                const mesh* m = inst.m;
                vk::Buffer vertex_buffers[] = {m->get_vertex_buffer(dev->index)};
                vk::DeviceSize offsets[] = {0};
                cb.bindVertexBuffers(0, 1, vertex_buffers, offsets);
                cb.bindIndexBuffer(
                    m->get_index_buffer(dev->index),
                    0, vk::IndexType::eUint32
                );
                control.instance_id = i;

                gfx->push_constants(cb, control);

                cb.drawIndexed(m->get_indices().size(), 1, 0, 0, 0);
            }
            gfx->end_render_pass(cb);
        }
        z_pass_timer.end(cb, i);
        end_graphics(cb, i);
    }
}

scene* z_pass_stage::get_scene()
{
    return cur_scene;
}

}
