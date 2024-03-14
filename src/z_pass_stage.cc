#include "z_pass_stage.hh"
#include "mesh.hh"
#include "shader_source.hh"
#include "scene_stage.hh"
#include "camera.hh"
#include "misc.hh"

namespace
{
using namespace tr;

// This must match the push_constant_buffer in shader/z_pass.glsl
struct push_constant_buffer
{
    uint32_t instance_id;
    int32_t base_camera_index;
};

}

namespace tr
{

z_pass_stage::z_pass_stage(
    device& dev,
    scene_stage& ss,
    const std::vector<render_target>& depth_buffer_arrays
):  single_device_stage(dev),
    ss(&ss),
    z_pass_timer(dev, "Z-pass (" + std::to_string(count_array_layers(depth_buffer_arrays)) + " viewports)"),
    scene_state_counter(0)
{
    for(const render_target& depth_buffer: depth_buffer_arrays)
    {
        array_pipelines.emplace_back(new raster_pipeline(dev, {
            depth_buffer.size,
            uvec4(0,0,depth_buffer.size),
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
                    depth_buffer.format,
                    (vk::SampleCountFlagBits)depth_buffer.msaa,
                    vk::AttachmentLoadOp::eClear,
                    vk::AttachmentStoreOp::eStore,
                    vk::AttachmentLoadOp::eDontCare,
                    vk::AttachmentStoreOp::eDontCare,
                    vk::ImageLayout::eUndefined,
                    vk::ImageLayout::eDepthStencilAttachmentOptimal
                }
            },
            false, false, true,
            {}, false, false, {&ss.get_descriptors()}
        }));
    }
}

void z_pass_stage::update(uint32_t)
{
    if(!ss->check_update(scene_stage::GEOMETRY, scene_state_counter))
        return;

    clear_commands();
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        // Record command buffer
        vk::CommandBuffer cb = begin_graphics();
        z_pass_timer.begin(cb, dev->id, i);

        size_t j = 0;
        for(std::unique_ptr<raster_pipeline>& gfx: array_pipelines)
        {
            // Bind descriptors
            gfx->begin_render_pass(cb, i);
            gfx->bind(cb, i);
            gfx->set_descriptors(cb, ss->get_descriptors(), 0, 0);

            const std::vector<scene_stage::instance>& instances = ss->get_instances();

            push_constant_buffer control;
            for(size_t i = 0; i < instances.size(); ++i)
            {
                const scene_stage::instance& inst = instances[i];
                // Only render opaque things.
                if(inst.mat->potentially_transparent())
                    continue;
                const mesh* m = inst.m;
                vk::Buffer vertex_buffers[] = {m->get_vertex_buffer(dev->id)};
                vk::DeviceSize offsets[] = {0};
                cb.bindVertexBuffers(0, 1, vertex_buffers, offsets);
                cb.bindIndexBuffer(
                    m->get_index_buffer(dev->id),
                    0, vk::IndexType::eUint32
                );
                control.instance_id = i;
                control.base_camera_index = j;

                gfx->push_constants(cb, control);

                cb.drawIndexed(m->get_indices().size(), 1, 0, 0, 0);
            }
            gfx->end_render_pass(cb);
            j += gfx->get_multiview_layer_count();
        }
        z_pass_timer.end(cb, dev->id, i);
        end_graphics(cb, i);
    }
}

}
