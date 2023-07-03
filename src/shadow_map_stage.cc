#include "shadow_map_stage.hh"
#include "scene.hh"
#include "camera.hh"
#include "placeholders.hh"
#include "misc.hh"

namespace
{
using namespace tr;

// The shadow map renderer uses a custom camera that is not part of the scene.
struct camera_data_buffer
{
    mat4 view_proj;
};

struct push_constant_buffer
{
    uint32_t instance_id;
    float alpha_clip;
};

namespace shadow
{
    raster_shader_sources load_sources()
    {
        static bool loaded = false;
        static raster_shader_sources src;
        if(!loaded)
        {
            src = {{"shader/shadow_map.vert"}, {"shader/shadow_map.frag"}};
            loaded = true;
        }
        return src;
    }
}

}

namespace tr
{

shadow_map_stage::shadow_map_stage(
    device_data& dev,
    uvec4 local_rect,
    render_target& depth_buffer,
    const options& opt
):  stage(dev),
    gfx(dev, {
        depth_buffer.get_size(),
        local_rect,
        shadow::load_sources(),
        {
            // Texture samplers are binding 1.
            {"textures", (uint32_t)opt.max_samplers},
        },
        mesh::get_bindings(),
        {mesh::get_attributes()[0], mesh::get_attributes()[2]},
        {},
        raster_pipeline::pipeline_state::depth_attachment_state{
            depth_buffer,
            {
                {},
                depth_buffer.get_format(),
                vk::SampleCountFlagBits::e1,
                vk::AttachmentLoadOp::eClear,
                vk::AttachmentStoreOp::eStore,
                vk::AttachmentLoadOp::eDontCare,
                vk::AttachmentStoreOp::eDontCare,
                depth_buffer.get_layout(),
                depth_buffer.get_layout()
            }
        }
    }),
    opt(opt),
    camera_data(dev, sizeof(camera_data_buffer), vk::BufferUsageFlagBits::eUniformBuffer),
    shadow_timer(dev, "shadow map"),
    cur_scene(nullptr)
{
    scene::bind_placeholders(gfx, opt.max_samplers, 0);
}

void shadow_map_stage::set_scene(scene* s)
{
    cur_scene = s;

    clear_commands();
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        // Bind descriptors
        cur_scene->bind(gfx, i, -1);
        gfx.update_descriptor_set({
            {"camera", {camera_data[dev->index], 0, VK_WHOLE_SIZE}}
        }, i);

        // Record command buffer
        vk::CommandBuffer cb = begin_graphics();

        shadow_timer.begin(cb, i);
        camera_data.upload(dev->index, i, cb);

        gfx.begin_render_pass(cb, i);
        gfx.bind(cb, i);
        const std::vector<scene::instance>& instances = cur_scene->get_instances();

        for(size_t i = 0; i < instances.size(); ++i)
        {
            const scene::instance& inst = instances[i];
            const mesh* m = inst.m;
            vk::Buffer vertex_buffers[] = {m->get_vertex_buffer(dev->index)};
            vk::DeviceSize offsets[] = {0};
            cb.bindVertexBuffers(0, 1, vertex_buffers, offsets);
            cb.bindIndexBuffer(
                m->get_index_buffer(dev->index),
                0, vk::IndexType::eUint32
            );
            push_constant_buffer control;
            control.instance_id = i;
            control.alpha_clip =
                inst.mat && inst.mat->potentially_transparent() ? 0.5f : 1.0f;

            gfx.push_constants(cb, control);

            cb.drawIndexed(m->get_indices().size(), 1, 0, 0, 0);
        }
        cb.endRenderPass();
        shadow_timer.end(cb, i);
        end_graphics(cb, i);
    }
}

void shadow_map_stage::set_camera(const camera& cur_cam)
{
    this->cur_cam = cur_cam;
}

void shadow_map_stage::update(uint32_t frame_index)
{
    mat4 inv_view = cur_cam.get_global_transform();
    mat4 view = inverse(inv_view);
    mat4 projection = cur_cam.get_projection_matrix();

    camera_data.map<camera_data_buffer>(
        frame_index,
        [&](camera_data_buffer* cuni){
            cuni->view_proj = projection * view;
        }
    );
}

}
