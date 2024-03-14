#include "shadow_map_stage.hh"
#include "scene_stage.hh"
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
    uint32_t cam_index;
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

const uvec2 face_offset_mul[6] = {
    uvec2(0,0), uvec2(0,1),
    uvec2(1,0), uvec2(1,1),
    uvec2(2,0), uvec2(2,1)
};

}

namespace tr
{

bool compatible(const std::vector<scene_stage::shadow_map_instance>& a, const std::vector<scene_stage::shadow_map_instance>& b)
{
    if(a.size() != b.size())
         return false;

    for(size_t i = 0; i < a.size(); ++i)
    {
        if(
            a[i].atlas_index != b[i].atlas_index ||
            a[i].faces.size() != b[i].faces.size() ||
            a[i].cascades.size() != b[i].cascades.size()
        ) return false;
    }

    return true;
}

shadow_map_stage::shadow_map_stage(
    device& dev,
    scene_stage& ss,
    const options& opt
):  single_device_stage(dev),
    opt(opt),
    camera_data(dev, sizeof(camera_data_buffer), vk::BufferUsageFlagBits::eStorageBuffer),
    prev_atlas_size(0),
    shadow_timer(dev, "shadow map"),
    ss(&ss),
    scene_state_counter(0)
{
}

void shadow_map_stage::update(uint32_t frame_index)
{
    std::vector<scene_stage::shadow_map_instance> new_shadow_maps = ss->get_shadow_maps();
    if(!compatible(shadow_maps, new_shadow_maps))
        scene_state_counter = 0; // Force re-record if shadow maps changed
    shadow_maps = std::move(new_shadow_maps);

    size_t total_passes = 0;
    for(scene_stage::shadow_map_instance& info: shadow_maps)
        total_passes += info.cascades.size()+info.faces.size();

    camera_data.resize(sizeof(camera_data_buffer) * total_passes);

    camera_data.map<camera_data_buffer>(
        frame_index,
        [&](camera_data_buffer* cuni){
            size_t i = 0;
            for(scene_stage::shadow_map_instance& info: shadow_maps)
            {
                for(auto& face: info.faces)
                {
                    mat4 inv_view = face.transform;
                    mat4 view = inverse(inv_view);
                    mat4 projection = face.cam.get_projection_matrix();

                    cuni[i++].view_proj = projection * view;
                }

                for(scene_stage::shadow_map_instance::cascade& c: info.cascades)
                {
                    mat4 inv_view = c.cam.transform;
                    mat4 view = inverse(inv_view);
                    mat4 projection = c.cam.cam.get_projection_matrix();

                    cuni[i++].view_proj = projection * view;
                }
            }
        }
    );

    atlas* shadow_map_atlas = ss->get_shadow_map_atlas();
    if(shadow_map_atlas->get_size() != prev_atlas_size)
    {
        prev_atlas_size = shadow_map_atlas->get_size();
        clear_commands();
        scene_state_counter = 0; // Force refresh
        gfx.emplace(*dev, raster_pipeline::pipeline_state{
            uvec2(shadow_map_atlas->get_size()),
            uvec4(0, 0, shadow_map_atlas->get_size()),
            shadow::load_sources(),
            {},
            mesh::get_bindings(),
            {mesh::get_attributes()[0], mesh::get_attributes()[2]},
            {},
            raster_pipeline::pipeline_state::depth_attachment_state{
                shadow_map_atlas->get_layer_render_target(dev->id, 0),
                {
                    {},
                    shadow_map_atlas->get_format(),
                    vk::SampleCountFlagBits::e1,
                    vk::AttachmentLoadOp::eClear,
                    vk::AttachmentStoreOp::eStore,
                    vk::AttachmentLoadOp::eDontCare,
                    vk::AttachmentStoreOp::eDontCare,
                    vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::ImageLayout::eShaderReadOnlyOptimal
                }
            },
            false, false, false,
            {}, false, true,
            {&ss->get_descriptors()}
        });
    }

    if(ss->check_update(scene_stage::GEOMETRY, scene_state_counter))
    {
        clear_commands();
        const std::vector<scene_stage::instance>& instances = ss->get_instances();
        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            // Bind descriptors
            gfx->update_descriptor_set({
                {"shadow_camera", {camera_data[dev->id], 0, VK_WHOLE_SIZE}}
            }, i);

            // Record command buffer
            vk::CommandBuffer cb = begin_graphics();

            shadow_timer.begin(cb, dev->id, i);
            camera_data.upload(dev->id, i, cb);

            unsigned cam_index = 0;
            auto render_cam = [&](
                unsigned atlas_index, unsigned face_index, unsigned face_count
            ){
                uvec4 rect = shadow_map_atlas->get_rect_px(atlas_index);

                if(face_count == 6)
                {
                    rect.z /= 3;
                    rect.w /= 2;
                    uvec2 offset = face_offset_mul[face_index];
                    rect.x += offset.x * rect.z;
                    rect.y += offset.y * rect.w;
                }

                vk::Viewport vp(
                    rect.x, shadow_map_atlas->get_size().y-rect.y,
                    rect.z, -int(rect.w),
                    0.0f, 1.0f
                );
                cb.setViewport(0, 1, &vp);
                gfx->begin_render_pass(cb, i, rect);
                gfx->bind(cb, i);
                gfx->set_descriptors(cb, ss->get_descriptors(), 0, 1);

                for(size_t i = 0; i < instances.size(); ++i)
                {
                    const scene_stage::instance& inst = instances[i];
                    const mesh* m = inst.m;
                    vk::Buffer vertex_buffers[] = {m->get_vertex_buffer(dev->id)};
                    vk::DeviceSize offsets[] = {0};
                    cb.bindVertexBuffers(0, 1, vertex_buffers, offsets);
                    cb.bindIndexBuffer(
                        m->get_index_buffer(dev->id),
                        0, vk::IndexType::eUint32
                    );
                    push_constant_buffer control;
                    control.instance_id = i;
                    control.alpha_clip =
                        inst.mat && inst.mat->potentially_transparent() ? 0.5f : 1.0f;
                    control.cam_index = cam_index;

                    gfx->push_constants(cb, control);

                    cb.drawIndexed(m->get_indices().size(), 1, 0, 0, 0);
                }
                cb.endRenderPass();

                cam_index++;
            };

            for(scene_stage::shadow_map_instance& info: shadow_maps)
            {
                unsigned atlas_index = info.atlas_index;
                for(size_t face_index = 0; face_index < info.faces.size(); ++face_index)
                    render_cam(atlas_index, face_index, info.faces.size());
                atlas_index++;
                for(size_t cascade = 0; cascade < info.cascades.size(); ++cascade)
                    render_cam(atlas_index++, 0, 1);
            }

            shadow_timer.end(cb, dev->id, i);
            end_graphics(cb, i);
        }
    }
}

}
