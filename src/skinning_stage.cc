#include "skinning_stage.hh"
#include "scene.hh"

namespace
{
using namespace tr;

struct push_constants
{
    uint32_t mesh_id;
    uint32_t model_id;
    uint32_t vertex_count;
};

}

namespace tr
{

skinning_stage::skinning_stage(device_data& dev, uint32_t max_instances)
:   stage(dev),
    comp(dev, {{"shader/skinning.comp"}, {
        {"source_data", max_instances},
        {"skin_data", max_instances},
        {"destination_data", max_instances},
        {"joint_data", max_instances}
    }}),
    cur_scene(nullptr),
    stage_timer(dev, "skinning"),
    max_instances(max_instances)
{
}

void skinning_stage::set_scene(scene* s)
{
    cur_scene = s;

    std::vector<model*> skinned_models;
    for(mesh_object* obj: cur_scene->get_mesh_objects())
    {
        model* m = const_cast<model*>(obj->get_model());
        if(!m) continue;
        if(m->has_joints_buffer())
            skinned_models.push_back(m);
    }

    std::vector<vk::DescriptorBufferInfo> dbi_source_data;
    std::vector<vk::DescriptorBufferInfo> dbi_destination_data;
    std::vector<vk::DescriptorBufferInfo> dbi_skin_data;
    std::vector<vk::DescriptorBufferInfo> dbi_joint_data;

    for(model* m: skinned_models)
    {
        for(auto& vg: *m)
        {
            mesh* dst = vg.m;
            mesh* src = dst->get_animation_source();
            vk::Buffer src_vertex_buffer = src->get_vertex_buffer(dev->index);
            vk::Buffer dst_vertex_buffer = dst->get_vertex_buffer(dev->index);
            vk::Buffer src_skin_buffer = src->get_skin_buffer(dev->index);
            dbi_source_data.push_back({src_vertex_buffer, 0, VK_WHOLE_SIZE});
            dbi_destination_data.push_back({dst_vertex_buffer, 0, VK_WHOLE_SIZE});
            dbi_skin_data.push_back({src_skin_buffer, 0, VK_WHOLE_SIZE});
        }
    }

    clear_commands();

    comp.update_descriptor_set({
        {"source_data", max_instances},
        {"destination_data", max_instances},
        {"skin_data", max_instances},
        {"joint_data", max_instances}
    });

    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        dbi_joint_data.clear();
        for(model* m: skinned_models)
        {
            dbi_joint_data.push_back({
                *m->get_joint_buffer(dev->index), 0, VK_WHOLE_SIZE
            });
        }

        // Bind descriptors
        comp.update_descriptor_set({
            {"source_data", dbi_source_data},
            {"destination_data", dbi_destination_data},
            {"skin_data", dbi_skin_data},
            {"joint_data", dbi_joint_data}
        }, i);

        // Record command buffer
        vk::CommandBuffer cb = begin_compute();
        stage_timer.begin(cb, i);
        for(model* m: skinned_models)
            m->upload_joints(cb, dev->index, i);

        comp.bind(cb, i);

        // Update vertex buffers
        for(uint32_t i = 0, j = 0; i < skinned_models.size(); ++i)
        {
            for(auto& vg: *skinned_models[i])
            {
                uint32_t vertex_count = vg.m->get_vertices().size();
                comp.push_constants(cb, push_constants{j, i, vertex_count});
                cb.dispatch((vertex_count+31u)/32u, 1, 1);
                j++;
            }
        }

        // Update acceleration structures
        if(dev->ctx->is_ray_tracing_supported())
        {
            // Barrier to ensure vertex buffers are updated by the time we try
            // to do BLAS updates.
            vk::MemoryBarrier barrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eAccelerationStructureWriteKHR
            );

            cb.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                {}, barrier, {}, {}
            );

            cur_scene->refresh_dynamic_acceleration_structures(dev->index, i, cb);
        }

        stage_timer.end(cb, i);
        end_compute(cb, i);
    }
}

scene* skinning_stage::get_scene()
{
    return cur_scene;
}

void skinning_stage::update(uint32_t frame_index)
{
    for(mesh_object* obj: cur_scene->get_mesh_objects())
    {
        model* m = const_cast<model*>(obj->get_model());
        if(!m) continue;
        if(m->has_joints_buffer())
            m->update_joints(dev->index, frame_index);
    }
}

}
