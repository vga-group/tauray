#include "aabb_scene.hh"
#include "misc.hh"

namespace tr
{

aabb_scene::aabb_scene(
    device_mask dev,
    const char* timer_name,
    size_t sbt_offset,
    size_t max_capacity
):  max_capacity(max_capacity), sbt_offset(sbt_offset),
    blas_update_timer(dev, timer_name),
    as_update(dev)
{
    init_acceleration_structures();
}

size_t aabb_scene::get_max_capacity() const
{
    return max_capacity;
}

void aabb_scene::update_acceleration_structures(
    device_id id,
    uint32_t frame_index,
    bool& need_scene_reset,
    bool& command_buffers_outdated
){
    auto& as = as_update[id];
    auto& f = as.per_frame[frame_index];

    // Update area point light buffer
    aabb_buffer.map<vk::AabbPositionsKHR>(
        frame_index,
        [&](vk::AabbPositionsKHR* aabb){
            f.aabb_count = get_aabbs(aabb);
        }
    );

    need_scene_reset |= as.scene_reset_needed;
    command_buffers_outdated |= f.command_buffers_outdated;

    as.scene_reset_needed = false;
    f.command_buffers_outdated = false;
}

void aabb_scene::record_acceleration_structure_build(
    vk::CommandBuffer& cb,
    device_id id,
    uint32_t frame_index,
    bool update_only
){
    auto& as = as_update[id];
    auto& f = as.per_frame[frame_index];

    blas_update_timer.begin(cb, id, frame_index);
    aabb_buffer.upload(id, frame_index, cb);

    blas->rebuild(
        id,
        frame_index,
        cb,
        {bottom_level_acceleration_structure::entry{nullptr, f.aabb_count, &aabb_buffer, mat4(1.0f), true}},
        update_only
    );
    blas_update_timer.end(cb, id, frame_index);
}

void aabb_scene::add_acceleration_structure_instances(
    vk::AccelerationStructureInstanceKHR* instances,
    device_id id,
    uint32_t frame_index,
    size_t& instance_index,
    size_t capacity
) const
{
    auto& as = as_update[id];
    auto& f = as.per_frame[frame_index];

    if(f.aabb_count != 0 && instance_index < capacity)
    {
        vk::AccelerationStructureInstanceKHR& inst = instances[instance_index++];
        inst = vk::AccelerationStructureInstanceKHR(
            {}, instance_index, 1<<1, sbt_offset,
            vk::GeometryInstanceFlagBitsKHR::eTriangleCullDisable,
            blas->get_blas_address(id)
        );
        mat4 id_mat = mat4(1.0f);
        memcpy((void*)&inst.transform, (void*)&id_mat, sizeof(inst.transform));
    }
}

void aabb_scene::init_acceleration_structures()
{
    if(!as_update.get_context()->is_ray_tracing_supported()) return;

    aabb_buffer = gpu_buffer(
        as_update.get_mask(), max_capacity * sizeof(vk::AabbPositionsKHR),
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eShaderDeviceAddress|
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
    );

    blas.emplace(
        as_update.get_mask(),
        std::vector<bottom_level_acceleration_structure::entry>{
            {nullptr, max_capacity, &aabb_buffer, mat4(1.0f), true}
        },
        false, true, false
    );
}

void aabb_scene::invalidate_acceleration_structures()
{
    as_update([&](device&, as_update_data& as){
        as.scene_reset_needed = true;
        for(auto& f: as.per_frame)
            f.command_buffers_outdated = true;
    });
}

}
