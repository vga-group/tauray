#include "aabb_scene.hh"
#include "misc.hh"

namespace
{
using namespace tr;

void record_blas_update(
    vk::CommandBuffer& cb,
    vk::AccelerationStructureKHR blas,
    size_t total_aabb_count,
    gpu_buffer& aabb_buffer,
    vkm<vk::Buffer>& scratch_buffer,
    bool update,
    timer& update_timer,
    uint32_t frame_index,
    device_id id
){
    if(total_aabb_count == 0)
        return;

    update_timer.begin(cb, frame_index);
    aabb_buffer.upload(id, frame_index, cb);

    vk::MemoryBarrier barrier(
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eAccelerationStructureWriteKHR
    );

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
        {},
        barrier, {}, {}
    );

    vk::AccelerationStructureGeometryKHR geom(
        vk::GeometryTypeKHR::eAabbs,
        vk::AccelerationStructureGeometryAabbsDataKHR(
            aabb_buffer.get_address(id),
            sizeof(vk::AabbPositionsKHR)
        ),
        vk::GeometryFlagBitsKHR::eOpaque
    );

    vk::AccelerationStructureBuildGeometryInfoKHR blas_build_info(
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace|
        vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
        update ? vk::BuildAccelerationStructureModeKHR::eUpdate : vk::BuildAccelerationStructureModeKHR::eBuild,
        update ? blas : vk::AccelerationStructureKHR{},
        blas,
        1,
        &geom,
        nullptr,
        scratch_buffer.get_address()
    );

    vk::AccelerationStructureBuildRangeInfoKHR offset(
        total_aabb_count, 0, 0, 0
    );

    cb.buildAccelerationStructuresKHR({blas_build_info}, {&offset});
    update_timer.end(cb, frame_index);
}

}

namespace tr
{

aabb_scene::aabb_scene(
    device_mask dev,
    const char* timer_name,
    size_t sbt_offset,
    size_t max_capacity
):  max_capacity(max_capacity), sbt_offset(sbt_offset), acceleration_structures(dev)
{
    init_acceleration_structures(timer_name);
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
    auto& as = acceleration_structures[id];
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
    auto& as = acceleration_structures[id];
    auto& f = as.per_frame[frame_index];

    record_blas_update(
        cb,
        as.blas,
        f.aabb_count,
        aabb_buffer,
        as.scratch_buffer,
        update_only,
        as.blas_update_timer,
        frame_index,
        id
    );
}

void aabb_scene::add_acceleration_structure_instances(
    vk::AccelerationStructureInstanceKHR* instances,
    device_id id,
    uint32_t frame_index,
    size_t& instance_index,
    size_t capacity
) const
{
    auto& as = acceleration_structures[id];
    auto& f = as.per_frame[frame_index];

    if(f.aabb_count != 0 && instance_index < capacity)
    {
        vk::AccelerationStructureInstanceKHR& inst = instances[instance_index++];
        inst = vk::AccelerationStructureInstanceKHR(
            {}, instance_index, 1<<1, sbt_offset,
            vk::GeometryInstanceFlagBitsKHR::eTriangleCullDisable,
            as.blas_buffer.get_address()
        );
        mat4 id_mat = mat4(1.0f);
        memcpy((void*)&inst.transform, (void*)&id_mat, sizeof(inst.transform));
    }
}

void aabb_scene::init_acceleration_structures(const char* timer_name)
{
    if(!acceleration_structures.get_context()->is_ray_tracing_supported()) return;

    aabb_buffer = gpu_buffer(
        acceleration_structures.get_mask(), max_capacity * sizeof(vk::AabbPositionsKHR),
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eShaderDeviceAddress|
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
    );

    acceleration_structures([&](device& dev, acceleration_structure_data& as){
        vk::AccelerationStructureGeometryKHR geom(
            VULKAN_HPP_NAMESPACE::GeometryTypeKHR::eAabbs,
            vk::AccelerationStructureGeometryAabbsDataKHR(
                aabb_buffer.get_address(dev.index), sizeof(vk::AabbPositionsKHR)
            ),
            vk::GeometryFlagBitsKHR::eOpaque
        );

        vk::AccelerationStructureBuildGeometryInfoKHR blas_info(
            vk::AccelerationStructureTypeKHR::eBottomLevel,
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace|
            vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
            vk::BuildAccelerationStructureModeKHR::eBuild,
            VK_NULL_HANDLE,
            VK_NULL_HANDLE,
            1,
            &geom
        );

        vk::AccelerationStructureBuildSizesInfoKHR size_info =
            dev.dev.getAccelerationStructureBuildSizesKHR(
                vk::AccelerationStructureBuildTypeKHR::eDevice, blas_info,
                {(uint32_t)max_capacity}
            );

        vk::BufferCreateInfo blas_buffer_info(
            {}, size_info.accelerationStructureSize,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR|
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::SharingMode::eExclusive
        );
        as.blas_buffer = create_buffer(
            dev,
            blas_buffer_info,
            VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
        );

        vk::AccelerationStructureCreateInfoKHR create_info(
            {},
            as.blas_buffer,
            {},
            size_info.accelerationStructureSize,
            vk::AccelerationStructureTypeKHR::eBottomLevel,
            {}
        );
        as.blas = vkm(
            dev,
            dev.dev.createAccelerationStructureKHR(create_info)
        );
        blas_info.dstAccelerationStructure = as.blas;

        vk::BufferCreateInfo scratch_info(
            {}, size_info.buildScratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer|
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::SharingMode::eExclusive
        );
        as.scratch_buffer = create_buffer_aligned(
            dev,
            scratch_info,
            VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
            dev.as_props.minAccelerationStructureScratchOffsetAlignment
        );

        as.blas_update_timer = timer(dev, timer_name);
    });
}

void aabb_scene::invalidate_acceleration_structures()
{
    acceleration_structures([&](device&, acceleration_structure_data& as){
        as.scene_reset_needed = true;
        for(auto& f: as.per_frame)
            f.command_buffers_outdated = true;
    });
}

}

