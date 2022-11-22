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
    uint32_t frame_index
){
    if(total_aabb_count == 0)
        return;

    update_timer.begin(cb, frame_index);
    aabb_buffer.upload(frame_index, cb);

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
            aabb_buffer.get_address(),
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
    context& ctx,
    const char* timer_name,
    size_t sbt_offset,
    size_t max_capacity
):  ctx(&ctx), max_capacity(max_capacity), sbt_offset(sbt_offset)
{
    init_acceleration_structures(timer_name);
}

size_t aabb_scene::get_max_capacity() const
{
    return max_capacity;
}

void aabb_scene::update_acceleration_structures(
    size_t device_index,
    uint32_t frame_index,
    bool& need_scene_reset,
    bool& command_buffers_outdated
){
    auto& as = acceleration_structures[device_index];
    auto& f = as.per_frame[frame_index];

    // Update area point light buffer
    as.aabb_buffer.map<vk::AabbPositionsKHR>(
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
    size_t device_index,
    uint32_t frame_index,
    bool update_only
){
    auto& as = acceleration_structures[device_index];
    auto& f = as.per_frame[frame_index];

    record_blas_update(
        cb,
        as.blas,
        f.aabb_count,
        as.aabb_buffer,
        as.scratch_buffer,
        update_only,
        as.blas_update_timer,
        frame_index
    );
}

void aabb_scene::add_acceleration_structure_instances(
    vk::AccelerationStructureInstanceKHR* instances,
    size_t device_index,
    uint32_t frame_index,
    size_t& instance_index,
    size_t capacity
) const
{
    auto& as = acceleration_structures[device_index];
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
    if(!ctx->is_ray_tracing_supported()) return;

    std::vector<device_data>& devices = ctx->get_devices();
    acceleration_structures.resize(devices.size());

    for(size_t i = 0; i < devices.size(); ++i)
    {
        auto& as = acceleration_structures[i];
        as.aabb_buffer = gpu_buffer(
            devices[i], max_capacity * sizeof(vk::AabbPositionsKHR),
            vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eShaderDeviceAddress|
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
        );

        vk::AccelerationStructureGeometryKHR geom(
            VULKAN_HPP_NAMESPACE::GeometryTypeKHR::eAabbs,
            vk::AccelerationStructureGeometryAabbsDataKHR(
                as.aabb_buffer.get_address(), sizeof(vk::AabbPositionsKHR)
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
            devices[i].dev.getAccelerationStructureBuildSizesKHR(
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
            devices[i],
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
            devices[i],
            devices[i].dev.createAccelerationStructureKHR(create_info)
        );
        blas_info.dstAccelerationStructure = as.blas;

        vk::BufferCreateInfo scratch_info(
            {}, size_info.buildScratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer|
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::SharingMode::eExclusive
        );
        as.scratch_buffer = create_buffer_aligned(
            devices[i],
            scratch_info,
            VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
            devices[i].as_props.minAccelerationStructureScratchOffsetAlignment
        );

        as.blas_update_timer = timer(devices[i], timer_name);
    }
}

void aabb_scene::invalidate_acceleration_structures()
{
    for(auto& as: acceleration_structures)
    {
        as.scene_reset_needed = true;
        for(auto& f: as.per_frame)
            f.command_buffers_outdated = true;
    }
}

}

