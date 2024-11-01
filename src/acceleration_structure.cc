#include "acceleration_structure.hh"
#include "mesh.hh"
#include "misc.hh"

namespace tr
{

bottom_level_acceleration_structure::bottom_level_acceleration_structure(
    device_mask dev,
    const std::vector<entry>& entries,
    bool backface_culled,
    bool dynamic,
    bool compact
):  updates_since_rebuild(0), geometry_count(entries.size()),
    backface_culled(backface_culled), dynamic(dynamic), compact(!dynamic && compact),
    buffers(dev)
{
    transform_buffer = gpu_buffer(
        dev,
        sizeof(vk::TransformMatrixKHR) * entries.size(),
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eShaderDeviceAddress |
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
    );

    for(size_t frame_index = 0; frame_index < MAX_FRAMES_IN_FLIGHT; ++frame_index)
        update_transforms(frame_index, entries);

    for(device& d: dev)
    {
        vk::CommandBuffer cb = begin_command_buffer(d);
        rebuild(d.id, 0, cb, entries, false);
        end_command_buffer(d, cb);
    }
}

void bottom_level_acceleration_structure::update_transforms(
    size_t frame_index,
    const std::vector<entry>& entries
){
    transform_buffer.map<uint8_t>(frame_index, [&](uint8_t* data)
    {
        for(size_t j = 0; j < entries.size(); ++j)
        {
            mat4 transform = transpose(entries[j].transform);
            memcpy(
                data + sizeof(vk::TransformMatrixKHR) * j,
                (void*)&transform,
                sizeof(vk::TransformMatrixKHR)
            );
        }
    });
}

void bottom_level_acceleration_structure::rebuild(
    device_id id,
    size_t frame_index,
    vk::CommandBuffer cb,
    const std::vector<entry>& entries,
    bool update
) {
    if(!update) updates_since_rebuild = 0;
    device& dev = buffers.get_device(id);
    buffer_data& bd = buffers[id];

    std::vector<vk::AccelerationStructureGeometryKHR> geometries(entries.size());
    std::vector<vk::TransformMatrixKHR> transforms(entries.size());
    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges(entries.size());
    std::vector<uint32_t> primitive_count(entries.size());

    for(size_t i = 0; i < entries.size(); ++i)
    {
        vk::DeviceOrHostAddressConstKHR transform_address{};
        transform_address.deviceAddress =
            transform_buffer.get_address(id) + sizeof(vk::TransformMatrixKHR) * i;

        vk::AccelerationStructureGeometryKHR geom{};
        const mesh* m = entries[i].m;
        if(m)
        {
            geom.geometryType = vk::GeometryTypeKHR::eTriangles;
            geom.geometry.triangles = vk::AccelerationStructureGeometryTrianglesDataKHR(
                vk::Format::eR32G32B32Sfloat,
                dev.logical.getBufferAddress({m->get_vertex_buffer(id)}),
                sizeof(mesh::vertex),
                m->get_vertices().size()-1,
                vk::IndexType::eUint32,
                dev.logical.getBufferAddress({m->get_index_buffer(id)}),
                transform_address
            );
            uint32_t triangle_count = m->get_indices().size()/3;
            ranges[i] = vk::AccelerationStructureBuildRangeInfoKHR{triangle_count, 0, 0, 0};
            primitive_count[i] = triangle_count;
        }
        else
        {
            geom.geometryType = vk::GeometryTypeKHR::eAabbs;
            geom.geometry.aabbs = vk::AccelerationStructureGeometryAabbsDataKHR(
                entries[i].aabb_buffer->get_address(id),
                sizeof(vk::AabbPositionsKHR)
            );
            ranges[i] = vk::AccelerationStructureBuildRangeInfoKHR{
                (uint32_t)entries[i].aabb_count, 0, 0, 0
            };
            primitive_count[i] = entries[i].aabb_count;
        }

        geom.setFlags(entries[i].opaque ?
                vk::GeometryFlagBitsKHR::eOpaque :
                vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation
        );
        geometries[i] = geom;
    }

    vk::AccelerationStructureBuildGeometryInfoKHR blas_info(
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        dynamic ?
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild|
            vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate :
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace|
            vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction,
        update ?
            vk::BuildAccelerationStructureModeKHR::eUpdate :
            vk::BuildAccelerationStructureModeKHR::eBuild,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        geometries.size(),
        geometries.data()
    );

    if(!*bd.blas)
    {
        // Need to calculate BLAS size.
        vk::AccelerationStructureBuildSizesInfoKHR size_info = dev.logical.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, blas_info, primitive_count
        );
        vk::BufferCreateInfo scratch_info(
            {}, size_info.buildScratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer|
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::SharingMode::eExclusive
        );
        bd.scratch_buffer = create_buffer_aligned(
            dev, scratch_info, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
            dev.as_props.minAccelerationStructureScratchOffsetAlignment
        );
        bd.scratch_address = bd.scratch_buffer.get_address();

        vk::BufferCreateInfo blas_buffer_info(
            {}, size_info.accelerationStructureSize,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR|
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::SharingMode::eExclusive
        );
        bd.blas_buffer = create_buffer(dev, blas_buffer_info, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

        vk::AccelerationStructureCreateInfoKHR create_info(
            {},
            bd.blas_buffer,
            {},
            size_info.accelerationStructureSize,
            vk::AccelerationStructureTypeKHR::eBottomLevel,
            {}
        );
        bd.blas = vkm(dev, dev.logical.createAccelerationStructureKHR(create_info));
    }
    blas_info.srcAccelerationStructure = update ? *bd.blas : VK_NULL_HANDLE;
    blas_info.dstAccelerationStructure = bd.blas;
    blas_info.scratchData.deviceAddress = bd.scratch_address;

    const vk::AccelerationStructureBuildRangeInfoKHR* range_ptr = ranges.data();

    vkm<vk::QueryPool> query_pool;
    vk::CommandBuffer initial_cb;
    if(compact)
    {
        initial_cb = begin_command_buffer(dev);
        transform_buffer.upload(id, frame_index, initial_cb);
        query_pool = vkm(dev, dev.logical.createQueryPool({
            {},
            vk::QueryType::eAccelerationStructureCompactedSizeKHR,
            1,
            {}
        }));
        initial_cb.resetQueryPool(query_pool, 0, 1);
        initial_cb.buildAccelerationStructuresKHR({blas_info}, range_ptr);

        vk::MemoryBarrier barrier(
            vk::AccessFlagBits::eAccelerationStructureWriteKHR,
            vk::AccessFlagBits::eAccelerationStructureReadKHR
        );
        initial_cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
            {}, barrier, {}, {}
        );

        initial_cb.writeAccelerationStructuresPropertiesKHR(
            *bd.blas,
            vk::QueryType::eAccelerationStructureCompactedSizeKHR,
            query_pool,
            0
        );
        end_command_buffer(dev, initial_cb);

        // NVIDIA bug as of 460.27.04: Only the lower 32 bits of the parameter
        // get written to, despite the spec saying that it's supposed to be a
        // VkDeviceSize (uint64_t). We need to make sure that the size is
        // zero-initialized to avoid the higher 32 bits breaking everything.
        vk::DeviceSize compact_size = 0;
        (void)dev.logical.getQueryPoolResults(
            query_pool, 0, 1,
            sizeof(vk::DeviceSize),
            &compact_size,
            sizeof(vk::DeviceSize),
            vk::QueryResultFlagBits::eWait
        );

        vkm<vk::AccelerationStructureKHR> fat_blas = std::move(bd.blas);
        vkm<vk::Buffer> fat_blas_buffer(std::move(bd.blas_buffer));

        vk::BufferCreateInfo blas_buffer_info(
            {}, compact_size,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR|
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::SharingMode::eExclusive
        );

        bd.blas_buffer = create_buffer(dev, blas_buffer_info, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
        vk::AccelerationStructureCreateInfoKHR create_info(
            {},
            bd.blas_buffer,
            {},
            compact_size,
            vk::AccelerationStructureTypeKHR::eBottomLevel,
            {}
        );

        bd.blas = vkm(dev, dev.logical.createAccelerationStructureKHR(create_info));
        blas_info.dstAccelerationStructure = bd.blas;

        cb.copyAccelerationStructureKHR({
            fat_blas,
            bd.blas,
            vk::CopyAccelerationStructureModeKHR::eCompact
        });

        fat_blas.drop();
        fat_blas_buffer.drop();
    }
    else
    {
        transform_buffer.upload(id, frame_index, cb);
        cb.buildAccelerationStructuresKHR({blas_info}, range_ptr);
    }
    bd.blas_address = dev.logical.getAccelerationStructureAddressKHR({bd.blas});
}

size_t bottom_level_acceleration_structure::get_updates_since_rebuild() const
{
    return updates_since_rebuild;
}

vk::AccelerationStructureKHR bottom_level_acceleration_structure::get_blas_handle(device_id id) const
{
    return *buffers[id].blas;
}

vk::DeviceAddress bottom_level_acceleration_structure::get_blas_address(device_id id) const
{
    return buffers[id].blas_address;
}

size_t bottom_level_acceleration_structure::get_geometry_count() const
{
    return geometry_count;
}

bool bottom_level_acceleration_structure::is_backface_culled() const
{
    return backface_culled;
}

top_level_acceleration_structure::top_level_acceleration_structure(
    device_mask dev,
    size_t capacity
):  updates_since_rebuild(0), instance_count(0),
    instance_capacity(capacity), require_rebuild(true), buffers(dev)
{
    instance_buffer = gpu_buffer(
        dev,
        capacity * sizeof(VkAccelerationStructureInstanceKHR),
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eShaderDeviceAddress|
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
    );

    for(auto[dev, bd]: buffers)
    {
        vk::AccelerationStructureGeometryKHR geom(
            VULKAN_HPP_NAMESPACE::GeometryTypeKHR::eInstances,
            vk::AccelerationStructureGeometryInstancesDataKHR{
                VK_FALSE, instance_buffer.get_address(dev.id)
            }
        );

        vk::AccelerationStructureBuildGeometryInfoKHR tlas_info(
            vk::AccelerationStructureTypeKHR::eTopLevel,
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace|
            vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
            vk::BuildAccelerationStructureModeKHR::eBuild,
            VK_NULL_HANDLE,
            VK_NULL_HANDLE,
            1,
            &geom
        );

        vk::AccelerationStructureBuildSizesInfoKHR size_info =
            dev.logical.getAccelerationStructureBuildSizesKHR(
                vk::AccelerationStructureBuildTypeKHR::eDevice, tlas_info, {(uint32_t)capacity}
            );

        vk::BufferCreateInfo tlas_buffer_info(
            {}, size_info.accelerationStructureSize,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR|
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::SharingMode::eExclusive
        );
        bd.tlas_buffer = create_buffer(dev, tlas_buffer_info, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

        vk::AccelerationStructureCreateInfoKHR create_info(
            {},
            bd.tlas_buffer,
            {},
            size_info.accelerationStructureSize,
            vk::AccelerationStructureTypeKHR::eTopLevel,
            {}
        );
        bd.tlas = vkm(dev, dev.logical.createAccelerationStructureKHR(create_info));
        tlas_info.dstAccelerationStructure = bd.tlas;

        vk::BufferCreateInfo scratch_info(
            {}, size_info.buildScratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer|
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::SharingMode::eExclusive
        );

        bd.scratch_buffer = create_buffer_aligned(
            dev, scratch_info, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
            dev.as_props.minAccelerationStructureScratchOffsetAlignment
        );
        tlas_info.scratchData = bd.scratch_buffer.get_address();
        bd.tlas_address = dev.logical.getAccelerationStructureAddressKHR({bd.tlas});
    }
}

gpu_buffer& top_level_acceleration_structure::get_instances_buffer()
{
    return instance_buffer;
}

void top_level_acceleration_structure::rebuild(
    device_id id,
    vk::CommandBuffer cb,
    size_t instance_count,
    bool update
){
    buffer_data& bd = buffers[id];

    // Barrier to make sure all BLAS's have updated already.
    vk::MemoryBarrier blas_barrier(
        vk::AccessFlagBits::eAccelerationStructureWriteKHR,
        vk::AccessFlagBits::eAccelerationStructureWriteKHR
    );

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
        {}, blas_barrier, {}, {}
    );

    vk::AccelerationStructureGeometryKHR tlas_geometry(
        vk::GeometryTypeKHR::eInstances,
        vk::AccelerationStructureGeometryInstancesDataKHR{
            false, instance_buffer.get_address(id)
        },
        {}
    );

    vk::AccelerationStructureBuildGeometryInfoKHR tlas_info(
        vk::AccelerationStructureTypeKHR::eTopLevel,
        vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace|
        vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
        update ? vk::BuildAccelerationStructureModeKHR::eUpdate : vk::BuildAccelerationStructureModeKHR::eBuild,
        update ? bd.tlas : vk::AccelerationStructureKHR{},
        bd.tlas,
        1,
        &tlas_geometry,
        nullptr,
        bd.scratch_buffer.get_address()
    );

    vk::AccelerationStructureBuildRangeInfoKHR build_offset_info(
        instance_count, 0, 0, 0
    );

    cb.buildAccelerationStructuresKHR({tlas_info}, {&build_offset_info});
}

size_t top_level_acceleration_structure::get_updates_since_rebuild() const
{
    return updates_since_rebuild;
}

void top_level_acceleration_structure::copy(
    device_id id,
    top_level_acceleration_structure& other,
    vk::CommandBuffer cmd
){
    if(other.instance_capacity != instance_capacity)
        throw std::runtime_error("Attempting to copy between top level acceleration structures of different capacities!");
    instance_count = other.instance_count;

    vk::CopyAccelerationStructureInfoKHR copy_info = {
        *other.buffers[id].tlas,
        *buffers[id].tlas,
        vk::CopyAccelerationStructureModeKHR::eClone
    };
    cmd.copyAccelerationStructureKHR(copy_info);
}

const vk::AccelerationStructureKHR* top_level_acceleration_structure::get_tlas_handle(device_id id) const
{
    return buffers[id].tlas;
}

vk::DeviceAddress top_level_acceleration_structure::get_tlas_address(device_id id) const
{
    return buffers[id].tlas_address;
}

}
