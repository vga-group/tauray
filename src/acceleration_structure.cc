#include "acceleration_structure.hh"
#include "mesh.hh"
#include "misc.hh"

namespace tr
{

bottom_level_acceleration_structure::bottom_level_acceleration_structure(
    context& ctx,
    const std::vector<entry>& entries,
    bool backface_culled,
    bool dynamic
):  ctx(&ctx), updates_since_rebuild(0), geometry_count(entries.size()),
    backface_culled(backface_culled), dynamic(dynamic)
{
    std::vector<device_data>& devices = ctx.get_devices();
    buffers.resize(devices.size());

    for(size_t i = 0; i < devices.size(); ++i)
    {
        buffers[i].transform_buffer = gpu_buffer(
            devices[i], sizeof(vk::TransformMatrixKHR) * entries.size(),
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
        );
    }

    update_transforms(0, entries);

    for(size_t i = 0; i < devices.size(); ++i)
    {
        vk::CommandBuffer cb = begin_command_buffer(devices[i]);
        rebuild(i, 0, cb, entries, false);
        end_command_buffer(devices[i], cb);
    }
}

void bottom_level_acceleration_structure::update_transforms(
    size_t frame_index,
    const std::vector<entry>& entries
){
    for(size_t i = 0; i < ctx->get_devices().size(); ++i)
    {
        buffers[i].transform_buffer.map<uint8_t>(frame_index, [&](uint8_t* data)
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
}

void bottom_level_acceleration_structure::rebuild(
    size_t device_index,
    size_t frame_index,
    vk::CommandBuffer cb,
    const std::vector<entry>& entries,
    bool update
) {
    if(!update) updates_since_rebuild = 0;
    device_data& dev = ctx->get_devices()[device_index];

    std::vector<vk::AccelerationStructureGeometryKHR> geometries(entries.size());
    std::vector<vk::TransformMatrixKHR> transforms(entries.size());
    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges(entries.size());
    std::vector<uint32_t> primitive_count(entries.size());

    buffer_data& bd = buffers[device_index];
    for(size_t i = 0; i < entries.size(); ++i)
    {
        const mesh* m = entries[i].m;
        vk::DeviceOrHostAddressConstKHR transform_address{};
        transform_address.deviceAddress =
            bd.transform_buffer.get_address() + sizeof(vk::TransformMatrixKHR) * i;

        geometries[i] = {
            vk::GeometryTypeKHR::eTriangles,
            vk::AccelerationStructureGeometryTrianglesDataKHR(
                vk::Format::eR32G32B32Sfloat,
                dev.dev.getBufferAddress({m->get_vertex_buffer(device_index)}),
                sizeof(mesh::vertex),
                m->get_vertices().size()-1,
                vk::IndexType::eUint32,
                dev.dev.getBufferAddress({m->get_index_buffer(device_index)}),
                transform_address
            ),
            (entries[i].opaque ?
                vk::GeometryFlagBitsKHR::eOpaque :
                vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation)
        };
        uint32_t triangle_count = m->get_indices().size()/3;
        ranges[i] = vk::AccelerationStructureBuildRangeInfoKHR{triangle_count, 0, 0, 0};
        primitive_count[i] = triangle_count;
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

    if(!bd.blas || !dynamic)
    {
        // Need to calculate BLAS size.
        vk::AccelerationStructureBuildSizesInfoKHR size_info = dev.dev.getAccelerationStructureBuildSizesKHR(
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
        bd.blas = vkm(dev, dev.dev.createAccelerationStructureKHR(create_info));
    }
    blas_info.srcAccelerationStructure = update ? *bd.blas : VK_NULL_HANDLE;
    blas_info.dstAccelerationStructure = bd.blas;
    blas_info.scratchData.deviceAddress = bd.scratch_address;

    const vk::AccelerationStructureBuildRangeInfoKHR* range_ptr = ranges.data();

    vkm<vk::QueryPool> query_pool;
    vk::CommandBuffer initial_cb;
    if(!dynamic)
    {
        initial_cb = begin_command_buffer(dev);
        bd.transform_buffer.upload(frame_index, initial_cb);
        query_pool = vkm(dev, dev.dev.createQueryPool({
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
        (void)dev.dev.getQueryPoolResults(
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

        bd.blas = vkm(dev, dev.dev.createAccelerationStructureKHR(create_info));
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
        bd.transform_buffer.upload(frame_index, cb);
        cb.buildAccelerationStructuresKHR({blas_info}, range_ptr);
    }
    bd.blas_address = dev.dev.getAccelerationStructureAddressKHR({bd.blas});
}

size_t bottom_level_acceleration_structure::get_updates_since_rebuild() const
{
    return updates_since_rebuild;
}

vk::AccelerationStructureKHR bottom_level_acceleration_structure::get_blas_handle(size_t device_index) const
{
    return *buffers[device_index].blas;
}

vk::DeviceAddress bottom_level_acceleration_structure::get_blas_address(size_t device_index) const
{
    return buffers[device_index].blas_address;
}

size_t bottom_level_acceleration_structure::get_geometry_count() const
{
    return geometry_count;
}

top_level_acceleration_structure::top_level_acceleration_structure(
    context& ctx,
    size_t capacity
):  ctx(&ctx), updates_since_rebuild(0), instance_count(0),
    instance_capacity(capacity), require_rebuild(true)
{
    std::vector<device_data>& devices = ctx.get_devices();
    buffers.resize(devices.size());
    for(size_t i = 0; i < devices.size(); ++i)
    {
        buffer_data& bd = buffers[i];
        bd.instance_buffer = gpu_buffer(
            devices[i],
            capacity * sizeof(VkAccelerationStructureInstanceKHR),
            vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eShaderDeviceAddress|
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
        );
        vk::AccelerationStructureGeometryKHR geom(
            VULKAN_HPP_NAMESPACE::GeometryTypeKHR::eInstances,
            vk::AccelerationStructureGeometryInstancesDataKHR{
                VK_FALSE, bd.instance_buffer.get_address()
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
            devices[i].dev.getAccelerationStructureBuildSizesKHR(
                vk::AccelerationStructureBuildTypeKHR::eDevice, tlas_info, {(uint32_t)capacity}
            );

        vk::BufferCreateInfo tlas_buffer_info(
            {}, size_info.accelerationStructureSize,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR|
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::SharingMode::eExclusive
        );
        bd.tlas_buffer = create_buffer(devices[i], tlas_buffer_info, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

        vk::AccelerationStructureCreateInfoKHR create_info(
            {},
            bd.tlas_buffer,
            {},
            size_info.accelerationStructureSize,
            vk::AccelerationStructureTypeKHR::eTopLevel,
            {}
        );
        bd.tlas = vkm(devices[i], devices[i].dev.createAccelerationStructureKHR(create_info));
        tlas_info.dstAccelerationStructure = bd.tlas;

        vk::BufferCreateInfo scratch_info(
            {}, size_info.buildScratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer|
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::SharingMode::eExclusive
        );

        bd.scratch_buffer = create_buffer_aligned(
            devices[i], scratch_info, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
            devices[i].as_props.minAccelerationStructureScratchOffsetAlignment
        );
        tlas_info.scratchData = bd.scratch_buffer.get_address();
        bd.tlas_address = devices[i].dev.getAccelerationStructureAddressKHR({bd.tlas});
    }
}

gpu_buffer& top_level_acceleration_structure::get_instances_buffer(size_t device_index)
{
    return buffers[device_index].instance_buffer;
}

void top_level_acceleration_structure::rebuild(
    size_t device_index,
    vk::CommandBuffer cb,
    size_t instance_count,
    bool update
){
    buffer_data& bd = buffers[device_index];

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
            false, bd.instance_buffer.get_address()
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

const vk::AccelerationStructureKHR* top_level_acceleration_structure::get_tlas_handle(size_t device_index) const
{
    return buffers[device_index].tlas;
}

vk::DeviceAddress top_level_acceleration_structure::get_tlas_address(size_t device_index) const
{
    return buffers[device_index].tlas_address;
}

}
