#include "mesh.hh"
#include "misc.hh"

namespace
{
using namespace tr;

void build_acceleration_structure(
    device_data& dev,
    vkm<vk::AccelerationStructureKHR>& blas,
    vkm<vk::Buffer>& blas_buffer,
    vk::DeviceAddress& blas_address,
    vkm<vk::Buffer>& blas_scratch_buffer,
    size_t vertex_count,
    vk::Buffer vertex_buffer,
    size_t index_count,
    vk::Buffer index_buffer,
    bool static_as,
    bool opaque
){
    vk::AccelerationStructureGeometryKHR geom(
        VULKAN_HPP_NAMESPACE::GeometryTypeKHR::eTriangles,
        vk::AccelerationStructureGeometryTrianglesDataKHR(
            vk::Format::eR32G32B32Sfloat,
            dev.dev.getBufferAddress({vertex_buffer}),
            sizeof(mesh::vertex),
            vertex_count-1,
            vk::IndexType::eUint32,
            dev.dev.getBufferAddress({index_buffer}),
            {}
        ),
        opaque ?
            vk::GeometryFlagBitsKHR::eOpaque :
            vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation
    );

    vk::AccelerationStructureBuildRangeInfoKHR offset((uint32_t)index_count/3);

    vk::AccelerationStructureBuildGeometryInfoKHR blas_info(
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        {},
        vk::BuildAccelerationStructureModeKHR::eBuild,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        1,
        &geom
    );

    if(static_as)
    {
        blas_info.flags =
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace|
            vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction;
    }
    else
    {
        blas_info.flags =
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild|
            vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
    }

    vk::AccelerationStructureBuildSizesInfoKHR size_info =
        dev.dev.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, blas_info, {(uint32_t)index_count/3}
        );

    vk::BufferCreateInfo blas_buffer_info(
        {}, size_info.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR|
        vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::SharingMode::eExclusive
    );
    blas_buffer = create_buffer(dev, blas_buffer_info, VMA_MEMORY_USAGE_GPU_ONLY);

    vk::AccelerationStructureCreateInfoKHR create_info(
        {},
        blas_buffer,
        {},
        size_info.accelerationStructureSize,
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        {}
    );
    blas = vkm(dev, dev.dev.createAccelerationStructureKHR(create_info));
    blas_info.dstAccelerationStructure = blas;

    vk::BufferCreateInfo scratch_info(
        {}, size_info.buildScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer|
        vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::SharingMode::eExclusive
    );
    blas_scratch_buffer = create_buffer(
        dev, scratch_info, VMA_MEMORY_USAGE_GPU_ONLY
    );
    blas_info.scratchData = blas_scratch_buffer.get_address();

    vkm<vk::QueryPool> query_pool = vkm(dev, dev.dev.createQueryPool({
        {},
        vk::QueryType::eAccelerationStructureCompactedSizeKHR,
        1,
        {}
    }));

    vk::CommandBuffer cb = begin_command_buffer(dev);
    cb.resetQueryPool(query_pool, 0, 1);
    cb.buildAccelerationStructuresKHR({blas_info}, {&offset});

    if(static_as)
    {
        vk::MemoryBarrier barrier(
            vk::AccessFlagBits::eAccelerationStructureWriteKHR,
            vk::AccessFlagBits::eAccelerationStructureReadKHR
        );
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
            {}, barrier, {}, {}
        );

        cb.writeAccelerationStructuresPropertiesKHR(
            *blas,
            vk::QueryType::eAccelerationStructureCompactedSizeKHR,
            query_pool,
            0
        );
    }

    end_command_buffer(dev, cb);

    // Compact the acceleration structure.
    if(static_as)
    {
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

        vkm<vk::AccelerationStructureKHR> fat_blas = std::move(blas);
        vkm<vk::Buffer> fat_blas_buffer(std::move(blas_buffer));

        blas_buffer_info.size = compact_size;
        blas_buffer = create_buffer(dev, blas_buffer_info, VMA_MEMORY_USAGE_GPU_ONLY);
        create_info.buffer = blas_buffer;

        create_info.size = compact_size;
        blas = vkm(dev, dev.dev.createAccelerationStructureKHR(create_info));
        blas_info.dstAccelerationStructure = blas;

        vk::CommandBuffer cb = begin_command_buffer(dev);
        cb.copyAccelerationStructureKHR({
            fat_blas,
            blas,
            vk::CopyAccelerationStructureModeKHR::eCompact
        });
        end_command_buffer(dev, cb);

        fat_blas.destroy();
        fat_blas_buffer.destroy();
        blas_scratch_buffer.destroy();
    }

    blas_address = dev.dev.getAccelerationStructureAddressKHR({blas});
}

}

namespace tr
{

mesh::mesh(context& ctx): ctx(&ctx), opaque(false), animation_source(nullptr) {}

mesh::mesh(
    context& ctx,
    std::vector<vertex>&& vertices,
    std::vector<uint32_t>&& indices,
    std::vector<skin_data>&& skin
):  ctx(&ctx), vertices(std::move(vertices)), indices(std::move(indices)),
    skin(std::move(skin)), opaque(false), animation_source(nullptr)
{
    init_buffers();
}

mesh::mesh(mesh* animation_source)
:   ctx(animation_source->ctx), opaque(animation_source->opaque),
    animation_source(animation_source)
{
    init_buffers();
}

std::vector<mesh::vertex>& mesh::get_vertices()
{
    if(animation_source) return animation_source->vertices;
    return vertices;
}

const std::vector<mesh::vertex>& mesh::get_vertices() const
{
    if(animation_source) return animation_source->vertices;
    return vertices;
}

std::vector<uint32_t>& mesh::get_indices()
{
    if(animation_source) return animation_source->indices;
    return indices;
}

const std::vector<uint32_t>& mesh::get_indices() const
{
    if(animation_source) return animation_source->indices;
    return indices;
}

std::vector<mesh::skin_data>& mesh::get_skin()
{
    if(animation_source) return animation_source->skin;
    return skin;
}

const std::vector<mesh::skin_data>& mesh::get_skin() const
{
    if(animation_source) return animation_source->skin;
    return skin;
}

vk::Buffer mesh::get_vertex_buffer(size_t device_index) const
{
    return buffers[device_index].vertex_buffer;
}

vk::Buffer mesh::get_index_buffer(size_t device_index) const
{
    if(animation_source)
        return animation_source->buffers[device_index].index_buffer;
    return buffers[device_index].index_buffer;
}

vk::Buffer mesh::get_skin_buffer(size_t device_index) const
{
    if(animation_source)
        return animation_source->buffers[device_index].skin_buffer;
    return buffers[device_index].skin_buffer;
}

vk::AccelerationStructureKHR mesh::get_blas(size_t device_index) const
{
    return buffers[device_index].blas;
}

vk::DeviceAddress mesh::get_blas_address(size_t device_index) const
{
    return buffers[device_index].blas_address;
}

void mesh::update_blas(
    vk::CommandBuffer cb,
    size_t device_index,
    blas_update_strategy strat
) const
{
    std::vector<device_data>& devices = ctx->get_devices();

    vk::AccelerationStructureBuildRangeInfoKHR offset(
        (uint32_t)get_indices().size()/3
    );

    vk::AccelerationStructureGeometryKHR geom(
        VULKAN_HPP_NAMESPACE::GeometryTypeKHR::eTriangles,
        vk::AccelerationStructureGeometryTrianglesDataKHR(
            vk::Format::eR32G32B32Sfloat,
            devices[device_index].dev.getBufferAddress({get_vertex_buffer(device_index)}),
            sizeof(mesh::vertex),
            get_vertices().size()-1,
            vk::IndexType::eUint32,
            devices[device_index].dev.getBufferAddress({get_index_buffer(device_index)}),
            {}
        ),
        opaque ?
            vk::GeometryFlagBitsKHR::eOpaque :
            vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation
    );

    vk::AccelerationStructureBuildGeometryInfoKHR blas_info(
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild|
        vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
        strat == UPDATE_REBUILD ?
            vk::BuildAccelerationStructureModeKHR::eBuild : 
            vk::BuildAccelerationStructureModeKHR::eUpdate,
        {},
        buffers[device_index].blas,
        1,
        &geom,
        {},
        buffers[device_index].blas_scratch_buffer.get_address()
    );

    switch(strat)
    {
    case UPDATE_FROM_ANIMATION_SOURCE:
        blas_info.srcAccelerationStructure = animation_source->buffers[device_index].blas;
        break;
    case UPDATE_FROM_PREVIOUS:
        blas_info.srcAccelerationStructure = buffers[device_index].blas;
        break;
    default:
        break;
    }

    cb.buildAccelerationStructuresKHR({blas_info}, {&offset});
}

bool mesh::is_skinned() const
{
    return skin.size() > 0;
}

mesh* mesh::get_animation_source() const
{
    return animation_source;
}

void mesh::set_opaque(bool opaque)
{
    this->opaque = opaque;
}

void mesh::refresh_buffers()
{
    // TODO: Make this smarter, no need to reinit if buffer size is the same
    // as before.
    buffers.clear();
    init_buffers();
}

void mesh::calculate_normals()
{
    // Clear existing data
    for(vertex& v: vertices)
    {
        v.normal = vec3(0);
    }

    // Go through triangles
    for(size_t i = 0; i < indices.size()/3; ++i)
    {
        vertex& v0 = vertices[indices[i*3]];
        vertex& v1 = vertices[indices[i*3+1]];
        vertex& v2 = vertices[indices[i*3+2]];

        vec3 d0 = v1.pos - v0.pos;
        vec3 d1 = v2.pos - v0.pos;
        pvec3 hard_normal = cross(d0, d1);
        float len = length(hard_normal);
        if(len > 1e-6) hard_normal /= len;
        v0.normal += hard_normal;
        v1.normal += hard_normal;
        v2.normal += hard_normal;
    }
 
    // Normalize results
    for(vertex& v: vertices)
    {
        float len = length(v.normal);
        if(len > 1e-6) v.normal /= len;
    }
}

void mesh::calculate_tangents()
{
    // Clear existing data
    for(vertex& v: vertices)
    {
        v.tangent = vec4(0);
    }

    // Go through triangles
    for(size_t i = 0; i < indices.size()/3; ++i)
    {
        vertex& v0 = vertices[indices[i*3]];
        vertex& v1 = vertices[indices[i*3+1]];
        vertex& v2 = vertices[indices[i*3+2]];

        vec3 d0 = v1.pos - v0.pos;
        vec3 d1 = v2.pos - v0.pos;
        vec3 hard_normal = cross(d0, d1);
        float len = length(hard_normal);
        if(len > 1e-6) hard_normal /= len;

        vec2 uv0 = v1.uv - v0.uv;
        vec2 uv1 = v2.uv - v0.uv;
        vec3 hard_tangent = normalize(uv1.y * d0 - uv0.y * d1);
        vec3 hard_bitangent = normalize(uv1.x * d1 - uv0.x * d0);
        v0.tangent += pvec4(
            hard_tangent,
            dot(cross(hard_normal, hard_tangent), hard_bitangent) < 0 ? -1 : 1
        );
    }
 
    // Normalize results
    for(vertex& v: vertices)
    {
        v.tangent = pvec4(
            normalize(pvec3(v.tangent) - v.normal * dot(v.normal, pvec3(v.tangent))),
            v.tangent.w < 0 ? -1 : 1
        );
    }
}

void mesh::init_buffers()
{
    std::vector<device_data>& devices = ctx->get_devices();
    buffers.resize(devices.size());

    const std::vector<vertex>& vertices = animation_source ? animation_source->vertices : this->vertices;
    const std::vector<uint32_t>& indices = animation_source ? animation_source->indices : this->indices;
    const std::vector<skin_data>& skin = animation_source ? animation_source->skin : this->skin;

    size_t vertex_bytes = vertices.size() * sizeof(vertices[0]);
    size_t index_bytes = indices.size() * sizeof(indices[0]);
    size_t skin_bytes = skin.size() * sizeof(skin[0]);

    for(size_t i = 0; i < devices.size(); ++i)
    {
        vk::BufferUsageFlags buf_flags =
            vk::BufferUsageFlagBits::eStorageBuffer;
        if(ctx->is_ray_tracing_supported())
            buf_flags = buf_flags | vk::BufferUsageFlagBits::eShaderDeviceAddress|
                vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;

        buffers[i].vertex_buffer = create_buffer(
            devices[i],
            {
                {}, vertex_bytes,
                vk::BufferUsageFlagBits::eVertexBuffer|buf_flags,
                vk::SharingMode::eExclusive
            },
            VMA_MEMORY_USAGE_GPU_ONLY,
            vertices.data()
        );

        if(!animation_source)
        {
            buffers[i].index_buffer = create_buffer(
                devices[i],
                {
                    {}, index_bytes,
                    vk::BufferUsageFlagBits::eIndexBuffer|buf_flags,
                    vk::SharingMode::eExclusive
                },
                VMA_MEMORY_USAGE_GPU_ONLY,
                indices.data()
            );
            if(skin_bytes > 0)
            {
                buffers[i].skin_buffer = create_buffer(
                    devices[i],
                    {
                        {}, skin_bytes,
                        vk::BufferUsageFlagBits::eStorageBuffer,
                        vk::SharingMode::eExclusive
                    },
                    VMA_MEMORY_USAGE_GPU_ONLY,
                    skin.data()
                );
            }
        }

        if(ctx->is_ray_tracing_supported())
        {
            build_acceleration_structure(
                devices[i],
                buffers[i].blas,
                buffers[i].blas_buffer,
                buffers[i].blas_address,
                buffers[i].blas_scratch_buffer,
                vertices.size(),
                buffers[i].vertex_buffer,
                indices.size(),
                get_index_buffer(i),
                !animation_source && skin_bytes == 0,
                opaque
            );
        }
    }
}

std::vector<vk::VertexInputBindingDescription> mesh::get_bindings()
{
    return {vk::VertexInputBindingDescription{
        0, sizeof(vertex), vk::VertexInputRate::eVertex
    }};
}

std::vector<vk::VertexInputAttributeDescription> mesh::get_attributes()
{
    return {
        vk::VertexInputAttributeDescription{
            0, 0, vk::Format::eR32G32B32Sfloat, offsetof(vertex, pos)
        },
        vk::VertexInputAttributeDescription{
            1, 0, vk::Format::eR32G32B32Sfloat, offsetof(vertex, normal)
        },
        vk::VertexInputAttributeDescription{
            2, 0, vk::Format::eR32G32Sfloat, offsetof(vertex, uv)
        },
        vk::VertexInputAttributeDescription{
            3, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(vertex, tangent)
        }
    };
}

}
