#include "mesh.hh"
#include "misc.hh"

namespace tr
{

uint64_t mesh::id_counter = 1;

mesh::mesh(context& ctx): ctx(&ctx), id(0), animation_source(nullptr) {}

mesh::mesh(
    context& ctx,
    std::vector<vertex>&& vertices,
    std::vector<uint32_t>&& indices,
    std::vector<skin_data>&& skin
):  ctx(&ctx), vertices(std::move(vertices)), indices(std::move(indices)),
    skin(std::move(skin)), animation_source(nullptr)
{
    init_buffers();
}

mesh::mesh(mesh* animation_source)
:   ctx(animation_source->ctx), animation_source(animation_source)
{
    init_buffers();
}

uint64_t mesh::get_id() const
{
    return id;
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

bool mesh::is_skinned() const
{
    return skin.size() > 0;
}

mesh* mesh::get_animation_source() const
{
    return animation_source;
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
    id = id_counter++;

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
        vk::CommandBuffer cb = begin_command_buffer(devices[i]);

        buffers[i].vertex_buffer = create_buffer(
            devices[i],
            {
                {}, vertex_bytes,
                vk::BufferUsageFlagBits::eVertexBuffer|buf_flags,
                vk::SharingMode::eExclusive
            },
            VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
            vertices.data(),
            cb
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
                VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                indices.data(),
                cb
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
                    VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                    skin.data(),
                    cb
                );
            }
        }

        end_command_buffer(devices[i], cb);
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
