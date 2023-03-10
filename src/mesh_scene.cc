#include "mesh_scene.hh"
#include "misc.hh"
#include <unordered_set>

namespace tr
{

mesh_scene::mesh_scene(context& ctx, size_t max_capacity)
: ctx(&ctx), max_capacity(max_capacity), instance_cache_frame(0)
{
    if(ctx.is_ray_tracing_supported())
        as_update.resize(ctx.get_devices().size());
}

mesh_scene::~mesh_scene()
{
}

void mesh_scene::add(mesh_object& o)
{
    sorted_insert(objects, &o);
    invalidate_tlas();
}

void mesh_scene::remove(mesh_object& o)
{
    sorted_erase(objects, &o);
    invalidate_tlas();
}

void mesh_scene::clear_mesh_objects()
{
    objects.clear();
    invalidate_tlas();
}

const std::vector<mesh_object*>& mesh_scene::get_mesh_objects() const
{
    return objects;
}

size_t mesh_scene::get_instance_count() const
{
    size_t total = 0;
    for(const mesh_object* o: objects)
    {
        if(!o) continue;
        const model* m = o->get_model();
        if(!m) continue;
        total += m->group_count();
    }
    return total;
}

size_t mesh_scene::get_sampler_count() const
{
    std::unordered_set<
        combined_tex_sampler, combined_tex_sampler_hash
    > samplers;

    for(const mesh_object* o: objects)
    {
        if(!o) continue;
        const model* m = o->get_model();
        if(!m) continue;
        for(const auto& group: *m)
        {
            samplers.insert(group.mat.albedo_tex);
            samplers.insert(group.mat.metallic_roughness_tex);
            samplers.insert(group.mat.normal_tex);
            samplers.insert(group.mat.emission_tex);
        }
    }
    return samplers.size();
}

size_t mesh_scene::get_max_capacity() const
{
    return max_capacity;
}

void mesh_scene::refresh_instance_cache(bool force)
{
    uint64_t frame_counter = ctx->get_frame_counter();
    if(!force && instance_cache_frame == frame_counter)
        return;
    instance_cache_frame = frame_counter;
    size_t i = 0;
    mat4 transform;
    mat4 normal_transform;
    for(const mesh_object* o: objects)
    {
        if(!o) continue;
        const model* m = o->get_model();
        if(!m) continue;
        bool fetched_transforms = false;
        for(const auto& vg: *m)
        {
            if(i == instance_cache.size())
            {
                instance_cache.push_back({
                    mat4(0),
                    mat4(0),
                    mat4(0),
                    nullptr,
                    nullptr,
                    nullptr,
                    frame_counter
                });
            }
            instance& inst = instance_cache[i];

            if(inst.mat != &vg.mat)
            {
                inst.mat = &vg.mat;
                inst.prev_transform = mat4(0);
                inst.last_refresh_frame = frame_counter;
            }
            if(inst.m != vg.m)
            {
                inst.m = vg.m;
                inst.prev_transform = mat4(0);
                inst.last_refresh_frame = frame_counter;
            }
            if(inst.o != o)
            {
                inst.o = o;
                inst.prev_transform = mat4(0);
                inst.last_refresh_frame = frame_counter;
            }
            if(inst.last_refresh_frame == frame_counter || !o->is_static())
            {
                if(!fetched_transforms)
                {
                    transform = o->get_global_transform();
                    normal_transform = o->get_global_inverse_transpose_transform();
                    fetched_transforms = true;
                }

                if(inst.prev_transform != inst.transform)
                {
                    inst.prev_transform = inst.transform;
                    inst.last_refresh_frame = frame_counter;
                }
                if(inst.transform != transform)
                {
                    inst.transform = transform;
                    inst.normal_transform = normal_transform;
                    inst.last_refresh_frame = frame_counter;
                }
            }
            ++i;
        }
    }
    instance_cache.resize(i);
}

bool mesh_scene::reserve_pre_transformed_vertices(size_t device_index, size_t max_vertex_count)
{
    if(!ctx->is_ray_tracing_supported())
        return false;

    auto& as = as_update[device_index];
    if(as.pre_transformed_vertex_count < max_vertex_count)
    {
        as.pre_transformed_vertices = create_buffer(
            ctx->get_devices()[device_index],
            {
                {}, max_vertex_count * sizeof(mesh::vertex),
                vk::BufferUsageFlagBits::eVertexBuffer|vk::BufferUsageFlagBits::eStorageBuffer,
                vk::SharingMode::eExclusive
            },
            VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
        );
        as.pre_transformed_vertex_count = max_vertex_count;
        return true;
    }
    return false;
}

void mesh_scene::clear_pre_transformed_vertices(size_t device_index)
{
    if(!ctx->is_ray_tracing_supported())
        return;

    auto& as = as_update[device_index];
    if(as.pre_transformed_vertex_count != 0)
    {
        as.pre_transformed_vertices.drop();
        as.pre_transformed_vertex_count = 0;
    }
}

std::vector<vk::DescriptorBufferInfo> mesh_scene::get_vertex_buffer_bindings(
    size_t device_index
) const {
    std::vector<vk::DescriptorBufferInfo> dbi_vertex;
    if(ctx->is_ray_tracing_supported())
    {
        auto& as = as_update[device_index];
        if(as.pre_transformed_vertex_count != 0)
        {
            size_t offset = 0;
            for(size_t i = 0; i < instance_cache.size(); ++i)
            {
                const mesh* m = instance_cache[i].m;
                size_t bytes = m->get_vertices().size() * sizeof(mesh::vertex);
                dbi_vertex.push_back({as.pre_transformed_vertices, offset, bytes});
                offset += bytes;
            }
            return dbi_vertex;
        }
    }

    for(size_t i = 0; i < instance_cache.size(); ++i)
    {
        const mesh* m = instance_cache[i].m;
        vk::Buffer vertex_buffer = m->get_vertex_buffer(device_index);
        dbi_vertex.push_back({vertex_buffer, 0, VK_WHOLE_SIZE});
    }
    return dbi_vertex;
}

std::vector<vk::DescriptorBufferInfo> mesh_scene::get_index_buffer_bindings(
    size_t device_index
) const {
    std::vector<vk::DescriptorBufferInfo> dbi_index;
    for(size_t i = 0; i < instance_cache.size(); ++i)
    {
        const mesh* m = instance_cache[i].m;
        vk::Buffer index_buffer = m->get_index_buffer(device_index);
        dbi_index.push_back({index_buffer, 0, VK_WHOLE_SIZE});
    }
    return dbi_index;
}

void mesh_scene::refresh_dynamic_acceleration_structures(
    size_t device_index,
    size_t frame_index,
    vk::CommandBuffer cmd
){
    // Run BLAS updates
    for(mesh_object* obj: objects)
    {
        model* m = const_cast<model*>(obj->get_model());
        if(!m) continue;
        if(m->has_joints_buffer())
        {
            for(auto& vg: *m)
            {
                per_mesh_blas_cache.at(vg.m->get_id()).rebuild(
                    device_index, frame_index, cmd,
                    {{vg.m, mat4(1), vg.mat.potentially_transparent()}},
                    {}
                );
            }
        }
    }
}

vk::Buffer mesh_scene::get_pre_transformed_vertices(size_t device_index)
{
    auto& as = as_update[device_index];
    return *as.pre_transformed_vertices;
}

const std::vector<mesh_scene::instance>& mesh_scene::get_instances() const
{
    return instance_cache;
}

void mesh_scene::update_acceleration_structures(
    size_t device_index,
    uint32_t frame_index,
    bool& need_scene_reset,
    bool& command_buffers_outdated
){
    auto& as = as_update[device_index];
    auto& f = as.per_frame[frame_index];

    need_scene_reset |= as.tlas_reset_needed;
    command_buffers_outdated |= f.command_buffers_outdated;

    if(as.tlas_reset_needed)
        ensure_blas();

    as.tlas_reset_needed = false;
    f.command_buffers_outdated = false;
}

void mesh_scene::record_acceleration_structure_build(
    vk::CommandBuffer&, size_t, uint32_t, bool
){
    // Unused, for now.
}

void mesh_scene::add_acceleration_structure_instances(
    vk::AccelerationStructureInstanceKHR* instances,
    size_t device_index,
    uint32_t,
    size_t& instance_index,
    size_t capacity
) const {
    // Update instance staging buffer
    for(size_t j = 0; j < instance_cache.size() && instance_index < capacity; ++j)
    {
        const instance& in = instance_cache[j];
        vk::AccelerationStructureInstanceKHR inst = vk::AccelerationStructureInstanceKHR(
            {}, j, 1<<0, 0, // Hit group 0 for triangle meshes.
            vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable,
            per_mesh_blas_cache.at(in.m->get_id()).get_blas_address(device_index)
        );
        mat4 global_transform = transpose(in.transform);
        memcpy(
            (void*)&inst.transform,
            (void*)&global_transform,
            sizeof(inst.transform)
        );
        instances[instance_index++] = inst;
    }
}

void mesh_scene::invalidate_tlas()
{
    for(auto& as: as_update)
    {
        as.tlas_reset_needed = true;
        for(auto& f: as.per_frame)
            f.command_buffers_outdated = true;
    }
}

void mesh_scene::ensure_blas()
{
    // Goes through all IDs and ensures they have valid BLASes.
    for(const mesh_object* o: objects)
    {
        if(!o) continue;
        const model* m = o->get_model();
        if(!m) continue;
        for(const auto& group: *m)
        {
            auto it = per_mesh_blas_cache.find(group.m->get_id());
            if(it != per_mesh_blas_cache.end())
                continue;

            per_mesh_blas_cache.emplace(
                group.m->get_id(),
                bottom_level_acceleration_structure(
                    *ctx,
                    {{group.m, mat4(1), !group.mat.potentially_transparent()}},
                    !group.mat.double_sided,
                    group.m->is_skinned() || group.m->get_animation_source()
                )
            );
        }
    }
}

}
