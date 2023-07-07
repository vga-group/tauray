#include "mesh_scene.hh"
#include "misc.hh"
#include "log.hh"
#include <unordered_set>

namespace tr
{

mesh_scene::mesh_scene(device_mask dev, size_t max_capacity)
:   max_capacity(max_capacity), as_update(dev),
    group_strategy(blas_strategy::PER_MATERIAL), instance_cache_frame(0)
{
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

void mesh_scene::set_blas_strategy(blas_strategy strat)
{
    group_strategy = strat;
    invalidate_tlas();
}

size_t mesh_scene::get_blas_group_count() const
{
    return group_cache.size();
}

void mesh_scene::refresh_instance_cache(bool force)
{
    uint64_t frame_counter = as_update.get_context()->get_frame_counter();
    if(!force && instance_cache_frame == frame_counter)
        return;
    instance_cache_frame = frame_counter;
    size_t i = 0;
    size_t last_object_index = SIZE_MAX;
    group_cache.clear();
    auto add_instances = [&](bool static_mesh, bool static_transformable){
        for(size_t object_index = 0; object_index < objects.size(); ++object_index)
        {
            const mesh_object* o = objects[object_index];
            if(!o) continue;

            // If requesting dynamic meshes, we don't care about the
            // transformable staticness any more.
            if(static_mesh && static_transformable != o->is_static())
                continue;

            const model* m = o->get_model();
            if(!m) continue;
            bool fetched_transforms = false;
            mat4 transform;
            mat4 normal_transform;
            for(const auto& vg: *m)
            {
                bool is_static = !vg.m->is_skinned() && !vg.m->get_animation_source();
                if(static_mesh != is_static)
                    continue;

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

                assign_group_cache(
                    vg.m->get_id(),
                    static_mesh,
                    static_transformable,
                    object_index,
                    last_object_index
                );

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
    };
    add_instances(true, true);
    add_instances(true, false);
    add_instances(false, false);
    instance_cache.resize(i);
    if(force) ensure_blas();
}

bool mesh_scene::reserve_pre_transformed_vertices(size_t max_vertex_count)
{
    if(!as_update.get_context()->is_ray_tracing_supported())
        return false;

    bool ret = false;
    for(auto[dev, as]: as_update)
    {
        if(as.pre_transformed_vertex_count < max_vertex_count)
        {
            as.pre_transformed_vertices = create_buffer(
                dev,
                {
                    {}, max_vertex_count * sizeof(mesh::vertex),
                    vk::BufferUsageFlagBits::eVertexBuffer|vk::BufferUsageFlagBits::eStorageBuffer,
                    vk::SharingMode::eExclusive
                },
                VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
            );
            as.pre_transformed_vertex_count = max_vertex_count;
            ret = true;
        }
    }
    return ret;
}

void mesh_scene::clear_pre_transformed_vertices()
{
    if(!as_update.get_context()->is_ray_tracing_supported())
        return;

    for(auto[dev, as]: as_update)
    {
        if(as.pre_transformed_vertex_count != 0)
        {
            as.pre_transformed_vertices.drop();
            as.pre_transformed_vertex_count = 0;
        }
    }
}

std::vector<vk::DescriptorBufferInfo> mesh_scene::get_vertex_buffer_bindings(
    device_id id
) const {
    std::vector<vk::DescriptorBufferInfo> dbi_vertex;
    if(as_update.get_context()->is_ray_tracing_supported())
    {
        auto& as = as_update[id];
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
        vk::Buffer vertex_buffer = m->get_vertex_buffer(id);
        dbi_vertex.push_back({vertex_buffer, 0, VK_WHOLE_SIZE});
    }
    return dbi_vertex;
}

std::vector<vk::DescriptorBufferInfo> mesh_scene::get_index_buffer_bindings(
    device_id id
) const {
    std::vector<vk::DescriptorBufferInfo> dbi_index;
    for(size_t i = 0; i < instance_cache.size(); ++i)
    {
        const mesh* m = instance_cache[i].m;
        vk::Buffer index_buffer = m->get_index_buffer(id);
        dbi_index.push_back({index_buffer, 0, VK_WHOLE_SIZE});
    }
    return dbi_index;
}

void mesh_scene::refresh_dynamic_acceleration_structures(
    device_id id,
    size_t frame_index,
    vk::CommandBuffer cmd
){
    // Run BLAS updates
    size_t offset = 0;
    std::vector<bottom_level_acceleration_structure::entry> entries;
    for(const instance_group& group: group_cache)
    {
        if(group.static_mesh)
        {
            offset += group.size;
            continue;
        }

        entries.clear();
        for(size_t i = 0; i < group.size; ++i, ++offset)
        {
            const instance& inst = instance_cache[offset];
            entries.push_back({
                inst.m,
                0, nullptr,
                group.static_transformable ? inst.transform : mat4(1),
                !inst.mat->potentially_transparent()
            });
        }
        blas_cache.at(group.id).rebuild(
            id, frame_index, cmd, entries,
            group_strategy == blas_strategy::ALL_MERGED_STATIC ? false : true
        );
    }
}

vk::Buffer mesh_scene::get_pre_transformed_vertices(device_id id)
{
    auto& as = as_update[id];
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

    // Run BLAS matrix updates. Only necessary when merged BLASes have dynamic
    // transformables.
    if(group_strategy == blas_strategy::ALL_MERGED_STATIC)
    {
        size_t offset = 0;
        std::vector<bottom_level_acceleration_structure::entry> entries;
        for(const instance_group& group: group_cache)
        {
            entries.clear();
            for(size_t i = 0; i < group.size; ++i, ++offset)
            {
                const instance& inst = instance_cache[offset];
                entries.push_back({
                    inst.m,
                    0, nullptr,
                    group.static_transformable ? inst.transform : mat4(1),
                    !inst.mat->potentially_transparent()
                });
            }
            blas_cache.at(group.id).update_transforms(frame_index, entries);
        }
    }

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
    size_t offset = 0;
    for(size_t j = 0; j < group_cache.size() && instance_index < capacity; ++j)
    {
        const instance_group& group = group_cache[j];
        const bottom_level_acceleration_structure& blas = blas_cache.at(group.id);
        vk::AccelerationStructureInstanceKHR inst = vk::AccelerationStructureInstanceKHR(
            {}, offset, 1<<0, 0, // Hit group 0 for triangle meshes.
            {}, blas.get_blas_address(device_index)
        );
        if(!blas.is_backface_culled())
            inst.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

        mat4 global_transform = group.static_transformable ?
            mat4(1) : transpose(instance_cache[offset].transform);
        memcpy(
            (void*)&inst.transform,
            (void*)&global_transform,
            sizeof(inst.transform)
        );
        instances[instance_index++] = inst;
        offset += group.size;
    }
}

void mesh_scene::invalidate_tlas()
{
    for(auto[dev, as]: as_update)
    {
        as.tlas_reset_needed = true;
        for(auto& f: as.per_frame)
            f.command_buffers_outdated = true;
    }
}

void mesh_scene::ensure_blas()
{
    if(!as_update.get_context()->is_ray_tracing_supported())
        return;
    bool built_one = false;
    // Goes through all groups and ensures they have valid BLASes.
    size_t offset = 0;
    std::vector<bottom_level_acceleration_structure::entry> entries;
    for(const instance_group& group: group_cache)
    {
        auto it = blas_cache.find(group.id);
        if(it != blas_cache.end())
        {
            offset += group.size;
            continue;
        }

        if(!built_one)
            TR_LOG("Building acceleration structures");

        built_one = true;

        entries.clear();
        bool double_sided = false;
        for(size_t i = 0; i < group.size; ++i, ++offset)
        {
            const instance& inst = instance_cache[offset];
            if(inst.mat->double_sided) double_sided = true;
            entries.push_back({
                inst.m,
                0, nullptr,
                group.static_transformable ? inst.transform : mat4(1),
                !inst.mat->potentially_transparent()
            });
        }
        blas_cache.emplace(
            group.id,
            bottom_level_acceleration_structure(
                as_update.get_mask(),
                entries,
                !double_sided,
                group_strategy == blas_strategy::ALL_MERGED_STATIC ? false : !group.static_mesh,
                group.static_mesh
            )
        );
    }
    if(built_one)
        TR_LOG("Finished building acceleration structures");
}

void mesh_scene::assign_group_cache(
    uint64_t id,
    bool static_mesh,
    bool static_transformable,
    size_t object_index,
    size_t& last_object_index
){
    switch(group_strategy)
    {
    case blas_strategy::PER_MATERIAL:
        group_cache.push_back({id, 1, static_mesh, false});
        break;
    case blas_strategy::PER_MODEL:
        if(last_object_index == object_index)
        {
            instance_group& group = group_cache.back();
            group.id = hash_combine(group.id, id);
            if(!static_mesh) group.static_mesh = false;
            group.size++;
        }
        else group_cache.push_back({id, 1, static_mesh, false});
        break;
    case blas_strategy::STATIC_MERGED_DYNAMIC_PER_MODEL:
        if(group_cache.size() == 0)
        {
            bool is_static = static_mesh && static_transformable;
            group_cache.push_back({
                id, 1, static_mesh, is_static
            });
        }
        else
        {
            instance_group& group = group_cache.back();
            bool prev_is_static = group.static_mesh && group.static_transformable;
            bool is_static = static_mesh && static_transformable;
            if(prev_is_static && is_static)
            {
                group.id = hash_combine(group.id, id);
                group.size++;
            }
            else
            {
                if(last_object_index == object_index)
                {
                    group.id = hash_combine(group.id, id);
                    if(!static_mesh) group.static_mesh = false;
                    group.size++;
                }
                else group_cache.push_back({id, 1, static_mesh, false});
            }
        }
        break;
    case blas_strategy::ALL_MERGED_STATIC:
        {
            if(group_cache.size() == 0)
                group_cache.push_back({0, 0, true, true});
            instance_group& group = group_cache.back();
            group.id = hash_combine(group.id, id);
            if(!static_mesh) group.static_mesh = false;
            group.size++;
        }
        break;
    }
    last_object_index = object_index;
}

}
