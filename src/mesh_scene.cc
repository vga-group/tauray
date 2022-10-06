#include "mesh_scene.hh"
#include "misc.hh"
#include <unordered_set>

namespace tr
{

mesh_scene::mesh_scene(context& ctx, size_t max_capacity)
: ctx(&ctx), max_capacity(max_capacity), instance_cache_frame(0)
{
    if(ctx.is_ray_tracing_supported())
        acceleration_structures.resize(ctx.get_devices().size());
}

mesh_scene::~mesh_scene()
{
}

void mesh_scene::add(mesh_object& o)
{
    sorted_insert(objects, &o);
    const model* m = o.get_model();
    if(m) for(const auto& vg: *m) find_mesh(vg.m)->second++;
    invalidate_mesh_ids();
    invalidate_acceleration_structures();
}

void mesh_scene::remove(mesh_object& o)
{
    sorted_erase(objects, &o);
    const model* m = o.get_model();
    if(m) for(const auto& vg: *m)
    {
        auto it = find_mesh(vg.m);
        it->second--;
        if(it->second <= 0)
            meshes.erase(it);
    }
    invalidate_mesh_ids();
    invalidate_acceleration_structures();
}

void mesh_scene::clear_mesh_objects()
{
    objects.clear();
    invalidate_mesh_ids();
    invalidate_acceleration_structures();
}

const std::vector<mesh_object*>& mesh_scene::get_mesh_objects() const
{
    return objects;
}

const std::vector<std::pair<const mesh*, int>>& mesh_scene::get_meshes() const
{
    return meshes;
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

size_t mesh_scene::get_mesh_count() const
{
    return meshes.size();
}

unsigned mesh_scene::find_mesh_id(const mesh* m) const
{
    if(mesh_ids.size() != meshes.size())
    {
        mesh_ids.clear();
        for(unsigned i = 0; i < meshes.size(); ++i)
            mesh_ids[meshes[i].first] = i;
    }

    auto it = mesh_ids.find(m);
    if(it == mesh_ids.end())
        throw std::runtime_error("Mesh instance is not from this scene!");
    return it->second;
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
    for(const mesh_object* o: objects)
    {
        if(!o) continue;
        const model* m = o->get_model();
        if(!m) continue;
        mat4 transform = o->get_global_transform();
        mat4 normal_transform = transpose(o->get_global_inverse_transform());
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
            ++i;
        }
    }
    instance_cache.resize(i);
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
    auto& as = acceleration_structures[device_index];
    auto& f = as.per_frame[frame_index];

    need_scene_reset |= as.scene_reset_needed;
    command_buffers_outdated |= f.command_buffers_outdated;

    as.scene_reset_needed = false;
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
        mat4 global_transform = transpose(in.transform);
        vk::AccelerationStructureInstanceKHR inst = vk::AccelerationStructureInstanceKHR(
            {}, j, 1<<0, 0, // Hit group 0 for triangle meshes.
            vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable,
            in.m->get_blas_address(device_index)
        );
        memcpy(
            (void*)&inst.transform,
            (void*)&global_transform,
            sizeof(inst.transform)
        );
        memcpy(
            &instances[instance_index++],
            &inst,
            sizeof(inst)
        );
    }
}

void mesh_scene::invalidate_acceleration_structures()
{
    for(auto& as: acceleration_structures)
    {
        as.scene_reset_needed = true;
        for(auto& f: as.per_frame)
            f.command_buffers_outdated = true;
    }
}

void mesh_scene::invalidate_mesh_ids()
{
    mesh_ids.clear();
}

std::vector<std::pair<const mesh*, int>>::iterator mesh_scene::find_mesh(const mesh* m)
{
    auto it = std::lower_bound(
        meshes.begin(), meshes.end(), m,
        [](const std::pair<const mesh*, int>& a, const mesh* b)
        {
            return a.first < b;
        }
    );
    if(it == meshes.end() || it->first != m)
        return meshes.insert(it, {m, 0});
    return it;
}

}
