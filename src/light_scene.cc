#include "light_scene.hh"
#include "sh_grid.hh"
#include "misc.hh"

namespace tr
{

light_scene::light_scene(device_mask dev, size_t max_capacity)
:   max_capacity(max_capacity),
    blas_update_timer(dev, "light BLAS update"),
    as_update(dev)
{
    if(as_update.get_context()->is_ray_tracing_supported())
    {
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
}

light_scene::~light_scene()
{
}

void light_scene::set_environment_map(environment_map* envmap)
{
    this->envmap = envmap;
}

environment_map* light_scene::get_environment_map() const
{
    return envmap;
}

void light_scene::set_ambient(vec3 ambient)
{
    this->ambient = ambient;
}

vec3 light_scene::get_ambient() const
{
    return ambient;
}

void light_scene::add(point_light& pl)
{
    sorted_insert(point_lights, &pl);
    invalidate_acceleration_structures();
}

void light_scene::remove(point_light& pl)
{
    sorted_erase(point_lights, &pl);
    point_shadow_maps.erase(&pl);
    invalidate_acceleration_structures();
}

void light_scene::clear_point_lights()
{
    for(point_light* pl: point_lights) point_shadow_maps.erase(pl);
    point_lights.clear();
    invalidate_acceleration_structures();
}

const std::vector<point_light*>& light_scene::get_point_lights() const
{ return point_lights; }

void light_scene::add(spotlight& sl)
{
    sorted_insert(spotlights, &sl);
    invalidate_acceleration_structures();
}

void light_scene::remove(spotlight& sl)
{
    sorted_erase(spotlights, &sl);
    point_shadow_maps.erase(&sl);
    invalidate_acceleration_structures();
}

void light_scene::clear_spotlights()
{
    for(spotlight* sl: spotlights) point_shadow_maps.erase(sl);
    spotlights.clear();
    invalidate_acceleration_structures();
}

const std::vector<spotlight*>& light_scene::get_spotlights() const
{ return spotlights; }

void light_scene::add(directional_light& dl)
{ sorted_insert(directional_lights, &dl); }

void light_scene::remove(directional_light& dl)
{
    directional_shadow_maps.erase(&dl);
    sorted_erase(directional_lights, &dl);
}

void light_scene::clear_directional_lights()
{
    directional_shadow_maps.clear();
    directional_lights.clear();
}

const std::vector<directional_light*>&
light_scene::get_directional_lights() const
{ return directional_lights; }

void light_scene::auto_shadow_maps(
    unsigned directional_res,
    vec3 directional_volume,
    vec2 directional_bias,
    unsigned cascades,
    unsigned point_res,
    float point_near,
    vec2 point_bias
){
    point_shadow_map psm;
    psm.resolution = uvec2(point_res);
    psm.near = point_near;
    psm.min_bias = point_bias.x;
    psm.max_bias = point_bias.y;

    for(point_light* pl: point_lights) point_shadow_maps[pl] = psm;
    for(spotlight* sl: spotlights) point_shadow_maps[sl] = psm;

    directional_shadow_map dsm;
    dsm.resolution = uvec2(directional_res);
    dsm.x_range = vec2(-directional_volume.x, directional_volume.x);
    dsm.y_range = vec2(-directional_volume.y, directional_volume.y);
    dsm.depth_range = vec2(-directional_volume.z, directional_volume.z);
    dsm.min_bias = directional_bias.x;
    dsm.max_bias = directional_bias.y;
    dsm.cascades.resize(cascades);

    for(directional_light* dl: directional_lights)
        directional_shadow_maps[dl] = dsm;
}

const directional_shadow_map* light_scene::get_shadow_map(
    const directional_light* dl
) const
{
    auto it = directional_shadow_maps.find(dl);
    if(it == directional_shadow_maps.end()) return nullptr;
    return &it->second;
}

const point_shadow_map* light_scene::get_shadow_map(const point_light* pl) const
{
    auto it = point_shadow_maps.find(pl);
    if(it == point_shadow_maps.end()) return nullptr;
    return &it->second;
}

void light_scene::track_shadow_maps(const std::vector<camera*>& cam)
{
    for(auto& pair: directional_shadow_maps)
    {
        pair.second.track_cameras(pair.first->get_global_transform(), cam);
    }
}

void light_scene::add(sh_grid& sh) { sorted_insert(sh_grids, &sh); }
void light_scene::remove(sh_grid& sh) { sorted_erase(sh_grids, &sh); }
void light_scene::clear_sh_grids() { sh_grids.clear(); }
const std::vector<sh_grid*>& light_scene::get_sh_grids() const
{ return sh_grids; }

sh_grid* light_scene::get_sh_grid(vec3 pos, int* index) const
{
    float closest_distance = std::numeric_limits<float>::infinity();
    float densest = 0.0f;
    int best_index = -1;
    for(size_t i = 0; i < sh_grids.size(); ++i)
    {
        sh_grid* g = sh_grids[i];
        if(!g) continue;

        float distance = g->point_distance(pos);
        if(distance < 0) continue;

        if(distance <= closest_distance)
        {
            closest_distance = distance;
            if(distance == 0)
            {
                float density = g->calc_density();
                if(density > densest)
                {
                    densest = density;
                    best_index = i;
                }
            }
            else best_index = i;
        }
    }
    if(index) *index = best_index;
    return best_index > 0 ? sh_grids[best_index] : nullptr;
}

sh_grid* light_scene::get_largest_sh_grid(int* index) const
{
    // Fast path: if there's just one, that will always be the largest one.
    if(sh_grids.size() == 1)
    {
        if(index) *index = 0;
        return sh_grids[0];
    }

    float largest = 0.0f;
    int best_index = -1;
    for(size_t i = 0; i < sh_grids.size(); ++i)
    {
        sh_grid* g = sh_grids[i];
        if(!g) continue;

        float volume = g->calc_volume();
        if(volume > largest)
        {
            largest = volume;
            best_index = i;
        }
    }
    if(index) *index = best_index;
    return best_index > 0 ? sh_grids[best_index] : nullptr;
}

size_t light_scene::get_aabbs(vk::AabbPositionsKHR* aabb)
{
    size_t i = 0;
    for(size_t j = 0; j < point_lights.size() && i < max_capacity; ++j)
    {
        point_light* pl = point_lights[j];

        float radius = pl->get_radius();

        vec3 pos = radius == 0.0f ? vec3(0) : pl->get_global_position();

        vec3 min = pos - vec3(radius);
        vec3 max = pos + vec3(radius);

        aabb[i] = vk::AabbPositionsKHR(
            min.x, min.y, min.z, max.x, max.y, max.z);
        i++;
    }
    for(size_t j = 0; j < spotlights.size() && i < max_capacity; ++j)
    {
        spotlight* sl = spotlights[j];

        float radius = sl->get_radius();
        vec3 pos = radius == 0.0f ? vec3(0) : sl->get_global_position();

        vec3 min = pos - vec3(radius);
        vec3 max = pos + vec3(radius);

        aabb[i] = vk::AabbPositionsKHR(
            min.x, min.y, min.z, max.x, max.y, max.z);
        i++;
    }
    return i;
}

void light_scene::invalidate_acceleration_structures()
{
    as_update([&](device&, as_update_data& as){
        as.scene_reset_needed = true;
        for(auto& f: as.per_frame)
            f.command_buffers_outdated = true;
    });
}

size_t light_scene::get_max_capacity() const
{
    return max_capacity;
}

void light_scene::update_acceleration_structures(
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

void light_scene::record_acceleration_structure_build(
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

void light_scene::add_acceleration_structure_instances(
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
            {}, instance_index, 1<<1, 2,
            vk::GeometryInstanceFlagBitsKHR::eTriangleCullDisable,
            blas->get_blas_address(id)
        );
        mat4 id_mat = mat4(1.0f);
        memcpy((void*)&inst.transform, (void*)&id_mat, sizeof(inst.transform));
    }
}

}
