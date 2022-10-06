#include "light_scene.hh"
#include "sh_grid.hh"
#include "misc.hh"

namespace tr
{

light_scene::light_scene(context& ctx, size_t max_capacity)
: aabb_scene(ctx, "light BLAS update", 2, max_capacity)
{
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
    for(size_t j = 0; j < point_lights.size() && i < get_max_capacity(); ++j)
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
    for(size_t j = 0; j < spotlights.size() && i < get_max_capacity(); ++j)
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

}
