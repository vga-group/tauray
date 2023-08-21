#include "scene.hh"
#include "camera.hh"
#include "misc.hh"
#include "sh_grid.hh"
#include "placeholders.hh"
#include "environment_map.hh"
#include "light.hh"
#include "shadow_map.hh"
#include <unordered_set>

namespace tr
{

scene::scene(): total_ticks(0)
{
}

void scene::set_camera(camera* cam) { cameras = {cam}; }

camera* scene::get_camera(unsigned index) const
{
    if(index >= cameras.size()) return nullptr;
    return cameras[index];
}

void scene::add(camera& c) { unsorted_insert(cameras, &c); }
void scene::remove(camera& c) { unsorted_erase(cameras, &c); }
void scene::clear_cameras() { cameras.clear(); }
const std::vector<camera*>& scene::get_cameras() const { return cameras; }

void scene::reorder_cameras_by_active(const std::set<int>& active_indices)
{
    std::vector<camera*> new_cameras;
    for(size_t i = 0; i < cameras.size(); ++i)
    {
        if(active_indices.count(i))
            new_cameras.push_back(cameras[i]);
    }
    for(size_t i = 0; i < cameras.size(); ++i)
    {
        if(!active_indices.count(i))
            new_cameras.push_back(cameras[i]);
    }
    cameras = std::move(new_cameras);
}

void scene::set_camera_jitter(const std::vector<vec2>& jitter)
{
    for(camera* cam: cameras)
        cam->set_jitter(jitter);
}

void scene::add_control_node(animated_node& o)
{ sorted_insert(control_nodes, &o); }
void scene::remove_control_node(animated_node& o)
{ sorted_erase(control_nodes, &o); }
void scene::clear_control_nodes() { control_nodes.clear(); }

void scene::add(mesh_object& o)
{
    sorted_insert(objects, &o);
}

void scene::remove(mesh_object& o)
{
    sorted_erase(objects, &o);
}

void scene::clear_mesh_objects()
{
    objects.clear();
}

const std::vector<mesh_object*>& scene::get_mesh_objects() const
{
    return objects;
}

size_t scene::get_instance_count() const
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

size_t scene::get_sampler_count() const
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

void scene::set_environment_map(environment_map* envmap)
{
    this->envmap = envmap;
}

environment_map* scene::get_environment_map() const
{
    return envmap;
}

void scene::set_ambient(vec3 ambient)
{
    this->ambient = ambient;
}

vec3 scene::get_ambient() const
{
    return ambient;
}

void scene::add(point_light& pl)
{
    sorted_insert(point_lights, &pl);
}

void scene::remove(point_light& pl)
{
    sorted_erase(point_lights, &pl);
    point_shadow_maps.erase(&pl);
}

void scene::clear_point_lights()
{
    for(point_light* pl: point_lights) point_shadow_maps.erase(pl);
    point_lights.clear();
}

const std::vector<point_light*>& scene::get_point_lights() const
{ return point_lights; }

void scene::add(spotlight& sl)
{
    sorted_insert(spotlights, &sl);
}

void scene::remove(spotlight& sl)
{
    sorted_erase(spotlights, &sl);
    point_shadow_maps.erase(&sl);
}

void scene::clear_spotlights()
{
    for(spotlight* sl: spotlights) point_shadow_maps.erase(sl);
    spotlights.clear();
}

const std::vector<spotlight*>& scene::get_spotlights() const
{ return spotlights; }

void scene::add(directional_light& dl)
{ sorted_insert(directional_lights, &dl); }

void scene::remove(directional_light& dl)
{
    directional_shadow_maps.erase(&dl);
    sorted_erase(directional_lights, &dl);
}

void scene::clear_directional_lights()
{
    directional_shadow_maps.clear();
    directional_lights.clear();
}

const std::vector<directional_light*>&
scene::get_directional_lights() const
{ return directional_lights; }

void scene::auto_shadow_maps(
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

const directional_shadow_map* scene::get_shadow_map(
    const directional_light* dl
) const
{
    auto it = directional_shadow_maps.find(dl);
    if(it == directional_shadow_maps.end()) return nullptr;
    return &it->second;
}

const point_shadow_map* scene::get_shadow_map(const point_light* pl) const
{
    auto it = point_shadow_maps.find(pl);
    if(it == point_shadow_maps.end()) return nullptr;
    return &it->second;
}

void scene::track_shadow_maps(const std::vector<camera*>& cam)
{
    for(auto& pair: directional_shadow_maps)
    {
        pair.second.track_cameras(pair.first->get_global_transform(), cam);
    }
}

void scene::add(sh_grid& sh) { sorted_insert(sh_grids, &sh); }
void scene::remove(sh_grid& sh) { sorted_erase(sh_grids, &sh); }
void scene::clear_sh_grids() { sh_grids.clear(); }
const std::vector<sh_grid*>& scene::get_sh_grids() const { return sh_grids; }

sh_grid* scene::get_sh_grid(vec3 pos, int* index) const
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

sh_grid* scene::get_largest_sh_grid(int* index) const
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

void scene::clear()
{
    clear_cameras();
    clear_mesh_objects();
    clear_point_lights();
    clear_spotlights();
    clear_directional_lights();
    clear_control_nodes();
}

void scene::play(const std::string& name, bool loop, bool use_fallback)
{
    auto play_handler = [&](animated_node* n){
        n->play(name, loop, use_fallback);
    };
    for(camera* c: cameras) play_handler(c);
    for(animated_node* o: control_nodes) play_handler(o);

    for(point_light* l: point_lights) l->play(name, loop, use_fallback);
    for(spotlight* l: spotlights) l->play(name, loop, use_fallback);
    for(directional_light* l: directional_lights) l->play(name, loop, use_fallback);
    for(mesh_object* o: objects) if(!o->is_static()) o->play(name, loop, use_fallback);
}

void scene::update(time_ticks dt, bool force_update)
{
    for(camera* c: cameras)
    {
        c->step_jitter();
    }

    if(dt > 0 || force_update)
    {
        for(camera* c: cameras) c->update(dt);
        for(animated_node* o: control_nodes) o->update(dt);
        for(point_light* l: point_lights) l->update(dt);
        for(spotlight* l: spotlights) l->update(dt);
        for(directional_light* l: directional_lights) l->update(dt);
        for(mesh_object* o: objects) if(!o->is_static()) o->update(dt);
    }
    total_ticks += dt;
}

void scene::set_animation_time(time_ticks dt)
{
    auto reset_handler = [&](animated_node* n){
        n->restart();
        n->update(dt);
    };
    for(camera* c: cameras) reset_handler(c);
    for(animated_node* o: control_nodes) reset_handler(o);

    for(point_light* l: point_lights) reset_handler(l);
    for(spotlight* l: spotlights) reset_handler(l);
    for(directional_light* l: directional_lights) reset_handler(l);
    for(mesh_object* o: objects) if(!o->is_static()) reset_handler(o);
    total_ticks = dt;
}

time_ticks scene::get_total_ticks() const
{
    return total_ticks;
}

bool scene::is_playing() const
{
    bool playing = false;
    auto check_playing = [&](animated_node* n){
        if(n->is_playing()) playing = true;
    };
    for(camera* c: cameras) check_playing(c);
    for(animated_node* o: control_nodes) check_playing(o);

    for(point_light* l: point_lights) check_playing(l);
    for(spotlight* l: spotlights) check_playing(l);
    for(directional_light* l: directional_lights) check_playing(l);
    for(mesh_object* o: objects) if(!o->is_static()) check_playing(o);

    return playing;
}

std::vector<uint32_t> get_viewport_reorder_mask(
    const std::set<int>& active_indices,
    size_t viewport_count
){
    std::vector<uint32_t> reorder;
    for(size_t i = 0; i < viewport_count; ++i)
    {
        if(active_indices.count(i))
            reorder.push_back(i);
    }
    for(size_t i = 0; i < viewport_count; ++i)
    {
        if(!active_indices.count(i))
            reorder.push_back(i);
    }
    return reorder;
}

}
