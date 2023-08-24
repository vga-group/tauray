#include "scene.hh"
#include "camera.hh"
#include "model.hh"
#include "light.hh"
#include "sh_grid.hh"
#include "shadow_map.hh"
#include "environment_map.hh"
#include <algorithm>
#include <unordered_set>

namespace tr
{

void set_camera_jitter(scene& s, const std::vector<vec2>& jitter)
{
    s.foreach([&](camera& c){
        c.set_jitter(jitter);
    });
}

std::vector<entity> get_sorted_cameras(scene& s)
{
    std::vector<entity> cameras;
    s.foreach([&](entity id, camera&, camera_metadata& md){
        if(md.enabled)
            cameras.push_back(id);
    });

    std::sort(
        cameras.begin(),
        cameras.end(),
        [&](entity a, entity b){
            camera_metadata* ac = s.get<camera_metadata>(a);
            camera_metadata* bc = s.get<camera_metadata>(b);
            if(!ac) return true;
            if(!bc) return false;

            if(ac->actively_rendered && !bc->actively_rendered)
                return true;
            if(!ac->actively_rendered && bc->actively_rendered)
                return false;
            return ac->index < bc->index;
        }
    );
    return cameras;
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

size_t get_instance_count(scene& s)
{
    size_t total = 0;
    s.foreach([&](transformable&, model& mod){
        total += mod.group_count();
    });
    return total;
}

size_t get_sampler_count(scene& s)
{
    std::unordered_set<
        combined_tex_sampler, combined_tex_sampler_hash
    > samplers;

    s.foreach([&](model& mod){
        for(const auto& group: mod)
        {
            samplers.insert(group.mat.albedo_tex);
            samplers.insert(group.mat.metallic_roughness_tex);
            samplers.insert(group.mat.normal_tex);
            samplers.insert(group.mat.emission_tex);
        }
    });
    return samplers.size();
}

environment_map* get_environment_map(scene& s)
{
    environment_map* found = nullptr;
    // We assume that there's only one. Hopefully that's true.
    s.foreach([&](environment_map& em){
        found = &em;
    });
    return found;
}

vec3 get_ambient_light(scene& s)
{
    vec3 sum = vec3(0);
    s.foreach([&](ambient_light& al){
        sum += al.color;
    });
    return sum;
}

void auto_assign_shadow_maps(
    scene& s,
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

    s.foreach([&](entity id, point_light&){
        s.emplace<point_shadow_map>(id, psm);
    });

    s.foreach([&](entity id, spotlight&){
        s.emplace<point_shadow_map>(id, psm);
    });

    directional_shadow_map dsm;
    dsm.resolution = uvec2(directional_res);
    dsm.x_range = vec2(-directional_volume.x, directional_volume.x);
    dsm.y_range = vec2(-directional_volume.y, directional_volume.y);
    dsm.depth_range = vec2(-directional_volume.z, directional_volume.z);
    dsm.min_bias = directional_bias.x;
    dsm.max_bias = directional_bias.y;
    dsm.cascades.resize(cascades);

    s.foreach([&](entity id, directional_light&){
        s.emplace<directional_shadow_map>(id, dsm);
    });
}

void track_shadow_maps(scene& s)
{
    std::vector<camera*> cameras;
    std::vector<transformable*> camera_transforms;
    s.foreach([&](transformable& t, camera& c){
        cameras.push_back(&c); camera_transforms.push_back(&t);
    });
    if(cameras.size() == 0) return;
    s.foreach([&](transformable& t, directional_shadow_map& dsm){
        dsm.track_cameras(t.get_global_transform(), cameras, camera_transforms);
    });
}

sh_grid* get_sh_grid(scene& s, vec3 pos, int* index)
{
    float closest_distance = std::numeric_limits<float>::infinity();
    float densest = 0.0f;
    entity best = INVALID_ENTITY;
    int cur_index = 0;
    s.foreach([&](entity id, transformable& self, sh_grid& g){
        float distance = g.point_distance(self, pos);
        if(distance >= 0 && distance <= closest_distance)
        {
            closest_distance = distance;
            if(distance == 0)
            {
                float density = g.calc_density(self);
                if(density > densest)
                {
                    densest = density;
                    best = id;
                    if(index) *index = cur_index;
                }
            }
            else
            {
                best = id;
                if(index) *index = cur_index;
            }
        }
        cur_index++;
    });
    return best != INVALID_ENTITY ? s.get<sh_grid>(best) : nullptr;
}

sh_grid* get_largest_sh_grid(scene& s, int* index)
{
    float largest = 0.0f;
    entity best = INVALID_ENTITY;
    int cur_index = 0;
    s.foreach([&](entity id, transformable& self, sh_grid& g){
        float volume = g.calc_volume(self);
        if(volume > largest)
        {
            largest = volume;
            best = id;
            if(index) *index = cur_index;
        }
        cur_index++;
    });
    return best != INVALID_ENTITY ? s.get<sh_grid>(best) : nullptr;
}

void play(
    scene& s,
    const std::string& name,
    bool loop,
    bool use_fallback
){
    s.foreach([&](transformable& t, animated& a){
        if(!t.is_static())
            a.play(name, loop, use_fallback);
    });
    s.emit(animation_update_event{true, 0});
}

void update(scene& s, time_ticks dt, bool force_update)
{
    s.foreach([&](camera& c){ c.step_jitter(); });

    if(dt > 0 || force_update)
    {
        s.foreach([&](transformable& t, animated& a){ a.update(t, dt); });
    }
    s.emit(animation_update_event{false, dt});
}

bool is_playing(scene& s)
{
    bool playing = false;
    s.foreach([&](transformable& t, animated& a){
        if(!t.is_static() && a.is_playing()) playing = true;
    });
    return playing;
}

void set_animation_time(scene& s, time_ticks dt)
{
    s.foreach([&](transformable& t, animated& a){
        if(!t.is_static())
        {
            a.restart();
            a.update(t, dt);
        }
    });
    s.emit(animation_update_event{true, dt});
}

}
