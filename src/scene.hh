#ifndef TAURAY_SCENE_HH
#define TAURAY_SCENE_HH
#include "monkeroecs.hh"
#include "math.hh"
#include "animation.hh"
#include <set>

namespace tr
{
using namespace monkero;

class environment_map;
class sh_grid;
// Used for internal camera list reordering; it's needed for spatial
// reprojection from sparsely rendered viewports.
struct camera_metadata
{
    bool enabled;
    int index;
    bool actively_rendered;
};

// You can listen to this if you want to track the time passing on scene
// updates.
struct animation_update_event
{
    bool reset;
    time_ticks delta;
};

void set_camera_jitter(scene& s, const std::vector<vec2>& jitter);
std::vector<entity> get_sorted_cameras(scene& s);

std::vector<uint32_t> get_viewport_reorder_mask(
    const std::set<int>& active_indices,
    size_t viewport_count
);

size_t get_instance_count(scene& s);
size_t get_sampler_count(scene& s);
environment_map* get_environment_map(scene& s);
vec3 get_ambient_light(scene& s);
void auto_assign_shadow_maps(
    scene& s,
    unsigned directional_res = 2048,
    vec3 directional_volume = vec3(10, 10, 100),
    vec2 directional_bias = vec2(0.01, 0.05),
    unsigned cascades = 4,
    unsigned point_res = 512,
    float point_near = 0.01f,
    vec2 point_bias = vec2(0.006, 0.02)
);
void track_shadow_maps(scene& s);

sh_grid* get_sh_grid(scene& s, vec3 pos, int* index = nullptr);
sh_grid* get_largest_sh_grid(scene& s, int* index = nullptr);

void play(
    scene& s,
    const std::string& name,
    bool loop = false,
    bool use_fallback = false
);
void update(scene& s, time_ticks dt, bool force_update = false);
bool is_playing(scene& s);
void set_animation_time(scene& s, time_ticks dt);

}

#endif
