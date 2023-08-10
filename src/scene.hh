#ifndef TAURAY_SCENE_HH
#define TAURAY_SCENE_HH
#include "mesh_object.hh"
#include "shadow_map.hh"

namespace tr
{

class camera;
class environment_map;
class sh_grid;
class light;
class point_light;
class spotlight;
class directional_light;
class basic_pipeline;

class scene
{
public:
    scene();
    scene(const scene& s) = delete;
    scene(scene&& s) = delete;

    void set_camera(camera* cam);
    camera* get_camera(unsigned index = 0) const;

    void add(camera& c);
    void remove(camera& c);
    void clear_cameras();
    const std::vector<camera*>& get_cameras() const;
    void reorder_cameras_by_active(const std::set<int>& active_indices);
    void set_camera_jitter(const std::vector<vec2>& jitter);

    void add_control_node(animated_node& o);
    void remove_control_node(animated_node& o);
    void clear_control_nodes();

    void add(mesh_object& o);
    void remove(mesh_object& o);
    void clear_mesh_objects();
    const std::vector<mesh_object*>& get_mesh_objects() const;

    // These can be very slow!
    size_t get_instance_count() const;
    size_t get_sampler_count() const;

    void set_environment_map(environment_map* envmap = nullptr);
    environment_map* get_environment_map() const;

    void set_ambient(vec3 ambient);
    vec3 get_ambient() const;

    void add(point_light& pl);
    void remove(point_light& pl);
    void clear_point_lights();
    const std::vector<point_light*>& get_point_lights() const;

    void add(spotlight& sp);
    void remove(spotlight& sp);
    void clear_spotlights();
    const std::vector<spotlight*>& get_spotlights() const;

    void add(directional_light& dl);
    void remove(directional_light& dl);
    void clear_directional_lights();
    const std::vector<directional_light*>& get_directional_lights() const;

    void auto_shadow_maps(
        unsigned directional_res = 2048,
        vec3 directional_volume = vec3(10, 10, 100),
        vec2 directional_bias = vec2(0.01, 0.05),
        unsigned cascades = 4,
        unsigned point_res = 512,
        float point_near = 0.01f,
        vec2 point_bias = vec2(0.006, 0.02)
    );

    const directional_shadow_map* get_shadow_map(
        const directional_light* dl) const;
    const point_shadow_map* get_shadow_map(
        const point_light* pl) const;

    void track_shadow_maps(const std::vector<camera*>& cameras);

    void add(sh_grid& sh);
    void remove(sh_grid& sh);
    void clear_sh_grids();
    const std::vector<sh_grid*>& get_sh_grids() const;
    sh_grid* get_sh_grid(vec3 pos, int* index = nullptr) const;
    sh_grid* get_largest_sh_grid(int* index = nullptr) const;

    void clear();

    void play(
        const std::string& name,
        bool loop = false,
        bool use_fallback = false
    );
    void update(time_ticks dt, bool force_update = false);
    bool is_playing() const;
    void set_animation_time(time_ticks dt);
    time_ticks get_total_ticks() const;

private:
    friend class scene_stage;

    environment_map* envmap = nullptr;
    vec3 ambient = vec3(0);

    std::vector<camera*> cameras;
    std::vector<animated_node*> control_nodes;
    std::vector<mesh_object*> objects;
    std::vector<point_light*> point_lights;
    std::vector<spotlight*> spotlights;
    std::vector<directional_light*> directional_lights;
    std::vector<sh_grid*> sh_grids;

    std::unordered_map<
        const directional_light*, directional_shadow_map
    > directional_shadow_maps;
    std::unordered_map<const point_light*, point_shadow_map> point_shadow_maps;

    time_ticks total_ticks;
};

std::vector<uint32_t> get_viewport_reorder_mask(
    const std::set<int>& active_indices,
    size_t viewport_count
);

}

#endif
