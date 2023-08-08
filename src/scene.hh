#ifndef TAURAY_SCENE_HH
#define TAURAY_SCENE_HH
#include "light_scene.hh"
#include "mesh_scene.hh"
#include "timer.hh"
#include "sampler_table.hh"
#include "descriptor_state.hh"
#include "acceleration_structure.hh"

namespace tr
{

class camera;
class environment_map;
class sh_grid;
class basic_pipeline;
class scene: public light_scene, public mesh_scene
{
public:
    scene();
    scene(const scene& s) = delete;
    scene(scene&& s) = delete;

    void set_camera(camera* cam);
    camera* get_camera(unsigned index = 0) const;

    using light_scene::add;
    using light_scene::remove;

    using mesh_scene::add;
    using mesh_scene::remove;

    void add(camera& c);
    void remove(camera& c);
    void clear_cameras();
    const std::vector<camera*>& get_cameras() const;
    void reorder_cameras_by_active(const std::set<int>& active_indices);
    void set_camera_jitter(const std::vector<vec2>& jitter);

    void add_control_node(animated_node& o);
    void remove_control_node(animated_node& o);
    void clear_control_nodes();

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

    std::vector<camera*> cameras;
    std::vector<animated_node*> control_nodes;
    time_ticks total_ticks;
};

std::vector<uint32_t> get_viewport_reorder_mask(
    const std::set<int>& active_indices,
    size_t viewport_count
);

}

#endif
