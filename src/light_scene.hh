#ifndef TAURAY_LIGHT_SCENE_HH
#define TAURAY_LIGHT_SCENE_HH
#include "device.hh"
#include "light.hh"
#include "acceleration_structure.hh"
#include "shadow_map.hh"
#include "timer.hh"

namespace tr
{

class environment_map;
class sh_grid;
class camera;

class light_scene
{
public:
    light_scene(device_mask dev, size_t max_capacity = 1024);
    light_scene(const light_scene& s) = delete;
    light_scene(light_scene&& s) = delete;
    ~light_scene();

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

protected:
    template<typename F>
    void visit_animated(F&& f) const
    {
        for(point_light* l: point_lights) f(l);
        for(spotlight* l: spotlights) f(l);
        for(directional_light* l: directional_lights) f(l);
    }

    template<typename F>
    void visit_animated(F&& f)
    {
        for(point_light* l: point_lights) f(l);
        for(spotlight* l: spotlights) f(l);
        for(directional_light* l: directional_lights) f(l);
    }

    size_t get_aabbs(vk::AabbPositionsKHR* aabb);

    size_t get_max_capacity() const;

    // TODO: Should probably not be device-specific as it doesn't actually do
    // any device-specific things.
    void update_acceleration_structures(
        device_id id,
        uint32_t frame_index,
        bool& need_scene_reset,
        bool& command_buffers_outdated
    );

    void record_acceleration_structure_build(
        vk::CommandBuffer& cb,
        device_id id,
        uint32_t frame_index,
        bool update_only
    );

    void add_acceleration_structure_instances(
        vk::AccelerationStructureInstanceKHR* instances,
        device_id id,
        uint32_t frame_index,
        size_t& instance_index,
        size_t capacity
    ) const;

    void invalidate_acceleration_structures();

private:
    environment_map* envmap = nullptr;
    vec3 ambient = vec3(0);

    std::vector<point_light*> point_lights;
    std::vector<spotlight*> spotlights;
    std::vector<directional_light*> directional_lights;
    std::unordered_map<
        const directional_light*, directional_shadow_map
    > directional_shadow_maps;
    std::unordered_map<const point_light*, point_shadow_map> point_shadow_maps;
    std::vector<sh_grid*> sh_grids;

    size_t max_capacity;

    std::optional<bottom_level_acceleration_structure> blas;
    gpu_buffer aabb_buffer;
    timer blas_update_timer;

    struct as_update_data
    {
        bool scene_reset_needed = true;

        struct per_frame_data
        {
            bool command_buffers_outdated = true;
            unsigned aabb_count = 0;
        };
        per_frame_data per_frame[MAX_FRAMES_IN_FLIGHT];
    };
    per_device<as_update_data> as_update;
};

}

#endif
