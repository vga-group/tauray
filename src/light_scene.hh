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
    light_scene();
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
};

}

#endif
