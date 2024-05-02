#ifndef TR_SH_GRID_HH
#define TR_SH_GRID_HH
#include "texture.hh"
#include "transformable.hh"
#include <vector>

namespace tr
{

// A 3D grid of spherical harmonics probes. Coefficients are stacked vertically
// in the 3D texture, so lookup must clamp manually. Similar to the shadow_map
// classes, this is only a specification that becomes fulfilled by a renderer.
class sh_grid
{
public:
    sh_grid(
        uvec3 resolution = uvec3(1),
        int order = 2
    );

    texture create_target_texture(
        device_mask dev,
        int samples_per_probe
    );
    void get_target_sampling_info(
        device_mask dev,
        int& samples_per_probe,
        int& samples_per_invocation
    );
    texture create_texture(device_mask dev);
    size_t get_required_bytes() const;

    void set_resolution(uvec3 res);
    uvec3 get_resolution() const;

    // Radius is added to the actual volume size.
    void set_radius(float radius = 0.0f);
    float get_radius() const;

    void set_order(int order);
    int get_order() const;

    int get_coef_count() const;
    static int get_coef_count(int order);

    // Negative: out of influence. Zero: fully in influence. Positive: outside,
    // but within radius.
    float point_distance(transformable& self, vec3 p) const;
    float calc_density(transformable& self) const;
    float calc_volume(transformable& self) const;

private:
    float radius;
    int order;
    uvec3 resolution;
};

}

#endif

