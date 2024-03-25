#ifndef TR_SHADOW_MAP_HH
#define TR_SHADOW_MAP_HH
#include "math.hh"

namespace tr
{

class camera;
class transformable;
struct directional_shadow_map
{
    uvec2 resolution = uvec2(512);
    vec2 x_range = glm::vec2(-10.0f, 10.0f);
    vec2 y_range = glm::vec2(-10.0f, 10.0f);
    vec2 depth_range = glm::vec2(-100.0f, 100.0f);
    float min_bias = 0.01;
    float max_bias = 0.02;

    // vec2 is cascade offset in shadow map space. If you plan to call
    // track_camera(), you only need to resize cascades to the number of
    // cascades you want. 4-5 is a good number, if you don't know what to
    // pick.
    std::vector<vec2> cascades;

    void track_cameras(
        const mat4& light_transform,
        const std::vector<camera*>& cam,
        const std::vector<transformable*>& camera_transforms,
        bool conservative = true
    );
};

struct point_shadow_map
{
    uvec2 resolution = uvec2(512);
    float near = 0.01f;
    float min_bias = 0.006;
    float max_bias = 0.02;
};

struct shadow_map_filter
{
    // Shadow filtering options.
    int pcf_samples = 64; // 0 => bilinear interpolation
    int omni_pcf_samples = 16; // 0 => bilinear interpolation
    int pcss_samples = 32; // 0 => disable PCSS
    // The minimum radius prevents PCSS from degrading to bilinear filtering
    // near shadow caster.
    float pcss_minimum_radius = 0.0f;
};

struct gpu_shadow_mapping_parameters
{
    pvec2 shadow_map_atlas_pixel_margin;
    float pcss_minimum_radius;
    float noise_scale;
    int pcf_samples;
    int omni_pcf_samples;
    int pcss_samples;
    int pad[1];
};

class scene_stage;
gpu_shadow_mapping_parameters create_shadow_mapping_parameters(shadow_map_filter filter, scene_stage& ss);

}

#endif
