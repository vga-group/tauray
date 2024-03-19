#ifndef FORWARD_GLSL
#define FORWARD_GLSL
#define USE_SHADOW_MAPPING
#define SCENE_SET 0
#define SCENE_RASTER_SET 1
#include "scene_raster.glsl"

layout(push_constant) uniform push_constant_buffer
{
    uint instance_id;
    int pcf_samples;
    int omni_pcf_samples;
    int pcss_samples;
    int base_camera_index;
    float pcss_minimum_radius;
    float noise_scale;
    int pad[3];
    vec2 shadow_map_atlas_pixel_margin;
    vec3 ambient_color;
} control;

#endif
