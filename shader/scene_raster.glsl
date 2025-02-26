#ifndef SCENE_RASTER_GLSL
#define SCENE_RASTER_GLSL
#include "scene.glsl"

#ifndef SCENE_RASTER_SET
#define SCENE_RASTER_SET 2
#endif

layout(binding = 0, set = SCENE_RASTER_SET) uniform sampler3D sh_grid_data[];

struct sh_grid
{
    mat4 pos_from_world;
    mat4 normal_from_world;
    vec3 grid_clamp;
    float pad0;
    vec3 grid_resolution;
    float pad1;
};

layout(binding = 1, set = SCENE_RASTER_SET, scalar) buffer sh_grid_buffer
{
    sh_grid grids[];
} sh_grids;

struct shadow_map
{
    int type;
    float min_bias;
    float max_bias;
    int cascade_index;
    vec4 rect;
    vec4 projection_info;
    vec4 range_radius;
    mat4 world_to_shadow;
};

struct shadow_map_cascade
{
    vec4 offset_scale;
    vec4 rect;
};

struct shadow_mapping_parameters
{
    vec2 shadow_map_atlas_pixel_margin;
    float pcss_minimum_radius;
    float noise_scale;
    int pcf_samples;
    int omni_pcf_samples;
    int pcss_samples;
    int pad[1];
};

layout(binding = 2, set = SCENE_RASTER_SET, std430) readonly buffer shadow_map_buffer
{
    shadow_map maps[];
} shadow_maps;

layout(binding = 3, set = SCENE_RASTER_SET, std430) readonly buffer shadow_map_cascade_buffer
{
    shadow_map_cascade cascades[];
} shadow_map_cascades;

layout(binding = 4, set = SCENE_RASTER_SET) uniform sampler2D shadow_map_atlas;
layout(binding = 5, set = SCENE_RASTER_SET) uniform sampler2DShadow shadow_map_atlas_test;
layout(binding = 6, set = SCENE_RASTER_SET) uniform sampler2D pcf_noise_vector_2d;
layout(binding = 7, set = SCENE_RASTER_SET) uniform sampler2D pcf_noise_vector_3d;
layout(binding = 8, set = SCENE_RASTER_SET) uniform sampler2D brdf_integration;


#endif
