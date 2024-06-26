// This is for rendering the shadow maps, not using them. See
// shadow_mapping.glsl for the use code.
#ifndef SHADOW_MAP_COMMON_GLSL
#define SHADOW_MAP_COMMON_GLSL

#include "scene.glsl"

layout(binding = 0, set = 0) buffer shadow_camera_data_buffer
{
    mat4 view_proj[];
} shadow_camera;

layout(push_constant) uniform push_constant_buffer
{
    uint instance_id;
    float alpha_clip;
    uint camera_index;
} control;

#endif
