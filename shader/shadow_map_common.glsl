// This is for rendering the shadow maps, not using them. See
// shadow_mapping.glsl for the use code.
#ifndef SHADOW_MAP_COMMON_GLSL
#define SHADOW_MAP_COMMON_GLSL

#include "scene.glsl"

layout(binding = 1, set = 0) uniform camera_data_buffer
{
    mat4 view_proj;
} camera;

layout(push_constant) uniform push_constant_buffer
{
    uint instance_id;
    float alpha_clip;
} control;

#endif
