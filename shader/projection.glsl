#ifndef PROJECTION_GLSL
#define PROJECTION_GLSL
#include "math.glsl"

vec3 unproject_position(float linear_depth, vec2 uv, vec4 projection_info, vec2 pan)
{
    return vec3((0.5f-uv-0.5f*pan) * projection_info.zw * linear_depth, linear_depth);
}

vec2 project_position(vec3 pos, vec4 projection_info, vec2 pan)
{
    return 0.5f - pos.xy / (projection_info.zw * pos.z) - 0.5f * pan;
}

// Note that this assumes that depth is -1 to 1, depth buffer has them at 0 to
// 1. You can just do the usual *2-1 to it.
float linearize_depth(float depth, vec4 proj_info)
{
    return -2.0f * proj_info.x / (depth + proj_info.y);
}

vec4 linearize_depth(vec4 depth, vec4 proj_info)
{
    return -2.0f * proj_info.x / (depth + proj_info.y);
}

float hyperbolic_depth(float linear_depth, vec4 proj_info)
{
    return (-2.0f * proj_info.x - proj_info.y * linear_depth) / linear_depth;
}

#endif
