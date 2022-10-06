#ifndef PROJECTION_GLSL
#define PROJECTION_GLSL
#include "math.glsl"

vec3 unproject_position(float linear_depth, vec2 uv, vec2 projection_info)
{
    return vec3((0.5f-uv) * projection_info * linear_depth, linear_depth);
}

vec2 project_position(vec3 pos, vec2 projection_info)
{
    return 0.5f - pos.xy / (projection_info * pos.z);
}

vec3 calculate_view_ray(vec2 uv, vec2 projection_info)
{
    return vec3((uv-0.5f) * projection_info, -1.0f);
}

vec2 horizon(vec3 o, vec3 d, float near, vec2 projection_info)
{
    if(d.z < 0) return 0.5f - d.xy / (projection_info * d.z);
    else
    {
        vec3 h = o + d * ((near - o.z)/(d.z));
        return 0.5f - h.xy / (projection_info * h.z);
    }
}

// Calculates t in project_position(o + t*d) == uv
float calculate_ray_length(vec3 o, vec3 d, vec2 uv, vec2 projection_info)
{
    vec2 v = (uv-0.5f) * projection_info;
    v.x = -v.x;
    return -dot(o.xy, v.yx)/dot(d.xy, v.yx);
}

vec2 projected_ray_direction(vec3 o, vec3 d, vec2 projection_info)
{
    return (o.xy*d.z - o.z*d.xy)/projection_info;
}

// Note that this assumes that depth is -1 to 1, depth buffer has them at 0 to
// 1. You can just do the usual *2-1 to it.
float linearize_depth(float depth, vec3 clip_info)
{
    return -2.0f * clip_info.x / (depth * clip_info.y + clip_info.z);
}

vec4 linearize_depth(vec4 depth, vec3 clip_info)
{
    return -2.0f * clip_info.x / (depth * clip_info.y + clip_info.z);
}

float hyperbolic_depth(float linear_depth, vec3 clip_info)
{
    return (-2.0f * clip_info.x - clip_info.z * linear_depth) /
           (linear_depth * clip_info.y);
}

#endif
