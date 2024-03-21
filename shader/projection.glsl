#ifndef PROJECTION_GLSL
#define PROJECTION_GLSL
#include "math.glsl"

vec3 unproject_position(float linear_depth, vec2 uv, vec4 projection_info)
{
    return vec3((0.5f-uv) * projection_info.zw * linear_depth, linear_depth);
}

vec2 project_position(vec3 pos, vec4 projection_info)
{
    return 0.5f - pos.xy / (projection_info.zw * pos.z);
}

float linearize_depth(float depth, vec4 proj_info)
{
    float tmp = (1-depth) * proj_info.x + proj_info.y;
    // If proj_info.x > 0, the projection is orthographic. If less than 0,
    // it is perspective.
    return proj_info.x > 0 ? tmp : 1.0/tmp;
}

vec4 linearize_depth(vec4 depth, vec4 proj_info)
{
    vec4 tmp = (1-depth) * proj_info.x + proj_info.y;
    return proj_info.x > 0 ? tmp : 1.0/tmp;
}

float hyperbolic_depth(float linear_depth, vec4 proj_info)
{
    if(proj_info.x < 0)
        linear_depth = 1.0 / linear_depth;
    return 1.0f - (linear_depth-proj_info.y)/proj_info.x;
}

#endif
