#ifndef WHITTED_GLSL
#define WHITTED_GLSL
#include "rt.glsl"

struct hit_payload
{
    vec4 color;
    int depth;
    // These are used for preventing self-intersections with refractions.
    int self_instance_id, self_primitive_id;
};

#ifdef USE_PUSH_CONSTANTS
layout(push_constant) uniform push_constant_buffer
{
    uint directional_light_count;
    uint point_light_count;
    uint max_depth;
    vec4 ambient;
    float min_ray_dist;
} control;
#endif

#endif
