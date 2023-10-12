#ifndef LIGHT_BVH_GLSL
#define LIGHT_BVH_GLSL

#include "math.glsl"

struct light_bvh_node
{
    uint bounds[3]; // Quantized to BVH-global AABB
    uint primary_dir; // Octahedral encoding
    float power; // Negative marks double-sided.
    float cos_normal_variation_angle;
    float cos_visibility_angle;
    // Top bit indicates which one. 1 == light index (leaf), 0 == child.
    uint child_or_light_index;
};

#ifdef LIGHT_BVH_BUFFER_BINDING
layout(binding = LIGHT_BVH_BUFFER_BINDING, set = 0) readonly buffer light_bvh_buffer
{
    vec4 min_bound;
    vec4 max_bound;
    light_bvh_node nodes[];
} light_bvh;
#endif

#ifdef LIGHT_BIT_TRAIL_BUFFER_BINDING
layout(binding = LIGHT_BIT_TRAIL_BUFFER_BINDING, set = 0, scalar) readonly buffer light_bit_trail_buffer
{
    uint bitmasks[];
} light_bit_trail;
#endif

float cos_sub_clamped(
    float sin_theta_a, float cos_theta_a,
    float sin_theta_b, float cos_theta_b
){
    if(cos_theta_a > cos_theta_b)
        return 1.0f;
    return cos_theta_a * cos_theta_b + sin_theta_a * sin_theta_b;
}

float sin_sub_clamped(
    float sin_theta_a, float cos_theta_a,
    float sin_theta_b, float cos_theta_b
){
    if(cos_theta_a > cos_theta_b)
        return 0.0f;
    return sin_theta_a * cos_theta_b + cos_theta_a * sin_theta_b;
}

float light_bvh_node_importance(
    vec3 global_min_bound,
    vec3 global_max_bound,
    light_bvh_node node,
    vec3 pos,
    vec3 normal
){
    vec3 min_bound;
    vec3 max_bound;
    for(int i = 0; i < 3; ++i)
    {
        uint bounds = node.bounds[i];
        uint qmin_bound = bounds & 0xFFFFu;
        uint qmax_bound = bounds >> 16u;

        min_bound[i] = float(qmin_bound)/65535.0f*(global_max_bound[i] - global_min_bound[i]) + global_min_bound[i];
        max_bound[i] = float(qmax_bound)/65535.0f*(global_max_bound[i] - global_min_bound[i]) + global_min_bound[i];
    }
    vec3 primary_dir = octahedral_decode(unpackSnorm2x16(node.primary_dir));

    // Compute clamped squared distance to reference point
    vec3 center = (min_bound + max_bound) * 0.5f;
    float radius = distance(max_bound, min_bound) * 0.5f;
    vec3 delta = pos - center;
    float d2 = dot(delta, delta);
    float clamped_d2 = max(d2, radius);

    // Compute sine and cosine of angle to vector w, theta_w
    vec3 wi = normalize(delta);
    float cos_theta_w = dot(primary_dir, wi);
    if(node.power < 0) cos_theta_w = abs(cos_theta_w);

    float sin_theta_w = sqrt(max(1.0f - cos_theta_w * cos_theta_w, 0.0f));

    // Compute cos theta_b for reference point
    float sin2_theta_b = min(radius * radius / d2, 1.0f);
    float cos_theta_b = d2 < radius * radius ? -1 : sqrt(1 - sin2_theta_b);
    float sin_theta_b = d2 < radius * radius ? 0 : sqrt(sin2_theta_b);

    // Compute cos theta' and test against cos theta_e
    float cos_theta_o = node.cos_normal_variation_angle;
    float sin_theta_o = sqrt(max(1.0f - cos_theta_o * cos_theta_o, 0.0f));

    float cos_theta_x = cos_sub_clamped(sin_theta_w, cos_theta_w, sin_theta_o, cos_theta_o);
    float sin_theta_x = sin_sub_clamped(sin_theta_w, cos_theta_w, sin_theta_o, cos_theta_o);
    float cos_theta_p = cos_sub_clamped(sin_theta_x, cos_theta_x, sin_theta_b, cos_theta_b);
    if(cos_theta_p <= node.cos_visibility_angle)
        return 0.0f;

    // eturn final importance at reference point
    float importance = abs(node.power) * cos_theta_p / clamped_d2;
    float cos_theta_i = abs(dot(wi, normal));
    float sin_theta_i = sqrt(max(1.0f - cos_theta_i * cos_theta_i, 0.0f));
    importance *= cos_sub_clamped(sin_theta_i, cos_theta_i, sin_theta_b, cos_theta_b);

    return importance;
}

#ifdef LIGHT_BVH_BUFFER_BINDING
void random_sample_tri_light(vec3 pos, vec3 normal, float u, out float pdf, out uint selected_index)
{
    uint node_index = 0;
    pdf = 1.0f;
    light_bvh_node node = light_bvh.nodes[node_index];
    vec3 global_min_bound = light_bvh.min_bound.xyz;
    vec3 global_max_bound = light_bvh.max_bound.xyz;

    // Bounded to 32 because of bit trail length -- we would be rendering
    // incorrectly anyway, if we went beyond that. I'm also kind of scared of
    // using unbounded while(true) on the GPU like PBRT is doing...
    for(int i = 0; i < 32; ++i)
    {
        bool is_leaf = (node.child_or_light_index & 0x80000000u) != 0;

        if(is_leaf)
        {
            selected_index = node.child_or_light_index & 0x7FFFFFFFu;
            return;
        }
        else
        {
            uint a_index = node_index+1;
            uint b_index = node.child_or_light_index;

            light_bvh_node a = light_bvh.nodes[a_index];
            light_bvh_node b = light_bvh.nodes[b_index];

            float a_importance = light_bvh_node_importance(global_min_bound, global_max_bound, a, pos, normal);
            float b_importance = light_bvh_node_importance(global_min_bound, global_max_bound, b, pos, normal);
            float sum_importance = a_importance + b_importance;
            float a_prob = sum_importance == 0.0f ? 0.5f : a_importance / sum_importance;
            float b_prob = 1.0f - a_prob;
            if(u < a_prob)
            {
                u = u/a_prob;
                node = a;
                node_index = a_index;
                pdf *= a_prob;
            }
            else
            {
                u = (u-a_prob)/(1.0f-a_prob);
                node = b;
                node_index = b_index;
                pdf *= b_prob;
            }
        }
    }

    // Failed to traverse correctly, so just use something that's probably
    // not going to explode.
    selected_index = 0;
    pdf = 0.0f;
    return;
}

#ifdef LIGHT_BIT_TRAIL_BUFFER_BINDING
float tri_light_sample_pdf(vec3 pos, vec3 normal, uint id)
{
    float pdf = 1.0f;

    uint bit_trail = light_bit_trail.bitmasks[id];

    uint node_index = 0;
    light_bvh_node node = light_bvh.nodes[node_index];
    vec3 global_min_bound = light_bvh.min_bound.xyz;
    vec3 global_max_bound = light_bvh.max_bound.xyz;

    for(int i = 0; i < 32; ++i)
    {
        bool is_leaf = (node.child_or_light_index & 0x80000000u) != 0;

        if(is_leaf)
        {
            return pdf;
        }
        else
        {
            uint a_index = node_index+1;
            uint b_index = node.child_or_light_index;

            light_bvh_node a = light_bvh.nodes[a_index];
            light_bvh_node b = light_bvh.nodes[b_index];

            float a_importance = light_bvh_node_importance(global_min_bound, global_max_bound, a, pos, normal);
            float b_importance = light_bvh_node_importance(global_min_bound, global_max_bound, b, pos, normal);
            float sum_importance = a_importance + b_importance;
            float a_prob = sum_importance == 0.0f ? 0.5f : a_importance / sum_importance;
            float b_prob = 1.0f - a_prob;

            bool first = ((bit_trail >> i)&1) == 0;
            if(first)
            {
                node = a;
                node_index = a_index;
                pdf *= a_prob;
            }
            else
            {
                node = b;
                node_index = b_index;
                pdf *= b_prob;
            }
        }
    }

    return pdf;
}
#endif
#endif

#endif
