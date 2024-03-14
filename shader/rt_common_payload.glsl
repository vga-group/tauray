#ifndef RT_COMMON_PAYLOAD_GLSL
#define RT_COMMON_PAYLOAD_GLSL

struct hit_payload
{
    // Needed by anyhit alpha handling.
    uint random_seed;

    // Negative if the ray escaped the scene or otherwise died. Otherwise, it's
    // the index of the mesh instance, and primitive_id and barycentrics are
    // valid as well.
    int instance_id;
    // If instance_id is non-negative, this is the triangle index. Otherwise,
    // if primitive_id is non-negative, it is the light index. If it is negative,
    // that means that the ray escaped the scene and hit the environment map
    // instead.
    int primitive_id;

    // Barycentric coordinates to the triangle that was hit.
    vec2 barycentrics;
};

#ifdef PAYLOAD_IN
layout(location = 0) rayPayloadInEXT hit_payload payload;
layout(location = 1) rayPayloadInEXT float shadow_visibility;
#else
layout(location = 0) rayPayloadEXT hit_payload payload;
layout(location = 1) rayPayloadEXT float shadow_visibility;
#endif

#endif
