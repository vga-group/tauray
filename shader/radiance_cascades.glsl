#ifndef RADIANCE_CASCADES_GLSL
#define RADIANCE_CASCADES_GLSL
#include "math.glsl"

#ifdef RADIANCE_CASCADES_SET
layout(set=RADIANCE_CASCADES_SET, binding = 0) uniform sampler2DArray radiance_cascades[];
layout(set=RADIANCE_CASCADES_SET, binding = 1) uniform radiance_cascade_metadata_buffer
{
    vec4 aabb_min;
    vec4 aabb_max;
    // size.x = c0 width (x)
    // size.y = c0 height (y)
    // size.z = c0 depth (z)
    // size.w = cascade count
    ivec4 size;
    int c0_angular_resolution;
} radiance_cascade_metadata;

#include "random_sampler.glsl"

vec3 query_radiance_cascades(vec3 origin, vec3 dir)
{
    vec3 aabb_min = radiance_cascade_metadata.aabb_min.xyz;
    vec3 aabb_max = radiance_cascade_metadata.aabb_max.xyz;
    // If outside cascade volume, continue ray until it is inside.
    if(any(lessThan(origin, aabb_min)) || any(greaterThan(origin, aabb_max)))
    {
        float t = intersect_aabb(aabb_min, aabb_max, origin, dir);
        if(t < 0)
        {
            // Miss: outside volume.
            // TODO: Maybe could fall back to top cascade?
            return vec3(1);
        }
        origin = origin + dir * t;
    }

    // 0-1 inside cascade volume
    vec3 fcoord = (origin - aabb_min) / (aabb_max - aabb_min);

    ivec3 cascade_size = radiance_cascade_metadata.size.xyz;
    ivec2 probe_resolution = ivec2(radiance_cascade_metadata.c0_angular_resolution);
    // TODO: Maybe make this a specialization constant?
    int cascade_count = radiance_cascade_metadata.size.w;
    ivec3 cascade_coord = clamp(ivec3(
        fcoord * cascade_size
    ), ivec3(0), ivec3(cascade_size-1));

    vec4 sum = vec4(0,0,0,1);

    vec2 uv = concentric_octahedral_mapping_inverse(dir);

    ivec2 p = clamp(ivec2(uv * probe_resolution), ivec2(0), probe_resolution-1);

    ivec3 tex_coord = cascade_coord * ivec3(probe_resolution, 1) + ivec3(p, 0);
    vec4 col = texelFetch(radiance_cascades[0], tex_coord, 0);
    sum.rgb += col.rgb * sum.a;
    sum.a *= 1.0f-col.a;

    for(int cascade = 1; sum.a > 0 && cascade < cascade_count; ++cascade)
    {
        probe_resolution *= 2;
        cascade_coord /= 2;

        ivec2 p = clamp(ivec2(uv * probe_resolution), ivec2(0), probe_resolution-1);

        tex_coord = cascade_coord * ivec3(probe_resolution, 1) + ivec3(p, 0);
        vec4 col = texelFetch(radiance_cascades[cascade], tex_coord, 0);
        sum.rgb += col.rgb * sum.a;
        sum.a *= 1.0f-col.a;
    }
    return sum.rgb;
}
#endif

#endif
