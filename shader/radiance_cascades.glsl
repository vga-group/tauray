#ifndef RADIANCE_CASCADES_GLSL
#define RADIANCE_CASCADES_GLSL
#include "math.glsl"
#include "color.glsl"

ivec3 get_cascade_layout(ivec3 cascade_size, int probe_resolution, ivec3 probe_coord, ivec2 probe_texel)
{
    // Layouts the same direction in adjacent probes in adjacent pixels.
    //return probe_coord + ivec3(probe_texel * cascade_size.xy, 0);

    // Layouts the all directions of a probe in adjacent pixels.
    return ivec3(probe_texel + probe_coord.xy * probe_resolution, probe_coord.z);
}

void get_cascade_layout_inverse(ivec3 p, ivec3 cascade_size, int probe_resolution, out ivec3 probe_coord, out ivec2 probe_texel)
{
    // Layouts the same direction in adjacent probes in adjacent pixels.
    //probe_coord = ivec3(p.xy % cascade_size.xy, p.z);
    //probe_texel = p.xy / cascade_size.xy;

    // Layouts the all directions of a probe in adjacent pixels.
    probe_coord = ivec3(p.xy / probe_resolution, p.z);
    probe_texel = p.xy % probe_resolution;
}

#ifdef RADIANCE_CASCADES_SET
layout(set=RADIANCE_CASCADES_SET, binding = 0) uniform sampler2DArray radiance_cascades[];
layout(set=RADIANCE_CASCADES_SET, binding = 1) uniform sampler2DArray radiance_cascades_visibility[];
layout(set=RADIANCE_CASCADES_SET, binding = 2) uniform radiance_cascade_metadata_buffer
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

//#define RC_C0_ANGULAR_RESOLUTION
//#define RC_C0_SPATIAL_RESOLUTION
//#define RC_CASCADE_COUNT

#include "random_sampler.glsl"

ivec2 octahedral_wrap(ivec2 p, ivec2 size)
{
    if(p.x < 0 || p.x >= size.x)
    {
        p.x = -1-p.x;
        if(p.x < 0) p.x += 2 * size.x;
        p.y = size.y-1-p.y;
    }

    if(p.y < 0 || p.y >= size.y)
    {
        p.y = -1-p.y;
        if(p.y < 0) p.y += 2 * size.y;
        p.x = size.x-1-p.x;
    }

    return p;
}

vec3 query_radiance_cascades(vec3 origin, vec3 dir, uvec4 seed)
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
    int probe_resolution = radiance_cascade_metadata.c0_angular_resolution;

    vec4 sum = vec4(0,0,0,1);

    vec2 uv = concentric_octahedral_mapping_inverse(dir);

    ivec2 p = octahedral_wrap(ivec2(floor(uv * probe_resolution)), ivec2(probe_resolution));
    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));
    ivec3 tex_coord = get_cascade_layout(cascade_size, probe_resolution, ivec3(cascade_coord), p);
    vec2 col = texelFetch(radiance_cascades[0], tex_coord, 0).rg;
    sum.rgb += col.rrr * col.g * sum.a;
    sum.a *= 1.0f-col.g;

    for(int cascade = 1; sum.a > 0 && cascade < RC_CASCADE_COUNT; ++cascade)
    {
        probe_resolution *= 2;
        cascade_size /= 2;

        ivec2 p = octahedral_wrap(ivec2(floor(uv * probe_resolution)), ivec2(probe_resolution));
        vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));
        ivec3 tex_coord = get_cascade_layout(cascade_size, probe_resolution, ivec3(cascade_coord), p);
        vec2 col = texelFetch(radiance_cascades[cascade], tex_coord, 0).rg;
        sum.rgb += col.rrr * col.g * sum.a;
        sum.a *= 1.0f-col.g;
    }
    return sum.rgb;
}

float eval_diffuse_radiance_cascades(vec3 origin, vec3 normal)
{
    vec3 aabb_min = radiance_cascade_metadata.aabb_min.xyz;
    vec3 aabb_max = radiance_cascade_metadata.aabb_max.xyz;

    // 0-1 inside cascade volume
    vec3 fcoord = (origin - aabb_min) / (aabb_max - aabb_min);

    ivec3 cascade_size = radiance_cascade_metadata.size.xyz;
    int probe_resolution = radiance_cascade_metadata.c0_angular_resolution;

    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));

    //mat3 ltc_irradiance_transform = create_tangent_space(normal);

    float sum_contrib = 0;

    float inv_probe_resolution = 1.0f / probe_resolution;
    float len = 4.0f * inv_probe_resolution * inv_probe_resolution;

    [[unroll]] for(int x = 0; x < RC_C0_ANGULAR_RESOLUTION; ++x)
    [[unroll]] for(int y = 0; y < RC_C0_ANGULAR_RESOLUTION; ++y)
    {
        ivec2 p = ivec2(x, y);
        ivec3 tex_coord = get_cascade_layout(cascade_size, RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), p);
        float value = texelFetch(radiance_cascades[0], tex_coord, 0).r;

        vec3 center = concentric_octahedral_mapping((vec2(p) + 0.5f) * inv_probe_resolution);
        float cdn = dot(center, normal);
        sum_contrib += value * max(len + cdn, 0.0f);
    }

    sum_contrib *= 4.0f / (4.0f + probe_resolution * probe_resolution);

    return sum_contrib;
}

// Only valid inside cascade volume!
vec3 sample_radiance_cascades(uint seed, vec3 origin, vec3 normal, vec3 albedo, out float pdf)
{
    vec3 aabb_min = radiance_cascade_metadata.aabb_min.xyz;
    vec3 aabb_max = radiance_cascade_metadata.aabb_max.xyz;

    // 0-1 inside cascade volume
    vec3 fcoord = (origin - aabb_min) / (aabb_max - aabb_min);
    int cascade_size = RC_C0_SPATIAL_RESOLUTION;
    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));

    ivec2 selected_cell = ivec2(0);
    float selected_weight = 0.0f;
    float sum_weight = 0.0f;

    float inv_probe_resolution = 1.0f / RC_C0_ANGULAR_RESOLUTION;
    float len = 4.0f * inv_probe_resolution * inv_probe_resolution;

    for(int x = 0; x < RC_C0_ANGULAR_RESOLUTION; ++x)
    for(int y = 0; y < RC_C0_ANGULAR_RESOLUTION; ++y)
    {
        ivec2 p = ivec2(x, y);
        ivec3 tex_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), p);
        float value = texelFetch(radiance_cascades[0], tex_coord, 0).r;

        vec3 center = concentric_octahedral_mapping((vec2(p) + 0.5f) * inv_probe_resolution);
        float cdn = dot(center, normal);
        float weight = value * max(len + cdn, 0.0f);
        float u = generate_single_uniform_random_fast(seed);

        weight += 1e-16f;
        sum_weight += weight;
        if(u*sum_weight < weight)
        {
            selected_cell = p;
            selected_weight = weight;
        }
    }
    ivec3 sel_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), selected_cell);
    float total_visibility = 1.0 - texelFetch(radiance_cascades_visibility[0], sel_coord, 0).r;

    pdf = selected_weight / sum_weight;

    for(int cascade = 1; cascade < RC_CASCADE_COUNT; ++cascade)
    { // Drill deeper
        cascade_size /= 2;
        inv_probe_resolution *= 0.5f;
        len *= 0.25f;
        cascade_coord = clamp(cascade_coord * 0.5f, vec3(0.5), vec3(cascade_size-0.5));

        ivec2 base_cell = selected_cell * 2;
        selected_weight = 0.0f;
        sum_weight = 0.0f;
        for(int x = 0; x < 2; ++x)
        for(int y = 0; y < 2; ++y)
        {
            ivec2 p = base_cell.xy + ivec2(x, y);
            ivec3 tex_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION<<cascade, ivec3(cascade_coord), p);

            float value = texelFetch(radiance_cascades[cascade], tex_coord, 0).r;
            vec3 center = concentric_octahedral_mapping((vec2(p) + 0.5f) * inv_probe_resolution);
            float cdn = dot(center, normal);
            float weight = rgb_to_luminance(albedo * value) * max(len + cdn, 0.0f);
            weight += 1e-16f;

            float u = generate_single_uniform_random_fast(seed);

            sum_weight += weight;
            if(u*sum_weight < weight)
            {
                selected_cell = p;
                selected_weight = weight;
            }
        }
        ivec3 sel_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION<<cascade, ivec3(cascade_coord), selected_cell);
        total_visibility *= 1.0 - texelFetch(radiance_cascades_visibility[cascade], sel_coord, 0).r;
        pdf *= selected_weight / sum_weight;
    }

    // Sample from inside the selected cell.
    vec2 random_offset = vec2(
        generate_single_uniform_random_fast(seed),
        generate_single_uniform_random_fast(seed)
    );

    vec2 uv = (vec2(selected_cell) + random_offset) * inv_probe_resolution;

    const int probe_resolution = RC_C0_ANGULAR_RESOLUTION<<(RC_CASCADE_COUNT-1);
    pdf *= (probe_resolution * probe_resolution) / (4 * M_PI);
    vec3 dir = concentric_octahedral_mapping(uv);
    return dir;
}

float radiance_cascades_pdf(vec3 origin, vec3 normal, vec3 albedo, vec3 dir)
{
    vec3 aabb_min = radiance_cascade_metadata.aabb_min.xyz;
    vec3 aabb_max = radiance_cascade_metadata.aabb_max.xyz;

    // 0-1 inside cascade volume
    vec3 fcoord = (origin - aabb_min) / (aabb_max - aabb_min);

    ivec3 cascade_size = radiance_cascade_metadata.size.xyz;
    int probe_resolution = radiance_cascade_metadata.c0_angular_resolution;

    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));
    vec2 tex_coord = concentric_octahedral_mapping_inverse(dir);

    float selected_weight = 0.0f;
    float sum_weight = 0.0f;

    float inv_probe_resolution = 1.0f / probe_resolution;
    float len = 4.0f * inv_probe_resolution * inv_probe_resolution;

    ivec2 itex_coord = ivec2(tex_coord * probe_resolution);

    ivec3 sel_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), itex_coord);
    float total_visibility = 1.0 - texelFetch(radiance_cascades_visibility[0], sel_coord, 0).r;

    for(int x = 0; x < radiance_cascade_metadata.c0_angular_resolution; ++x)
    for(int y = 0; y < radiance_cascade_metadata.c0_angular_resolution; ++y)
    {
        ivec2 p = ivec2(x, y);
        ivec3 tex_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), p);
        float value = texelFetch(radiance_cascades[0], tex_coord, 0).r;

        vec3 center = concentric_octahedral_mapping((vec2(p) + 0.5f) * inv_probe_resolution);
        float cdn = dot(center, normal);
        float weight = rgb_to_luminance(albedo * value) * max(len + cdn, 0.0f);

        weight += 1e-16f;

        sum_weight += weight;
        if(x == itex_coord.x && y == itex_coord.y)
            selected_weight = weight;
    }

    float pdf = selected_weight / sum_weight;

    for(int cascade = 1; cascade < RC_CASCADE_COUNT; ++cascade)
    { // Drill deeper
        cascade_size /= 2;
        probe_resolution *= 2;
        inv_probe_resolution *= 0.5f;
        len *= 0.25f;
        cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));
        ivec2 base_cell = itex_coord * 2;

        itex_coord = ivec2(tex_coord * probe_resolution);
        selected_weight = 0.0f;
        sum_weight = 0.0f;
        for(int x = 0; x < 2; ++x)
        for(int y = 0; y < 2; ++y)
        {
            ivec2 p = base_cell.xy + ivec2(x, y);
            ivec3 tex_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION<<cascade, ivec3(cascade_coord), p);
            float value = texelFetch(radiance_cascades[cascade], tex_coord, 0).r;

            vec3 center = concentric_octahedral_mapping((vec2(p) + 0.5f) * inv_probe_resolution);
            float cdn = dot(center, normal);
            float weight = rgb_to_luminance(albedo * value) * max(len + cdn, 0.0f);
            weight += 1e-16f;

            sum_weight += weight;
            if(p.x == itex_coord.x && p.y == itex_coord.y)
                selected_weight = weight;
        }
        ivec3 sel_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION<<cascade, ivec3(cascade_coord), itex_coord);
        total_visibility *= 1.0 - texelFetch(radiance_cascades_visibility[cascade], sel_coord, 0).r;
        pdf *= selected_weight / sum_weight;
    }

    // Sample from inside the selected cell.
    pdf *= (probe_resolution * probe_resolution) / (4 * M_PI);
    return pdf;
}
#endif

#endif
