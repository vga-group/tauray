#ifndef RADIANCE_CASCADES_GLSL
#define RADIANCE_CASCADES_GLSL
#include "math.glsl"
#include "ltc.glsl"

#ifdef RADIANCE_CASCADES_SET
layout(set=RADIANCE_CASCADES_SET, binding = 0) uniform sampler3D radiance_cascades[];
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
    ivec2 probe_resolution = ivec2(radiance_cascade_metadata.c0_angular_resolution);
    // TODO: Maybe make this a specialization constant?
    int cascade_count = radiance_cascade_metadata.size.w;

    vec4 sum = vec4(0,0,0,1);

    vec2 uv = concentric_octahedral_mapping_inverse(dir);

    ivec2 p = octahedral_wrap(ivec2(floor(uv * probe_resolution)), probe_resolution);
    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), ivec3(cascade_size-0.5));
    vec3 tex_coord = cascade_coord + vec3(p * cascade_size.xy, 0);
    vec2 col = textureLod(radiance_cascades[0], tex_coord, 0).rg;
    sum.rgb += col.rrr * col.g * sum.a;
    sum.a *= 1.0f-col.g;

    for(int cascade = 1; sum.a > 0 && cascade < cascade_count; ++cascade)
    {
        probe_resolution *= 2;
        cascade_size /= 2;

        ivec2 p = octahedral_wrap(ivec2(floor(uv * probe_resolution)), probe_resolution);
        vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), ivec3(cascade_size-0.5));
        vec3 tex_coord = cascade_coord + ivec3(p * cascade_size.xy, 0);
        vec2 col = textureLod(radiance_cascades[cascade], tex_coord, 0).rg;
        sum.rgb += col.rrr * col.g * sum.a;
        sum.a *= 1.0f-col.g;
    }
    return sum.rgb;
}

// Only valid inside cascade volume!
vec3 sample_radiance_cascades(uint seed, vec3 origin, vec3 normal, out float pdf)
{
    vec3 aabb_min = radiance_cascade_metadata.aabb_min.xyz;
    vec3 aabb_max = radiance_cascade_metadata.aabb_max.xyz;

    // 0-1 inside cascade volume
    vec3 fcoord = (origin - aabb_min) / (aabb_max - aabb_min);

    ivec3 cascade_size = radiance_cascade_metadata.size.xyz;
    int probe_resolution = radiance_cascade_metadata.c0_angular_resolution;
    // TODO: Maybe make this a specialization constant?
    int cascade_count = radiance_cascade_metadata.size.w;

    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), ivec3(cascade_size-0.5));

    mat3 ltc_irradiance_transform = create_tangent_space(normal);

    ivec2 selected_cell = ivec2(0);
    float selected_weight = 0.0f;
    float selected_value = 0.0f;
    float selected_visibility = 1.0f;
    float sum_weight = 0.0f;

    float inv_probe_resolution = 1.0f / probe_resolution;

    for(int x = 0; x < radiance_cascade_metadata.c0_angular_resolution; ++x)
    for(int y = 0; y < radiance_cascade_metadata.c0_angular_resolution; ++y)
    {
        ivec2 p = ivec2(x, y);
        vec3 tex_coord = cascade_coord + vec3(p * cascade_size.xy, 0);

        vec2 col = textureLod(radiance_cascades[0], tex_coord, 0).rg;
        float value = col.r;
        float visibility = col.g;

        vec3 corners[4] =  vec3[4](
            concentric_octahedral_mapping((p + ivec2(0,0)) * inv_probe_resolution),
            concentric_octahedral_mapping((p + ivec2(0,1)) * inv_probe_resolution),
            concentric_octahedral_mapping((p + ivec2(1,1)) * inv_probe_resolution),
            concentric_octahedral_mapping((p + ivec2(1,0)) * inv_probe_resolution)
        );

        float weight = value * cosine_hemisphere_poly_light(ltc_irradiance_transform, vec3(0), corners, false);
        weight += 1e-16f;

        float u = generate_single_uniform_random_fast(seed);

        sum_weight += weight;
        if(u*sum_weight < weight)
        {
            selected_cell = p;
            selected_weight = weight;
            selected_value = value;
            selected_visibility = visibility;
        }
    }

    pdf = selected_weight / sum_weight;
    float total_visibility = selected_visibility;

    for(int cascade = 1; cascade < cascade_count && total_visibility > 0; ++cascade)
    { // Drill deeper
        cascade_size /= 2;
        probe_resolution *= 2;
        inv_probe_resolution *= 0.5f;
        cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), ivec3(cascade_size-0.5));

        ivec2 base_cell = selected_cell * 2;
        float base_value = selected_value;
        selected_weight = 0.0f;
        selected_value = 0.0f;
        selected_visibility = 0.0f;
        sum_weight = 0.0f;
        for(int x = base_cell.x; x < base_cell.x + 2; ++x)
        for(int y = base_cell.y; y < base_cell.y + 2; ++y)
        {
            ivec2 p = ivec2(x, y);
            vec3 tex_coord = cascade_coord + vec3(p * cascade_size.xy, 0);
            vec2 col = textureLod(radiance_cascades[cascade], tex_coord, 0).rg;
            float value = mix(base_value, col.r, total_visibility);
            float visibility = col.g;

            vec3 corners[4] =  vec3[4](
                concentric_octahedral_mapping((p + ivec2(0,0)) * inv_probe_resolution),
                concentric_octahedral_mapping((p + ivec2(0,1)) * inv_probe_resolution),
                concentric_octahedral_mapping((p + ivec2(1,1)) * inv_probe_resolution),
                concentric_octahedral_mapping((p + ivec2(1,0)) * inv_probe_resolution)
            );
            float weight = value * cosine_hemisphere_poly_light(ltc_irradiance_transform, vec3(0), corners, false);
            weight += 1e-16f;

            float u = generate_single_uniform_random_fast(seed);

            sum_weight += weight;
            if(u*sum_weight < weight)
            {
                selected_cell = p;
                selected_weight = weight;
                selected_value = value;
                selected_visibility = visibility;
            }
        }
        pdf *= selected_weight / sum_weight;
        total_visibility *= selected_visibility;
    }

    // Sample from inside the selected cell.
    vec2 random_offset = vec2(
        generate_single_uniform_random_fast(seed),
        generate_single_uniform_random_fast(seed)
    );

    vec2 uv = (vec2(selected_cell) + random_offset) * inv_probe_resolution;

    pdf *= (probe_resolution * probe_resolution) / (4 * M_PI);
    return concentric_octahedral_mapping(uv);
}

float radiance_cascades_pdf(vec3 origin, vec3 normal, vec3 dir)
{
    vec3 aabb_min = radiance_cascade_metadata.aabb_min.xyz;
    vec3 aabb_max = radiance_cascade_metadata.aabb_max.xyz;

    // 0-1 inside cascade volume
    vec3 fcoord = (origin - aabb_min) / (aabb_max - aabb_min);

    ivec3 cascade_size = radiance_cascade_metadata.size.xyz;
    int probe_resolution = radiance_cascade_metadata.c0_angular_resolution;
    // TODO: Maybe make this a specialization constant?
    int cascade_count = radiance_cascade_metadata.size.w;

    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), ivec3(cascade_size-0.5));
    vec2 tex_coord = concentric_octahedral_mapping_inverse(dir);

    mat3 ltc_irradiance_transform = create_tangent_space(normal);

    ivec2 selected_cell = ivec2(0);
    float selected_weight = 0.0f;
    float selected_value = 0.0f;
    float selected_visibility = 1.0f;
    float sum_weight = 0.0f;

    float inv_probe_resolution = 1.0f / probe_resolution;

    ivec2 itex_coord = ivec2(tex_coord * probe_resolution);

    for(int x = 0; x < radiance_cascade_metadata.c0_angular_resolution; ++x)
    for(int y = 0; y < radiance_cascade_metadata.c0_angular_resolution; ++y)
    {
        ivec2 p = ivec2(x, y);
        vec3 tex_coord = cascade_coord + vec3(p * cascade_size.xy, 0);

        vec2 col = textureLod(radiance_cascades[0], tex_coord, 0).rg;
        float value = col.r;
        float visibility = col.g;

        vec3 corners[4] =  vec3[4](
            concentric_octahedral_mapping((p + ivec2(0,0)) * inv_probe_resolution),
            concentric_octahedral_mapping((p + ivec2(0,1)) * inv_probe_resolution),
            concentric_octahedral_mapping((p + ivec2(1,1)) * inv_probe_resolution),
            concentric_octahedral_mapping((p + ivec2(1,0)) * inv_probe_resolution)
        );

        float weight = value * cosine_hemisphere_poly_light(ltc_irradiance_transform, vec3(0), corners, false);
        weight += 1e-16f;

        sum_weight += weight;
        if(x == itex_coord.x && y == itex_coord.y)
        {
            selected_cell = p;
            selected_weight = weight;
            selected_value = value;
            selected_visibility = visibility;
        }
    }

    float pdf = selected_weight / sum_weight;
    float total_visibility = selected_visibility;

    for(int cascade = 1; cascade < cascade_count && total_visibility > 0; ++cascade)
    { // Drill deeper
        cascade_size /= 2;
        probe_resolution *= 2;
        inv_probe_resolution *= 0.5f;
        cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), ivec3(cascade_size-0.5));
        itex_coord = ivec2(tex_coord * probe_resolution);

        ivec2 base_cell = selected_cell * 2;
        float base_value = selected_value;
        selected_weight = 0.0f;
        selected_value = 0.0f;
        selected_visibility = 0.0f;
        sum_weight = 0.0f;
        for(int x = base_cell.x; x < base_cell.x + 2; ++x)
        for(int y = base_cell.y; y < base_cell.y + 2; ++y)
        {
            ivec2 p = ivec2(x, y);
            vec3 tex_coord = cascade_coord + vec3(p * cascade_size.xy, 0);
            vec2 col = textureLod(radiance_cascades[cascade], tex_coord, 0).rg;
            float value = mix(base_value, col.r, total_visibility);
            float visibility = col.g;

            vec3 corners[4] =  vec3[4](
                concentric_octahedral_mapping((p + ivec2(0,0)) * inv_probe_resolution),
                concentric_octahedral_mapping((p + ivec2(0,1)) * inv_probe_resolution),
                concentric_octahedral_mapping((p + ivec2(1,1)) * inv_probe_resolution),
                concentric_octahedral_mapping((p + ivec2(1,0)) * inv_probe_resolution)
            );
            float weight = value * cosine_hemisphere_poly_light(ltc_irradiance_transform, vec3(0), corners, false);
            weight += 1e-16f;

            sum_weight += weight;
            if(x == itex_coord.x && y == itex_coord.y)
            {
                selected_cell = p;
                selected_weight = weight;
                selected_value = value;
                selected_visibility = visibility;
            }
        }
        pdf *= selected_weight / sum_weight;
        total_visibility *= selected_visibility;
    }

    // Sample from inside the selected cell.
    pdf *= (probe_resolution * probe_resolution) / (4 * M_PI);
    return pdf;
}
#endif

#endif
