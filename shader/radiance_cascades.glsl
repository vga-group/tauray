#ifndef RADIANCE_CASCADES_GLSL
#define RADIANCE_CASCADES_GLSL
#include "math.glsl"
#include "color.glsl"
#include "ltc.glsl"

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

vec3 get_texel_corner(ivec2 texel, vec2 corner, float inv_probe_resolution, mat3 tbn)
{
    vec3 dir = concentric_octahedral_mapping((vec2(texel) + corner) * inv_probe_resolution) * tbn;
    dir.z = max(dir.z, 0.0);
    return dir;
}

vec4 integrate_quad(
    mat3 tbn,
    vec3 ltc_transform,
    float inv_probe_resolution,
    int cascade,
    ivec2 p,
    ivec3 base_tex_coord,
    float fresnel,
    float specular_amplitude
){
    vec4 values = textureGather(radiance_cascades[cascade], vec3(base_tex_coord.xy + 1.0, base_tex_coord.z));
    vec4 diff_z = vec4(0.0f);
    vec4 spec_z = vec4(0.0f);

    // [v00]----[v10]----[v20]
    //   |   <-   |   <-   |
    //   | v .w ^ | v .z ^ |
    //   |   ->   |   ->   |
    // [v01]----[v11]----[v21]
    //   |   <-   |   <-   |
    //   | v .x ^ | v .y ^ |
    //   |   ->   |   ->   |
    // [v02]----[v12]----[v22]

    vec3 v00 = get_texel_corner(p, vec2(0,0), inv_probe_resolution, tbn);
    vec3 v01 = get_texel_corner(p, vec2(0,1), inv_probe_resolution, tbn);
    vec3 v02 = get_texel_corner(p, vec2(0,2), inv_probe_resolution, tbn);
    vec3 v10 = get_texel_corner(p, vec2(1,0), inv_probe_resolution, tbn);
    vec3 v11 = get_texel_corner(p, vec2(1,1), inv_probe_resolution, tbn);
    vec3 v12 = get_texel_corner(p, vec2(1,2), inv_probe_resolution, tbn);
    vec3 v20 = get_texel_corner(p, vec2(2,0), inv_probe_resolution, tbn);
    vec3 v21 = get_texel_corner(p, vec2(2,1), inv_probe_resolution, tbn);
    vec3 v22 = get_texel_corner(p, vec2(2,2), inv_probe_resolution, tbn);

    diff_z.w += edge_integral(v10, v00);
    diff_z.w += edge_integral(v00, v01);
    diff_z.z += edge_integral(v21, v20);
    diff_z.z += edge_integral(v20, v10);
    diff_z.x += edge_integral(v01, v02);
    diff_z.x += edge_integral(v02, v12);
    diff_z.y += edge_integral(v12, v22);
    diff_z.y += edge_integral(v22, v21);

    float d01_11 = edge_integral(v01, v11);
    diff_z.w += d01_11;
    diff_z.x -= d01_11;

    float d11_10 = edge_integral(v11, v10);
    diff_z.w += d11_10;
    diff_z.z -= d11_10;

    float d11_21 = edge_integral(v11, v21);
    diff_z.z += d11_21;
    diff_z.y -= d11_21;

    float d12_11 = edge_integral(v12, v11);
    diff_z.x += d12_11;
    diff_z.y -= d12_11;

    v11 = ltc_transform_dir3(ltc_transform, v11);
    v00 = ltc_transform_dir3(ltc_transform, v00);
    v02 = ltc_transform_dir3(ltc_transform, v02);
    v01 = ltc_transform_dir3(ltc_transform, v01);
    spec_z.w += edge_integral(v00, v01);
    spec_z.x += edge_integral(v01, v02);
    float s01_11 = edge_integral(v01, v11);
    spec_z.w += s01_11;
    spec_z.x -= s01_11;

    v20 = ltc_transform_dir3(ltc_transform, v20);
    v10 = ltc_transform_dir3(ltc_transform, v10);
    spec_z.w += edge_integral(v10, v00);
    spec_z.z += edge_integral(v20, v10);
    float s11_10 = edge_integral(v11, v10);
    spec_z.w += s11_10;
    spec_z.z -= s11_10;

    v22 = ltc_transform_dir3(ltc_transform, v22);
    v21 = ltc_transform_dir3(ltc_transform, v21);
    spec_z.z += edge_integral(v21, v20);
    spec_z.y += edge_integral(v22, v21);
    float s11_21 = edge_integral(v11, v21);
    spec_z.z += s11_21;
    spec_z.y -= s11_21;

    v12 = ltc_transform_dir3(ltc_transform, v12);
    spec_z.x += edge_integral(v02, v12);
    spec_z.y += edge_integral(v12, v22);
    float s12_11 = edge_integral(v12, v11);
    spec_z.x += s12_11;
    spec_z.y -= s12_11;

    vec4 sum_contrib = (1.0 - fresnel) * max(diff_z, vec4(0.0)) + specular_amplitude * max(spec_z, vec4(0.0));
    return values * sum_contrib;
}

float eval_diffuse_radiance_cascades_split(vec3 origin, vec3 normal, vec3 view, float roughness, float f0)
{
    vec3 aabb_min = radiance_cascade_metadata.aabb_min.xyz;
    vec3 aabb_max = radiance_cascade_metadata.aabb_max.xyz;

    // 0-1 inside cascade volume
    vec3 fcoord = (origin - aabb_min) / (aabb_max - aabb_min);

    ivec3 cascade_size = radiance_cascade_metadata.size.xyz;
    int probe_resolution = radiance_cascade_metadata.c0_angular_resolution;

    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));

    float vdotn = dot(view, normal);
    vec3 ltc_transform = ltc_ggx_transform(vdotn, max(roughness, 0.01f));
    mat3 tbn = create_tangent_space(normal, view);

    float inv_probe_resolution = 1.0f / probe_resolution;
    float fresnel = f0 + (1.0 - f0) * ggx_fresnel(vdotn, roughness);
    float specular_amplitude = fresnel + f0 * ggx_albedo(vdotn, roughness) - f0;

    float contrib = dot(integrate_quad(
        tbn,
        ltc_transform,
        inv_probe_resolution,
        0,
        ivec2(0,0),
        get_cascade_layout(cascade_size, RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), ivec2(0,0)),
        fresnel,
        specular_amplitude
    ), vec4(1));

    contrib += dot(integrate_quad(
        tbn,
        ltc_transform,
        inv_probe_resolution,
        0,
        ivec2(2,0),
        get_cascade_layout(cascade_size, RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), ivec2(2,0)),
        fresnel,
        specular_amplitude
    ), vec4(1));

    contrib += dot(integrate_quad(
        tbn,
        ltc_transform,
        inv_probe_resolution,
        0,
        ivec2(0,2),
        get_cascade_layout(cascade_size, RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), ivec2(0,2)),
        fresnel,
        specular_amplitude
    ), vec4(1));

    contrib += dot(integrate_quad(
        tbn,
        ltc_transform,
        inv_probe_resolution,
        0,
        ivec2(2,2),
        get_cascade_layout(cascade_size, RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), ivec2(2,2)),
        fresnel,
        specular_amplitude
    ), vec4(1));
    return contrib;

    /*
    float sum_diff = 0.0f;
    float sum_spec = 0.0f;
    [[unroll]] for(int x = 0; x < RC_C0_ANGULAR_RESOLUTION; ++x)
    [[unroll]] for(int y = 0; y < RC_C0_ANGULAR_RESOLUTION; ++y)
    {
        ivec2 p = ivec2(x, y);
        ivec3 tex_coord = get_cascade_layout(cascade_size, RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), p);
        float value = texelFetch(radiance_cascades[0], tex_coord, 0).r;

        vec3 v[4] = {
            get_texel_corner(p, vec2(0,0), inv_probe_resolution, tbn),
            get_texel_corner(p, vec2(0,1), inv_probe_resolution, tbn),
            get_texel_corner(p, vec2(1,1), inv_probe_resolution, tbn),
            get_texel_corner(p, vec2(1,0), inv_probe_resolution, tbn),
        };

        float diff_z = 0.0f;
        float spec_z = 0.0f;
        for(uint i = 0; i < 4; ++i)
        {
            vec3 a = v[i];
            vec3 b = v[(i+1)&3];
            diff_z += edge_integral(a, b);
            spec_z += edge_integral(
                ltc_transform_dir3(ltc_transform, a),
                ltc_transform_dir3(ltc_transform, b)
            );
        }
        sum_diff += value * max(diff_z, 0.0);
        sum_spec += value * max(spec_z, 0.0);
    }
    float sum_contrib = (1.0 - fresnel) * sum_diff + specular_amplitude * sum_spec;
    return sum_contrib;
    */
}

float eval_diffuse_radiance_cascades(vec3 origin, vec3 normal, vec3 view, float roughness, float f0)
{
    return eval_diffuse_radiance_cascades_split(origin, normal, view, roughness, f0);

    vec3 aabb_min = radiance_cascade_metadata.aabb_min.xyz;
    vec3 aabb_max = radiance_cascade_metadata.aabb_max.xyz;

    // 0-1 inside cascade volume
    vec3 fcoord = (origin - aabb_min) / (aabb_max - aabb_min);

    ivec3 cascade_size = radiance_cascade_metadata.size.xyz;
    int probe_resolution = radiance_cascade_metadata.c0_angular_resolution;

    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));

    //mat3 ltc_irradiance_transform = create_tangent_space(normal);

    float vdotn = dot(view, normal);
    vec3 ltc_transform = ltc_ggx_transform(vdotn, max(roughness, 0.01f));

    // TODO: Create TBN aligned with view
    mat3 tbn = create_tangent_space(normal, view);

    vec3 r = reflect(-view, normal);

    float sum_diffuse = 0;
    float sum_specular = 0;

    float inv_probe_resolution = 1.0f / probe_resolution;
    float len = 4.0f * inv_probe_resolution * inv_probe_resolution;
    float cross_result = inv_probe_resolution * inv_probe_resolution / (4.0f * M_PI * M_PI);


    vec3 ltc_ref = ltc_transform_dir3(ltc_transform, vec3(0,0,1));

    [[unroll]] for(int x = 0; x < RC_C0_ANGULAR_RESOLUTION; ++x)
    [[unroll]] for(int y = 0; y < RC_C0_ANGULAR_RESOLUTION; ++y)
    {
        ivec2 p = ivec2(x, y);
        ivec3 tex_coord = get_cascade_layout(cascade_size, RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), p);
        float value = texelFetch(radiance_cascades[0], tex_coord, 0).r;

        vec3 center = concentric_octahedral_mapping((vec2(p) + 0.5f) * inv_probe_resolution);
        center = center * tbn;
        float cdn = center.z;
        // Diffuse
        sum_diffuse += value * max(len + cdn, 0.0f);

        // Specular
        float inv_len;
        vec4 ltc_center = ltc_transform_dir(ltc_transform, center, inv_len);
        //cdn = ltcCenter.z;
        //// TODO: Can't use same approximation as diffuse. Explore options.
        //float spec_len = 2.0f * ltcCenter.w * inv_probe_resolution;
        //sum_specular += value * max(spec_len * spec_len + cdn, 0.0f) / (spec_len + 1.0);

        //float spec_len = len * ltc_center.w;
        //sum_specular += value * spec_len * max(spec_len + ltc_center.z, 0.0f) / (spec_len + 1.0);


        // TODO: Accurate, but way too slow.
        // Instead, formulate this as a disk _somehow_.
        // Computing illumination from a disk?
        vec3 v[4] = {
            concentric_octahedral_mapping((vec2(p) + vec2(0,0)) * inv_probe_resolution) * tbn,
            concentric_octahedral_mapping((vec2(p) + vec2(0,1)) * inv_probe_resolution) * tbn,
            concentric_octahedral_mapping((vec2(p) + vec2(1,1)) * inv_probe_resolution) * tbn,
            concentric_octahedral_mapping((vec2(p) + vec2(1,0)) * inv_probe_resolution) * tbn
        };

        vec3 last = ltc_transform_dir3(ltc_transform, v[3]);
        vec3 prev = last;
        vec3 form_factor = vec3(0);

        for(uint i = 0; i < 3; ++i)
        {
            vec3 d = ltc_transform_dir3(ltc_transform, v[i]);
            form_factor += edge_vector_form_factor(prev, d);
            prev = d;
        }
        form_factor += edge_vector_form_factor(prev, last);

        // Approx horizon clipping
        float flen = length(form_factor);
        float irradiance1 = max((flen*flen+form_factor.z)/(flen+1.0), 0.0);
        sum_specular += value * irradiance1;

        /*
        // Computing illumination from a disk: poor quality.
        float A1 = min(len / ltc_center.w, 1.0f);
        float radius = sqrt(A1 - A1*A1*0.25)/(1.0-A1);
        vec3 center_to_ray = ltc_center.xyz - dot(ltc_center.xyz, ltc_ref) * ltc_ref;
        vec3 closest_dir = normalize(ltc_center.xyz + center_to_ray * min(radius/length(center_to_ray), 1.0));

        // Main source of error:
        // How to replace? Needs a better horizon approximation.
        float spec_len = A1;
        //sum_specular += value * max(spec_len * ltc_center.z / (spec_len + 1.0f), 0.0f);
        //sum_specular += value * max(spec_len * closest_dir.z / (spec_len + 1.0f), 0.0f);
        sum_specular += value * max(closest_dir.z, 0.0f);
        */


        /*
        float spec_z = min(
            ltc_center.z + sqrt(len * ltc_center.w * max(1.0f - ltc_center.z * ltc_center.z, 0.0f)),
            1.0f
        );
        //float spec_z = min(ltc_center.z, 1.0f);
        float spec_len = ltc_center.w * len;
        //float irradiance2 = max(texel_area * centroid_t.w * z2, 0.0f);
        sum_specular += value * max(spec_len * spec_z / (spec_len + 1.0f), 0.0f);
        */
    }

    sum_diffuse *= 1.0f / (1.0f + 0.25f * probe_resolution * probe_resolution);

    float fresnel = f0 + (1.0 - f0) * ggx_fresnel(vdotn, roughness);
    float specular_amplitude = fresnel + f0 * ggx_albedo(vdotn, roughness) - f0;

    float sum_contrib = (1.0 - fresnel) * sum_diffuse + specular_amplitude * sum_specular;

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
