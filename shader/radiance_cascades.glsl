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
layout(set=RADIANCE_CASCADES_SET, binding = 2) uniform sampler2DArray radiance_cascades_read[];
layout(set=RADIANCE_CASCADES_SET, binding = 3) uniform radiance_cascade_metadata_buffer
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

vec2 radiance_cascade_probe_mapping_inverse(vec3 dir)
{
    return octahedral_mapping_inverse(dir) * 0.5f + 0.5f;
    //return concentric_octahedral_mapping_inverse(dir);
}

vec3 radiance_cascade_probe_mapping(vec2 u)
{
    return octahedral_mapping(u * 2.0f - 1.0f);
    //return concentric_octahedral_mapping(clamp(u, vec2(0.0f), vec2(1.0f)));
}

void octahedral_mapping(
    f16vec2 packed_normal_x,
    f16vec2 packed_normal_y,
    out f16vec2 normal_x,
    out f16vec2 normal_y,
    out f16vec2 normal_z
){
    normal_x = packed_normal_x;
    normal_y = f16vec2(1.0) - abs(packed_normal_x) - abs(packed_normal_y);
    normal_z = packed_normal_y;

    f16vec2 ny = clamp(normal_y, f16vec2(-1.0), f16vec2(0.0));
    normal_x = normal_x + mix(-ny, ny, greaterThan(normal_x, f16vec2(0.0)));
    normal_z = normal_z + mix(-ny, ny, greaterThan(normal_z, f16vec2(0.0)));
}

void radiance_cascade_probe_mapping(
    f16vec2 u_x,
    f16vec2 u_y,
    out f16vec2 normal_x,
    out f16vec2 normal_y,
    out f16vec2 normal_z
){
    octahedral_mapping(
        u_x * f16vec2(2.0) - f16vec2(1.0),
        u_y * f16vec2(2.0) - f16vec2(1.0),
        normal_x,
        normal_y,
        normal_z
    );
    //return concentric_octahedral_mapping(clamp(u, vec2(0.0f), vec2(1.0f)));
}

f16vec2 ltc_eval(
    f16vec3 transform,
    f16vec2 dir_x,
    f16vec2 dir_y,
    f16vec2 dir_z
){
    f16vec2 new_dir_x = dir_x * transform.x + dir_z * transform.y;
    f16vec2 new_dir_y = dir_y;
    f16vec2 new_dir_z = dir_z * transform.z;
    //f16vec2 inv_len2 = f16vec2(1.0) / (new_dir_x * new_dir_x + new_dir_y * new_dir_y + new_dir_z * new_dir_z);
    f16vec2 inv_len2 = f16vec2(transform.z) / (new_dir_x * new_dir_x + new_dir_y * new_dir_y + new_dir_z * new_dir_z);
    return (transform.x * inv_len2) * (inv_len2 * max(dir_z, f16vec2(0.0f)));
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

    vec2 uv = radiance_cascade_probe_mapping_inverse(dir);

    //ivec2 p = octahedral_wrap(ivec2(floor(uv * probe_resolution)), ivec2(probe_resolution));
    ivec2 p = ivec2(uv * probe_resolution);
    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));
    ivec3 tex_coord = get_cascade_layout(cascade_size, probe_resolution, ivec3(cascade_coord), p);
    float col = texelFetch(radiance_cascades[0], tex_coord, 0).r;
    float vis = texelFetch(radiance_cascades_visibility[0], tex_coord, 0).r;
    sum.rgb += col.rrr * vis * sum.a;
    sum.a *= 1.0f-vis;

    for(int cascade = 1; sum.a > 0 && cascade < RC_CASCADE_COUNT; ++cascade)
    {
        probe_resolution *= 2;
        cascade_size /= 2;

        //ivec2 p = octahedral_wrap(ivec2(floor(uv * probe_resolution)), ivec2(probe_resolution));
        ivec2 p = ivec2(uv * probe_resolution);
        vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));
        ivec3 tex_coord = get_cascade_layout(cascade_size, probe_resolution, ivec3(cascade_coord), p);
        col = texelFetch(radiance_cascades[cascade], tex_coord, 0).r;
        vis = texelFetch(radiance_cascades_visibility[cascade], tex_coord, 0).r;
        sum.rgb += col.rrr * vis * sum.a;
        sum.a *= 1.0f-vis;
    }
    return sum.rgb;
}

void get_unclamped_texel_corner(
    f16vec2 base,
    f16vec2 corner_x,
    f16vec2 corner_y,
    float16_t inv_probe_resolution,
    f16vec3 tangent,
    f16vec3 bitangent,
    f16vec3 normal,
    out f16vec2 result_x,
    out f16vec2 result_y,
    out f16vec2 result_z
){
    f16vec2 u_x = corner_x * inv_probe_resolution + base.x;
    f16vec2 u_y = corner_y * inv_probe_resolution + base.y;
    f16vec2 dir_x;
    f16vec2 dir_y;
    f16vec2 dir_z;
    radiance_cascade_probe_mapping(
        u_x,
        u_y,
        dir_x,
        dir_y,
        dir_z
    );

    result_x = dir_x * tangent.x + dir_y * tangent.y + dir_z * tangent.z;
    result_y = dir_x * bitangent.x + dir_y * bitangent.y + dir_z * bitangent.z;
    result_z = dir_x * normal.x + dir_y * normal.y + dir_z * normal.z;

    f16vec2 len2 = result_x * result_x + result_y * result_y + result_z * result_z;
    f16vec2 inv_len = inversesqrt(len2 + f16vec2(0.0004));

    result_x = result_x * inv_len;
    result_y = result_y * inv_len;
    result_z = result_z * inv_len;
}

void get_texel_corner(
    f16vec2 base,
    f16vec2 corner_x,
    f16vec2 corner_y,
    float16_t inv_probe_resolution,
    f16vec3 tangent,
    f16vec3 bitangent,
    f16vec3 normal,
    out f16vec2 result_x,
    out f16vec2 result_y,
    out f16vec2 result_z
){
    f16vec2 u_x = corner_x * inv_probe_resolution + base.x;
    f16vec2 u_y = corner_y * inv_probe_resolution + base.y;
    f16vec2 dir_x;
    f16vec2 dir_y;
    f16vec2 dir_z;
    radiance_cascade_probe_mapping(
        u_x,
        u_y,
        dir_x,
        dir_y,
        dir_z
    );

    result_x = dir_x * tangent.x + dir_y * tangent.y + dir_z * tangent.z;
    result_y = dir_x * bitangent.x + dir_y * bitangent.y + dir_z * bitangent.z;
    result_z = max(dir_x * normal.x + dir_y * normal.y + dir_z * normal.z, f16vec2(0.0));

    f16vec2 len2 = result_x * result_x + result_y * result_y + result_z * result_z;
    f16vec2 inv_len = inversesqrt(len2 + f16vec2(0.0004));

    result_x = result_x * inv_len;
    result_y = result_y * inv_len;
    result_z = result_z * inv_len;
}

#ifndef printf
#define printf
#endif

vec3 rc_uniform_texel_sample(
    inout uint seed,
    bool sample_specular,
    float specular_prob,
    ivec2 selected_cell,
    int probe_resolution,
    float16_t inv_probe_resolution,
    inout float pdf
#ifdef RC_SAMPLE_SINGLE_LOBE
    , out float mis_pdf
#endif
){
    vec2 uv = vec2(
        generate_single_uniform_random_fast(seed),
        generate_single_uniform_random_fast(seed)
    );
    uv = (vec2(selected_cell) + uv) * inv_probe_resolution;

    pdf *= probe_resolution * probe_resolution * 0.25f * octahedral_mapping_abs_jacobian_det(uv*2.0-1.0);

#ifdef RC_SAMPLE_SINGLE_LOBE
    mis_pdf = pdf;
    pdf *= sample_specular ? specular_prob : (1.0-specular_prob);
#endif

    vec3 dir = radiance_cascade_probe_mapping(uv);
    return dir;
}

vec3 rc_texel_sample(
    inout uint seed,
    ivec2 selected_cell,
    int probe_resolution,
    float16_t inv_probe_resolution,
    f16vec3 tangent,
    f16vec3 bitangent,
    f16vec3 normal,
    f16vec3 ltc_transform,
    float roughness,
    float diffuse_weight,
    float specular_weight,
    f16vec2 brdf_weight,
    inout float pdf
#ifdef RC_SAMPLE_SINGLE_LOBE
    , out float mis_pdf,
    out uint sampled_lobe
#endif
){
    float sum_brdf_weight = (diffuse_weight * brdf_weight.x + specular_weight * brdf_weight.y) * M_PI;
    float specular_prob = specular_weight * brdf_weight.y * M_PI / sum_brdf_weight;
    bool sample_specular = generate_single_uniform_random_fast(seed) < specular_prob;

#ifdef RC_SAMPLE_SINGLE_LOBE
    sampled_lobe = sample_specular ? MATERIAL_LOBE_REFLECTION : MATERIAL_LOBE_DIFFUSE;
#endif

    return rc_uniform_texel_sample(
        seed, sample_specular, specular_prob, selected_cell, probe_resolution,
        inv_probe_resolution, pdf
#ifdef RC_SAMPLE_SINGLE_LOBE
        , mis_pdf
#endif
    );
}

float rc_uniform_texel_pdf(
    vec2 uv,
    int probe_resolution,
    float diffuse_weight,
    float specular_weight,
    f16vec2 brdf_weight,
    float pdf
#ifdef RC_SAMPLE_SINGLE_LOBE
    , uint sampled_lobe
#endif
){
    float sum_brdf_weight = (diffuse_weight * brdf_weight.x + specular_weight * brdf_weight.y) * M_PI;

    pdf *= probe_resolution * probe_resolution * 0.25f * octahedral_mapping_abs_jacobian_det(uv*2.0-1.0);

#ifdef RC_SAMPLE_SINGLE_LOBE
    float specular_prob = specular_weight * brdf_weight.y * M_PI / sum_brdf_weight;
    if(sampled_lobe != MATERIAL_LOBE_ALL)
        pdf *= sampled_lobe == MATERIAL_LOBE_REFLECTION ? specular_prob : (1.0-specular_prob);
#endif

    return pdf;
}

float rc_texel_pdf(
    vec2 uv,
    vec3 tdir,
    int probe_resolution,
    f16vec3 ltc_transform,
    float roughness,
    float diffuse_weight,
    float specular_weight,
    f16vec2 brdf_weight,
    float pdf
#ifdef RC_SAMPLE_SINGLE_LOBE
    , uint sampled_lobe
#endif
){
    return rc_uniform_texel_pdf(
        uv,
        probe_resolution,
        diffuse_weight,
        specular_weight,
        brdf_weight,
        pdf
#ifdef RC_SAMPLE_SINGLE_LOBE
        , sampled_lobe
#endif
    );
}

void integrate_quad_half_precision(
    f16vec3 tangent,
    f16vec3 bitangent,
    f16vec3 normal,
    f16vec3 ltc_transform,
    float16_t inv_probe_resolution,
    f16vec2 base,
    out f16vec4 diffuse,
    out f16vec4 specular
){
    // [v00]----[v10]----[v20]
    //   |   <-   |   <-   |
    //   | v .z ^ | v .x ^ |
    //   |   ->   |   ->   |
    // [v01]----[v11]----[v21]
    //   |   <-   |   <-   |
    //   | v .y ^ | v .w ^ |
    //   |   ->   |   ->   |
    // [v02]----[v12]----[v22]

    // To keep things simple, there should be no shared edges between the pairs
    // here. v11 is the only one that gets orphaned, but since there's an
    // odd number of vectors here, that'll happen anyway.
    f16vec2 h00_h22_x, h00_h22_y, h00_h22_z;
    get_texel_corner(
        base, f16vec2(0,2), f16vec2(0,2), inv_probe_resolution,
        tangent, bitangent, normal,
        h00_h22_x, h00_h22_y, h00_h22_z
    );

    f16vec2 h10_h12_x, h10_h12_y, h10_h12_z;
    get_texel_corner(
        base, f16vec2(1,1), f16vec2(0,2), inv_probe_resolution,
        tangent, bitangent, normal,
        h10_h12_x, h10_h12_y, h10_h12_z
    );

    f16vec2 h20_h02_x, h20_h02_y, h20_h02_z;
    get_texel_corner(
        base, f16vec2(2,0), f16vec2(0,2), inv_probe_resolution,
        tangent, bitangent, normal,
        h20_h02_x, h20_h02_y, h20_h02_z
    );

    f16vec2 h21_h01_x, h21_h01_y, h21_h01_z;
    get_texel_corner(
        base, f16vec2(2,0), f16vec2(1,1), inv_probe_resolution,
        tangent, bitangent, normal,
        h21_h01_x, h21_h01_y, h21_h01_z
    );

    f16vec2 h11_h11_x, h11_h11_y, h11_h11_z;
    get_texel_corner(
        base, f16vec2(1,1), f16vec2(1,1), inv_probe_resolution,
        tangent, bitangent, normal,
        h11_h11_x, h11_h11_y, h11_h11_z
    );

    f16vec4 diff_z = f16vec4(0.0f);
    f16vec2 r = edge_integral(h10_h12_x, h10_h12_y, h10_h12_z, h00_h22_x, h00_h22_y, h00_h22_z);
    diff_z.zw += r.xy;

    r = edge_integral(h20_h02_x, h20_h02_y, h20_h02_z, h10_h12_x, h10_h12_y, h10_h12_z);
    diff_z.xy += r.xy;

    r = edge_integral(h21_h01_x, h21_h01_y, h21_h01_z, h20_h02_x, h20_h02_y, h20_h02_z);
    diff_z.xy += r.xy;

    r = edge_integral(h00_h22_x, h00_h22_y, h00_h22_z, h21_h01_x.yx, h21_h01_y.yx, h21_h01_z.yx);
    diff_z.zw += r.xy;

    r = edge_integral(h11_h11_x, h11_h11_y, h11_h11_z, h21_h01_x, h21_h01_y, h21_h01_z);
    diff_z.xy += r.xy;
    diff_z.zw -= r.yx;

    r = edge_integral(h11_h11_x, h11_h11_y, h11_h11_z, h10_h12_x, h10_h12_y, h10_h12_z);
    diff_z.xy -= r.xy;
    diff_z.zw += r.xy;

    ltc_transform_dir3(ltc_transform, h00_h22_x, h00_h22_y, h00_h22_z, h00_h22_x, h00_h22_y, h00_h22_z);
    ltc_transform_dir3(ltc_transform, h10_h12_x, h10_h12_y, h10_h12_z, h10_h12_x, h10_h12_y, h10_h12_z);
    ltc_transform_dir3(ltc_transform, h20_h02_x, h20_h02_y, h20_h02_z, h20_h02_x, h20_h02_y, h20_h02_z);
    ltc_transform_dir3(ltc_transform, h21_h01_x, h21_h01_y, h21_h01_z, h21_h01_x, h21_h01_y, h21_h01_z);
    ltc_transform_dir3(ltc_transform, h11_h11_x, h11_h11_y, h11_h11_z, h11_h11_x, h11_h11_y, h11_h11_z);

    f16vec4 spec_z = f16vec4(0.0f);
    r = edge_integral(h10_h12_x, h10_h12_y, h10_h12_z, h00_h22_x, h00_h22_y, h00_h22_z);
    spec_z.zw += r.xy;

    r = edge_integral(h20_h02_x, h20_h02_y, h20_h02_z, h10_h12_x, h10_h12_y, h10_h12_z);
    spec_z.xy += r.xy;

    r = edge_integral(h21_h01_x, h21_h01_y, h21_h01_z, h20_h02_x, h20_h02_y, h20_h02_z);
    spec_z.xy += r.xy;

    r = edge_integral(h00_h22_x, h00_h22_y, h00_h22_z, h21_h01_x.yx, h21_h01_y.yx, h21_h01_z.yx);
    spec_z.zw += r.xy;

    r = edge_integral(h11_h11_x, h11_h11_y, h11_h11_z, h21_h01_x, h21_h01_y, h21_h01_z);
    spec_z.xy += r.xy;
    spec_z.zw -= r.yx;

    r = edge_integral(h11_h11_x, h11_h11_y, h11_h11_z, h10_h12_x, h10_h12_y, h10_h12_z);
    spec_z.xy -= r.xy;
    spec_z.zw += r.xy;

    // Mask out quads that are below the horizon.
    // These get included often due to rounding errors in our half-precision
    // approximate math.
    f16vec4 mask = f16vec4(1.0f);
    if (h11_h11_z.x == float16_t(0))
    {
        if (h00_h22_z.x == float16_t(0) && h10_h12_z.x == float16_t(0) && h21_h01_z.y == float16_t(0))
            mask.z = float16_t(0);
        if (h10_h12_z.x == float16_t(0) && h20_h02_z.x == float16_t(0) && h21_h01_z.x == float16_t(0))
            mask.x = float16_t(0);
        if (h21_h01_z.y == float16_t(0) && h20_h02_z.y == float16_t(0) && h10_h12_z.y == float16_t(0))
            mask.y = float16_t(0);
        if (h10_h12_z.y == float16_t(0) && h21_h01_z.x == float16_t(0) && h00_h22_z.y == float16_t(0))
            mask.w = float16_t(0);
    }

    diffuse = max(diff_z * mask, f16vec4(0.0));
    specular = max(spec_z * mask, f16vec4(0.0));
    //specular = max(spec_z, f16vec4(0.0));
    //specular += dot(specular, f16vec4(0.01));
    //specular = specular * mask;
}

float eval_radiance_cascades(vec3 origin, vec3 normal, vec3 view, float roughness, float f0)
{
    vec3 aabb_min = radiance_cascade_metadata.aabb_min.xyz;
    vec3 aabb_max = radiance_cascade_metadata.aabb_max.xyz;

    // 0-1 inside cascade volume
    vec3 fcoord = (origin - aabb_min) / (aabb_max - aabb_min);

    ivec3 cascade_size = radiance_cascade_metadata.size.xyz;
    int probe_resolution = radiance_cascade_metadata.c0_angular_resolution;

    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));

    float vdotn = dot(view, normal);
    f16vec3 ltc_transform = f16vec3(ltc_ggx_transform(vdotn, max(roughness, 0.01f)));
    mat3 tbn = create_tangent_space(normal, view);
    f16vec3 tangent = f16vec3(tbn[0]);
    f16vec3 bitangent = f16vec3(tbn[1]);
    f16vec3 hnormal = f16vec3(tbn[2]);

    float16_t inv_probe_resolution = float16_t(1.0f / probe_resolution);
    float16_t fresnel = float16_t(f0 + (1.0 - f0) * ggx_fresnel(vdotn, roughness));
    float16_t specular_amplitude = fresnel + float16_t(f0 * ggx_albedo(vdotn, roughness) - f0);
    float16_t flip_fresnel = float16_t(1.0) - fresnel;

    float16_t contrib = float16_t(0);
    for(int x = 0; x < RC_C0_ANGULAR_RESOLUTION; x+=2)
    for(int y = 0; y < RC_C0_ANGULAR_RESOLUTION; y+=2)
    {
        ivec3 base_tex_coord = get_cascade_layout(cascade_size, RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), ivec2(x, y));
        f16vec4 values = f16vec4(textureGather(radiance_cascades[0], vec3(base_tex_coord.xy + 1.0, base_tex_coord.z)).zxwy);
        f16vec4 diffuse;
        f16vec4 specular;
        integrate_quad_half_precision(
            tangent,
            bitangent,
            hnormal,
            ltc_transform,
            inv_probe_resolution,
            f16vec2(x,y) * inv_probe_resolution,
            diffuse,
            specular
        );
        contrib += dot(flip_fresnel * diffuse + specular_amplitude * specular, values);
    }
    return contrib;
}

float eval_diffuse_radiance_cascades(vec3 origin, vec3 normal, vec3 view, float roughness, float f0)
{
    return eval_radiance_cascades(origin, normal, view, roughness, f0);
}

float16_t generate_single_uniform_random_fp16(inout uint seed)
{
    return float16_t(lcg(seed)&0xFFFF) * float16_t(INV_UINT16_MAX);
}

void rc_wrs_update(
    inout uint seed,
    ivec2 p,
    f16vec4 weights,
    inout float16_t sum_weight,
    inout float16_t selected_weight,
    inout ivec2 selected_cell,
    f16vec4 values,
    inout float16_t selected_value,
    f16vec4 diffuse,
    f16vec4 specular,
    inout f16vec2 selected_brdf
){
    f16vec2 weight2 = weights.xy + weights.zw;
    float16_t weight = weight2.x + weight2.y;
    sum_weight += weight;
    ivec2 selected_offset = ivec2(0);

    // Don't override a non-zero value with a zero one.
    if (weight == float16_t(0) && sum_weight != float16_t(0))
        return;

    float16_t u = generate_single_uniform_random_fp16(seed);
    if(u*sum_weight <= weight || sum_weight == weight)
    {
        float16_t q = generate_single_uniform_random_fp16(seed);

        if (weight == float16_t(0))
        {
            int i = min(int(q * float16_t(4)), 3);
            selected_offset.x = i&1;
            selected_offset.y = i>>1;
        }
        else
        {
            q *= weight;
            // PLS don't touch the logic here, it's very precise to eliminate
            // float issues.
            if(q >= weight2.x && weight2.y != float16_t(0)) // q > x+z
            {
                q -= weight2.x;
                selected_offset.y = 1;
                selected_offset.x = q >= weights.y && weights.w != float16_t(0) ? 1 : 0;
            }
            else
            {
                selected_offset.x = q >= weights.z && weights.x != float16_t(0) ? 1 : 0;
            }
        }

        selected_weight = selected_offset.y == 0 ?
            (selected_offset.x == 0 ? weights.z : weights.x) :
            (selected_offset.x == 0 ? weights.y : weights.w);
        selected_value = selected_offset.y == 0 ?
            (selected_offset.x == 0 ? values.z : values.x) :
            (selected_offset.x == 0 ? values.y : values.w);
        selected_brdf = selected_offset.y == 0 ?
            (selected_offset.x == 0 ? f16vec2(diffuse.z, specular.z) : f16vec2(diffuse.x, specular.x)) :
            (selected_offset.x == 0 ? f16vec2(diffuse.y, specular.y) : f16vec2(diffuse.w, specular.w));
        selected_cell = p + selected_offset;
    }
}

void rc_wrs_pdf(
    ivec2 p,
    ivec2 itex_coord,
    f16vec4 weights,
    inout float16_t sum_weight,
    inout float16_t selected_weight,
    f16vec4 values,
    inout float16_t selected_value,
    f16vec4 diffuse,
    f16vec4 specular,
    inout f16vec2 selected_brdf
){
    f16vec2 weight2 = weights.xy + weights.zw;
    float16_t weight = weight2.x + weight2.y;
    sum_weight += weight;
    ivec2 selected_offset = itex_coord - p;
    if(selected_offset.x >= 0 && selected_offset.y >= 0 && selected_offset.x <= 1 && selected_offset.y <= 1)
    {
        selected_weight = selected_offset.y == 0 ?
            (selected_offset.x == 0 ? weights.z : weights.x) :
            (selected_offset.x == 0 ? weights.y : weights.w);
        selected_value = selected_offset.y == 0 ?
            (selected_offset.x == 0 ? values.z : values.x) :
            (selected_offset.x == 0 ? values.y : values.w);
        selected_brdf = selected_offset.y == 0 ?
            (selected_offset.x == 0 ? f16vec2(diffuse.z, specular.z) : f16vec2(diffuse.x, specular.x)) :
            (selected_offset.x == 0 ? f16vec2(diffuse.y, specular.y) : f16vec2(diffuse.w, specular.w));
    }
}

f16vec4 sort_f16vec4(f16vec4 v)
{
    if (v.x > v.z) v.xz = v.zx;
    if (v.y > v.w) v.yw = v.wy;
    if (v.x > v.y) v.xy = v.yx;
    if (v.z > v.w) v.zw = v.wz;
    if (v.y > v.z) v.yz = v.zy;
    return v;
}

vec3 sample_radiance_cascades(
    uint seed,
    vec3 origin,
    vec3 normal,
    vec3 view,
    float roughness,
    float f0,
    float albedo,
    out float pdf
#ifdef RC_SAMPLE_SINGLE_LOBE
    , out float mis_pdf,
    out uint sampled_lobe
#endif
){
    vec3 aabb_min = radiance_cascade_metadata.aabb_min.xyz;
    vec3 aabb_max = radiance_cascade_metadata.aabb_max.xyz;

    // 0-1 inside cascade volume
    vec3 fcoord = (origin - aabb_min) / (aabb_max - aabb_min);
    int cascade_size = RC_C0_SPATIAL_RESOLUTION;
    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));

    float vdotn = dot(view, normal);
    f16vec3 ltc_transform = f16vec3(ltc_ggx_transform(vdotn, max(roughness, 0.001f)));
    mat3 tbn = create_tangent_space(normal, view);
    f16vec3 tangent = f16vec3(tbn[0]);
    f16vec3 bitangent = f16vec3(tbn[1]);
    f16vec3 hnormal = f16vec3(tbn[2]);
    float16_t fresnel = float16_t(f0 + (1.0 - f0) * ggx_fresnel(vdotn, roughness));
    float16_t specular_amplitude = fresnel * float16_t(ggx_albedo(vdotn, roughness));
    float16_t flip_fresnel = (float16_t(1.0) - fresnel) * float16_t(albedo/M_PI);

    ivec2 selected_cell = ivec2(0);
    float16_t selected_weight = float16_t(0);
    float16_t selected_value = float16_t(0.0f);
    f16vec2 selected_brdf = f16vec2(0);
    float16_t sum_weight = float16_t(0.0f);

    float16_t inv_probe_resolution = float16_t(1.0f / RC_C0_ANGULAR_RESOLUTION);

    for(int x = 0; x < RC_C0_ANGULAR_RESOLUTION; x+=2)
    for(int y = 0; y < RC_C0_ANGULAR_RESOLUTION; y+=2)
    {
        ivec2 p = ivec2(x, y);
        ivec3 tex_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), p);
        f16vec4 values = f16vec4(textureGather(radiance_cascades_read[0], vec3(tex_coord.xy + 1.0, tex_coord.z)).zxwy);

        f16vec4 diffuse;
        f16vec4 specular;
        integrate_quad_half_precision(
            tangent,
            bitangent,
            hnormal,
            ltc_transform,
            inv_probe_resolution,
            f16vec2(x,y) * inv_probe_resolution,
            diffuse,
            specular
        );
        f16vec4 contrib = values * (flip_fresnel * diffuse + specular_amplitude * specular);

        rc_wrs_update(seed, p, contrib, sum_weight, selected_weight, selected_cell, values, selected_value, diffuse, specular, selected_brdf);
    }

    ivec3 sel_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), selected_cell);
    float16_t visibility = float16_t(1.0 - texelFetch(radiance_cascades_visibility[0], sel_coord, 0).r);
    float16_t total_visibility = float16_t(1.0);

    pdf = sum_weight == float16_t(0) ? 1.0 : float(selected_weight) / float(sum_weight);

    for(int cascade = 1; cascade < RC_CASCADE_COUNT; ++cascade)
    { // Drill deeper
        cascade_size /= 2;
        inv_probe_resolution *= float16_t(0.5f);
        cascade_coord = clamp(cascade_coord * 0.5f, vec3(0.5), vec3(cascade_size-0.5));

        ivec2 base_cell = selected_cell * 2;
        float16_t prev_weight = selected_weight;
        selected_weight = float16_t(0);
        sum_weight = float16_t(0.0f);

        ivec3 tex_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION<<cascade, ivec3(cascade_coord), base_cell);
        f16vec4 values = f16vec4(textureGather(radiance_cascades_read[cascade], vec3(tex_coord.xy + 1.0, tex_coord.z)).zxwy);
#ifdef RC_DEFENSIVE
        f16vec4 r = sort_f16vec4(values);
        float16_t base = dot(r, clamp(visibility-f16vec4(0.75,0.50,0.25,0.00), f16vec4(0.0), f16vec4(0.25)));
        //float16_t base = 
        //    ((visibility > float16_t(0.75) ? r.x : float16_t(0)) +
        //    (visibility > float16_t(0.50) ? r.y : float16_t(0)) +
        //    (visibility > float16_t(0.25) ? r.z : float16_t(0)) +
        //    (visibility > float16_t(0.00) ? r.w : float16_t(0))) * float16_t(0.25);
        values = max(selected_value - base * total_visibility, float16_t(0)) + values * visibility * total_visibility;
        total_visibility *= visibility;
#else
        total_visibility *= visibility;
        values = max(selected_value - dot(values, f16vec4(0.25)) * total_visibility, float16_t(0)) + values * total_visibility;
#endif

        f16vec4 diffuse;
        f16vec4 specular;
        integrate_quad_half_precision(
            tangent,
            bitangent,
            hnormal,
            ltc_transform,
            inv_probe_resolution,
            f16vec2(base_cell) * inv_probe_resolution,
            diffuse,
            specular
        );
        f16vec4 contrib = values * (flip_fresnel * diffuse + specular_amplitude * specular);
        rc_wrs_update(seed, base_cell, contrib, sum_weight, selected_weight, selected_cell, values, selected_value, diffuse, specular, selected_brdf);

        ivec3 sel_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION<<cascade, ivec3(cascade_coord), selected_cell);
        visibility = float16_t(1.0 - texelFetch(radiance_cascades_visibility[cascade], sel_coord, 0).r);
        if (sum_weight != float16_t(0))
            pdf *= float(selected_weight) / float(sum_weight);
    }

    //const int probe_resolution = int(round(1.0 / inv_probe_resolution)); //RC_C0_ANGULAR_RESOLUTION<<(RC_CASCADE_COUNT-1);
    const int probe_resolution = RC_C0_ANGULAR_RESOLUTION<<(RC_CASCADE_COUNT-1);
    //const int probe_resolution = RC_C0_ANGULAR_RESOLUTION;
    return rc_texel_sample(
        seed,
        selected_cell,
        probe_resolution,
        inv_probe_resolution,
        tangent,
        bitangent,
        hnormal,
        ltc_transform,
        roughness,
        flip_fresnel,
        specular_amplitude,
        selected_brdf,
        pdf
#ifdef RC_SAMPLE_SINGLE_LOBE
        , mis_pdf,
        sampled_lobe
#endif
    );
}

float radiance_cascades_pdf(
    vec3 origin,
    vec3 normal,
    vec3 view,
    bool light,
    float roughness,
    float f0,
    float albedo,
    vec3 dir
#ifdef RC_SAMPLE_SINGLE_LOBE
    , uint sampled_lobe
#endif
){
#ifdef RC_USE_RASTER_DI
    // Directional and point lights are skipped in the RC sampling and left to
    // NEE, so no PDF for those.
    if (light)
        return 0.0f;
#endif
    if(dot(dir, dir) == 0.0f)
        return 0.0f;

    vec3 aabb_min = radiance_cascade_metadata.aabb_min.xyz;
    vec3 aabb_max = radiance_cascade_metadata.aabb_max.xyz;

    // 0-1 inside cascade volume
    vec3 fcoord = (origin - aabb_min) / (aabb_max - aabb_min);

    ivec3 cascade_size = radiance_cascade_metadata.size.xyz;
    int probe_resolution = radiance_cascade_metadata.c0_angular_resolution;

    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));
    vec2 tex_coord = radiance_cascade_probe_mapping_inverse(dir);

    float vdotn = dot(view, normal);
    f16vec3 ltc_transform = f16vec3(ltc_ggx_transform(vdotn, max(roughness, 0.001f)));

    mat3 tbn = create_tangent_space(normal, view);
    f16vec3 tangent = f16vec3(tbn[0]);
    f16vec3 bitangent = f16vec3(tbn[1]);
    f16vec3 hnormal = f16vec3(tbn[2]);

    float16_t fresnel = float16_t(f0 + (1.0 - f0) * ggx_fresnel(vdotn, roughness));
    float16_t specular_amplitude = fresnel * float16_t(ggx_albedo(vdotn, roughness));
    float16_t flip_fresnel = (float16_t(1.0) - fresnel) * float16_t(albedo / M_PI);

    float16_t selected_weight = float16_t(0);
    float16_t selected_value = float16_t(0);
    f16vec2 selected_brdf = f16vec2(0);
    float16_t sum_weight = float16_t(0);

    float16_t inv_probe_resolution = float16_t(1.0f / RC_C0_ANGULAR_RESOLUTION);

    ivec2 itex_coord = ivec2(tex_coord * probe_resolution);

    ivec3 sel_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), itex_coord);
    float16_t visibility = float16_t(1.0 - texelFetch(radiance_cascades_visibility[0], sel_coord, 0).r);
    float16_t total_visibility = float16_t(1.0);

    for(int x = 0; x < RC_C0_ANGULAR_RESOLUTION; x+=2)
    for(int y = 0; y < RC_C0_ANGULAR_RESOLUTION; y+=2)
    {
        ivec2 p = ivec2(x, y);
        ivec3 tex_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), p);
        f16vec4 values = f16vec4(textureGather(radiance_cascades_read[0], vec3(tex_coord.xy + 1.0, tex_coord.z)).zxwy);

        f16vec4 diffuse;
        f16vec4 specular;
        integrate_quad_half_precision(
            tangent,
            bitangent,
            hnormal,
            ltc_transform,
            inv_probe_resolution,
            f16vec2(x,y) * inv_probe_resolution,
            diffuse,
            specular
        );
        f16vec4 contrib = values * (flip_fresnel * diffuse + specular_amplitude * specular);

        rc_wrs_pdf(p, itex_coord, contrib, sum_weight, selected_weight, values, selected_value, diffuse, specular, selected_brdf);
    }

    float pdf = sum_weight == float16_t(0) ? 1.0 : float(selected_weight) / float(sum_weight);

    for(int cascade = 1; cascade < RC_CASCADE_COUNT; ++cascade)
    { // Drill deeper
        cascade_size /= 2;
        probe_resolution *= 2;
        inv_probe_resolution *= float16_t(0.5f);
        cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));
        ivec2 base_cell = itex_coord * 2;

        itex_coord = ivec2(tex_coord * probe_resolution);
        float16_t prev_weight = selected_weight;
        selected_weight = float16_t(0);
        sum_weight = float16_t(0);

        ivec3 tex_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION<<cascade, ivec3(cascade_coord), base_cell);
        f16vec4 values = f16vec4(textureGather(radiance_cascades_read[cascade], vec3(tex_coord.xy + 1.0, tex_coord.z)).zxwy);
#ifdef RC_DEFENSIVE
        f16vec4 r = sort_f16vec4(values);
        float16_t base = dot(r, clamp(visibility-f16vec4(0.75,0.50,0.25,0.00), f16vec4(0.0), f16vec4(0.25)));
        values = max(selected_value - base * total_visibility, float16_t(0)) + values * total_visibility * visibility;
        total_visibility *= visibility;
#else
        total_visibility *= visibility;
        values = max(selected_value - dot(values, f16vec4(0.25)) * total_visibility, float16_t(0)) + values * total_visibility;
#endif

        ivec3 sel_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION<<cascade, ivec3(cascade_coord), itex_coord);
        visibility = float16_t(1.0 - texelFetch(radiance_cascades_visibility[cascade], sel_coord, 0).r);

        f16vec4 diffuse;
        f16vec4 specular;
        integrate_quad_half_precision(
            tangent,
            bitangent,
            hnormal,
            ltc_transform,
            inv_probe_resolution,
            f16vec2(base_cell) * inv_probe_resolution,
            diffuse,
            specular
        );
        f16vec4 contrib = values * (flip_fresnel * diffuse + specular_amplitude * specular);
        rc_wrs_pdf(base_cell, itex_coord, contrib, sum_weight, selected_weight, values, selected_value, diffuse, specular, selected_brdf);

        pdf *= sum_weight == float16_t(0) ? 1.0 : float(selected_weight) / float(sum_weight);
    }

    return rc_texel_pdf(
        tex_coord,
        dir*tbn,
        probe_resolution,
        ltc_transform,
        roughness,
        flip_fresnel,
        specular_amplitude,
        selected_brdf,
        pdf
#ifdef RC_SAMPLE_SINGLE_LOBE
        , sampled_lobe
#endif
    );
}

#endif

#endif
