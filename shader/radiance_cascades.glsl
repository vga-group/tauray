#ifndef RADIANCE_CASCADES_GLSL
#define RADIANCE_CASCADES_GLSL
#include "math.glsl"
#include "color.glsl"
#include "ltc.glsl"
#define MAX_POLYGON_VERTEX_COUNT 5
#define MIN_POLYGON_VERTEX_COUNT_BEFORE_CLIPPING 4
#include "brdf-area-light-sampling/polygon_clipping.glsl"
#include "brdf-area-light-sampling/polygon_sampling.glsl"

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

    f16vec2 len2 = normal_x * normal_x + normal_y * normal_y + normal_z * normal_z;
    f16vec2 inv_len = inversesqrt(len2);

    normal_x = normal_x * inv_len;
    normal_y = normal_y * inv_len;
    normal_z = normal_z * inv_len;
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

vec3 nearest_dir_on_arc(vec3 ref, vec3 arc_start, vec3 arc_end)
{
    vec3 normal = cross(arc_start, arc_end);
    float len2 = dot(normal, normal);
    vec3 q = normalize(ref * len2 - normal * dot(normal, ref));

    vec3 delta = arc_start - arc_end;
    float d_arc2 = dot(delta, delta);
    delta = arc_start - q;
    float d_start2 = dot(delta, delta);
    delta = arc_end - q;
    float d_end2 = dot(delta, delta);

    if (d_start2 < d_arc2 && d_end2 < d_arc2)
        return q;
    return d_start2 > d_end2 ? arc_end : arc_start;
}

// poly must be convex, clockwise and vertices normalized.
vec3 maximal_dir_on_texel(vec3 poly[4], vec3 target)
{
    // Check if peak is inside polygon.
    if (
        dot(cross(poly[0], poly[1]), target) > 0 &&
        dot(cross(poly[1], poly[2]), target) > 0 &&
        dot(cross(poly[2], poly[3]), target) > 0 &&
        dot(cross(poly[3], poly[0]), target) > 0
    ) return target;

    vec3 maxq = poly[0];
    float maxd = dot(poly[0], target);
    vec3 q = nearest_dir_on_arc(target, poly[0], poly[1]);
    float d = dot(q, target);
    if(d > maxd) { maxq = q; maxd = d; }

    q = nearest_dir_on_arc(target, poly[1], poly[2]);
    d = dot(q, target);
    if(d > maxd) { maxq = q; maxd = d; }

    q = nearest_dir_on_arc(target, poly[2], poly[3]);
    d = dot(q, target);
    if(d > maxd) { maxq = q; maxd = d; }

    q = nearest_dir_on_arc(target, poly[3], poly[0]);
    d = dot(q, target);

    if(d > maxd) { maxq = q; maxd = d; }
    return maxq;
}

void nearest_dir_on_arc_half_precision(
    f16vec2 ref_x,
    f16vec2 ref_y,
    f16vec2 ref_z,
    f16vec2 arc_start_x,
    f16vec2 arc_start_y,
    f16vec2 arc_start_z,
    f16vec2 arc_end_x,
    f16vec2 arc_end_y,
    f16vec2 arc_end_z,
    out f16vec2 nearest_x,
    out f16vec2 nearest_y,
    out f16vec2 nearest_z
){
    //vec3 normal = cross(arc_start, arc_end);
    f16vec2 normal_x = arc_start_y * arc_end_z - arc_end_y * arc_start_z;
    f16vec2 normal_y = arc_start_z * arc_end_x - arc_end_z * arc_start_x;
    f16vec2 normal_z = arc_start_x * arc_end_y - arc_end_x * arc_start_y;

    f16vec2 len2 = normal_x * normal_x + normal_y * normal_y + normal_z * normal_z;
    f16vec2 ndotr = normal_x * ref_x + normal_y * ref_y + normal_z * ref_z;

    f16vec2 q_x = ref_x * len2 - normal_x * ndotr;
    f16vec2 q_y = ref_y * len2 - normal_y * ndotr;
    f16vec2 q_z = ref_z * len2 - normal_z * ndotr;
    f16vec2 inv_len = inversesqrt(q_x*q_x + q_y*q_y + q_z*q_z);
    q_x *= inv_len;
    q_y *= inv_len;
    q_z *= inv_len;

    f16vec2 delta_x = arc_start_x - arc_end_x;
    f16vec2 delta_y = arc_start_y - arc_end_y;
    f16vec2 delta_z = arc_start_z - arc_end_z;
    f16vec2 d_arc2 = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;

    delta_x = arc_start_x - q_x;
    delta_y = arc_start_y - q_y;
    delta_z = arc_start_z - q_z;
    f16vec2 d_start2 = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;

    delta_x = arc_end_x - q_x;
    delta_y = arc_end_y - q_y;
    delta_z = arc_end_z - q_z;
    f16vec2 d_end2 = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;

    bvec2 q_cond = and(lessThan(d_start2, d_arc2), lessThan(d_end2, d_arc2));

    bvec2 end_cond = greaterThan(d_start2, d_end2);
    nearest_x = mix(mix(arc_start_x, arc_end_x, end_cond), q_x, q_cond);
    nearest_y = mix(mix(arc_start_y, arc_end_y, end_cond), q_y, q_cond);
    nearest_z = mix(mix(arc_start_z, arc_end_z, end_cond), q_z, q_cond);
}

f16vec2 dotcross(
    f16vec2 x1,
    f16vec2 y1,
    f16vec2 z1,
    f16vec2 x2,
    f16vec2 y2,
    f16vec2 z2,
    f16vec2 tx,
    f16vec2 ty,
    f16vec2 tz
){
    return
        (y1 * z2 - y2 * z1) * tx +
        (z1 * x2 - z2 * x1) * ty +
        (x1 * y2 - x2 * y1) * tz;
}

void maximal_dir_on_texel_half_precision(
    f16vec2 poly_x[4],
    f16vec2 poly_y[4],
    f16vec2 poly_z[4],
    f16vec2 target_x,
    f16vec2 target_y,
    f16vec2 target_z,
    out f16vec2 result_x,
    out f16vec2 result_y,
    out f16vec2 result_z
){
    nearest_dir_on_arc_half_precision(
        target_x,
        target_y,
        target_z,
        poly_x[0],
        poly_y[0],
        poly_z[0],
        poly_x[1],
        poly_y[1],
        poly_z[1],
        result_x, result_y, result_z
    );
    f16vec2 maxd = result_x * target_x + result_y * target_y + result_z * target_z;

    f16vec2 q_x;
    f16vec2 q_y;
    f16vec2 q_z;

    nearest_dir_on_arc_half_precision(
        target_x,
        target_y,
        target_z,
        poly_x[1],
        poly_y[1],
        poly_z[1],
        poly_x[2],
        poly_y[2],
        poly_z[2],
        q_x, q_y, q_z
    );
    f16vec2 d = q_x * target_x + q_y * target_y + q_z * target_z;
    result_x = mix(result_x, q_x, greaterThan(d, maxd));
    result_y = mix(result_y, q_y, greaterThan(d, maxd));
    result_z = mix(result_z, q_z, greaterThan(d, maxd));
    maxd = max(maxd, d);

    nearest_dir_on_arc_half_precision(
        target_x,
        target_y,
        target_z,
        poly_x[2],
        poly_y[2],
        poly_z[2],
        poly_x[3],
        poly_y[3],
        poly_z[3],
        q_x, q_y, q_z
    );
    d = q_x * target_x + q_y * target_y + q_z * target_z;
    result_x = mix(result_x, q_x, greaterThan(d, maxd));
    result_y = mix(result_y, q_y, greaterThan(d, maxd));
    result_z = mix(result_z, q_z, greaterThan(d, maxd));
    maxd = max(maxd, d);

    nearest_dir_on_arc_half_precision(
        target_x,
        target_y,
        target_z,
        poly_x[3],
        poly_y[3],
        poly_z[3],
        poly_x[0],
        poly_y[0],
        poly_z[0],
        q_x, q_y, q_z
    );
    d = q_x * target_x + q_y * target_y + q_z * target_z;
    result_x = mix(result_x, q_x, greaterThan(d, maxd));
    result_y = mix(result_y, q_y, greaterThan(d, maxd));
    result_z = mix(result_z, q_z, greaterThan(d, maxd));
    maxd = max(maxd, d);

    bvec2 dc0 = greaterThan(dotcross(
        poly_x[0], poly_y[0], poly_z[0],
        poly_x[1], poly_y[1], poly_z[1],
        target_x, target_y, target_z
    ), f16vec2(0));
    bvec2 dc1 = greaterThan(dotcross(
        poly_x[1], poly_y[1], poly_z[1],
        poly_x[2], poly_y[2], poly_z[2],
        target_x, target_y, target_z
    ), f16vec2(0));
    bvec2 dc2 = greaterThan(dotcross(
        poly_x[2], poly_y[2], poly_z[2],
        poly_x[3], poly_y[3], poly_z[3],
        target_x, target_y, target_z
    ), f16vec2(0));
    bvec2 dc3 = greaterThan(dotcross(
        poly_x[3], poly_y[3], poly_z[3],
        poly_x[0], poly_y[0], poly_z[0],
        target_x, target_y, target_z
    ), f16vec2(0));
    bvec2 pass = and(and(dc0, dc1), and(dc2, dc3));

    result_x = mix(result_x, target_x, pass);
    result_y = mix(result_y, target_y, pass);
    result_z = mix(result_z, target_z, pass);
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

    ivec2 p = octahedral_wrap(ivec2(floor(uv * probe_resolution)), ivec2(probe_resolution));
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

        ivec2 p = octahedral_wrap(ivec2(floor(uv * probe_resolution)), ivec2(probe_resolution));
        vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));
        ivec3 tex_coord = get_cascade_layout(cascade_size, probe_resolution, ivec3(cascade_coord), p);
        col = texelFetch(radiance_cascades[cascade], tex_coord, 0).r;
        vis = texelFetch(radiance_cascades_visibility[cascade], tex_coord, 0).r;
        sum.rgb += col.rrr * vis * sum.a;
        sum.a *= 1.0f-vis;
    }
    return sum.rgb;
}

vec3 get_texel_corner(ivec2 texel, vec2 corner, float inv_probe_resolution, mat3 tbn)
{
    vec3 dir = radiance_cascade_probe_mapping((vec2(texel) + corner) * inv_probe_resolution) * tbn;
    dir.z = max(dir.z, 0.0);
    return dir;
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
    inout f16vec2 result_z
){
    get_unclamped_texel_corner(
        base,
        corner_x,
        corner_y,
        inv_probe_resolution,
        tangent,
        bitangent,
        normal,
        result_x,
        result_y,
        result_z
    );

    result_z = max(result_z, f16vec2(0.0));
}

#ifndef printf
#define printf
#endif

vec3 rc_texel_sample(
    inout uint seed,
    ivec2 selected_cell,
    int probe_resolution,
    float16_t inv_probe_resolution,
    f16vec3 tangent,
    f16vec3 bitangent,
    f16vec3 normal,
    f16vec3 ltc_transform,
    inout float pdf
){
    /*
    // [v00]----[v10]
    //   |   <-   |
    //   | v    ^ |
    //   |   ->   |
    // [v01]----[v11]
    f16vec2 h00_h11_x, h00_h11_y, h00_h11_z;
    get_unclamped_texel_corner(
        f16vec2(selected_cell) * inv_probe_resolution, f16vec2(0,1), f16vec2(0,1),
        inv_probe_resolution,
        tangent, bitangent, normal,
        h00_h11_x, h00_h11_y, h00_h11_z
    );
    f16vec2 h01_h10_x, h01_h10_y, h01_h10_z;
    get_unclamped_texel_corner(
        f16vec2(selected_cell) * inv_probe_resolution, f16vec2(0,1), f16vec2(1,0),
        inv_probe_resolution,
        tangent, bitangent, normal,
        h01_h10_x, h01_h10_y, h01_h10_z
    );

    vec3 polygon[5] = {
        vec3(h00_h11_x[0], h00_h11_y[0], h00_h11_z[0]),
        vec3(h01_h10_x[0], h01_h10_y[0], h01_h10_z[0]),
        vec3(h00_h11_x[1], h00_h11_y[1], h00_h11_z[1]),
        vec3(h01_h10_x[1], h01_h10_y[1], h01_h10_z[1]),
        vec3(0)
    };
    uint vertex_count = clip_polygon(4, polygon);
    projected_solid_angle_polygon_t diffuse_poly = prepare_projected_solid_angle_polygon_sampling(vertex_count, polygon);

    vec2 uv = vec2(
        generate_single_uniform_random_fast(seed),
        generate_single_uniform_random_fast(seed)
    );

    vec3 r = sample_projected_solid_angle_polygon(diffuse_poly, uv);

    //if (!(diffuse_poly.projected_solid_angle > 0))
    //{
    //    printf("zero (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f) (%f, %u, %u, %e)",
    //        polygon[0].x, polygon[0].y, polygon[0].z,
    //        polygon[1].x, polygon[1].y, polygon[1].z,
    //        polygon[2].x, polygon[2].y, polygon[2].z,
    //        polygon[3].x, polygon[3].y, polygon[3].z,
    //        diffuse_poly.projected_solid_angle,
    //        vertex_count,
    //        (polygon[0].z>0?1:0)+ (polygon[1].z>0?1:0)+ (polygon[2].z>0?1:0)+ (polygon[3].z>0?1:0),
    //        pdf
    //    );
    //}
    if (diffuse_poly.projected_solid_angle <= 0 || r.z == 0)
        pdf = -1.0f;
    else
        pdf *= r.z / diffuse_poly.projected_solid_angle;
    vec3 dir = vec3(
        r.x * tangent.x + r.y * bitangent.x + r.z * normal.x,
        r.x * tangent.y + r.y * bitangent.y + r.z * normal.y,
        r.x * tangent.z + r.y * bitangent.z + r.z * normal.z
    );
    return dir;
    */

    vec2 uv = vec2(
        generate_single_uniform_random_fast(seed),
        generate_single_uniform_random_fast(seed)
    );
    uv = (vec2(selected_cell) + uv) * inv_probe_resolution;

    pdf *= probe_resolution * probe_resolution * 0.25f * octahedral_mapping_abs_jacobian_det(uv*2.0-1.0);

    vec3 dir = radiance_cascade_probe_mapping(uv);
    return dir;
}

float rc_texel_pdf(
    ivec2 selected_cell,
    vec2 uv,
    vec3 tdir,
    int probe_resolution,
    float16_t inv_probe_resolution,
    f16vec3 tangent,
    f16vec3 bitangent,
    f16vec3 normal,
    f16vec3 ltc_transform,
    float pdf
){
    // [v00]----[v10]
    //   |   <-   |
    //   | v    ^ |
    //   |   ->   |
    // [v01]----[v11]
    /*
    f16vec2 h00_h11_x, h00_h11_y, h00_h11_z;
    get_unclamped_texel_corner(
        f16vec2(selected_cell) * inv_probe_resolution, f16vec2(0,1), f16vec2(0,1),
        inv_probe_resolution,
        tangent, bitangent, normal,
        h00_h11_x, h00_h11_y, h00_h11_z
    );
    f16vec2 h01_h10_x, h01_h10_y, h01_h10_z;
    get_unclamped_texel_corner(
        f16vec2(selected_cell) * inv_probe_resolution, f16vec2(0,1), f16vec2(1,0),
        inv_probe_resolution,
        tangent, bitangent, normal,
        h01_h10_x, h01_h10_y, h01_h10_z
    );

    vec3 polygon[5] = {
        vec3(h00_h11_x[0], h00_h11_y[0], h00_h11_z[0]),
        vec3(h01_h10_x[0], h01_h10_y[0], h01_h10_z[0]),
        vec3(h00_h11_x[1], h00_h11_y[1], h00_h11_z[1]),
        vec3(h01_h10_x[1], h01_h10_y[1], h01_h10_z[1]),
        vec3(0)
    };
    uint vertex_count = clip_polygon(4, polygon);
    projected_solid_angle_polygon_t diffuse_poly = prepare_projected_solid_angle_polygon_sampling(vertex_count, polygon);

    if (diffuse_poly.projected_solid_angle == 0)
        return 0;
    pdf *= tdir.z / diffuse_poly.projected_solid_angle;
    return pdf;
    */
    pdf *= probe_resolution * probe_resolution * 0.25f * octahedral_mapping_abs_jacobian_det(uv*2.0-1.0);
    return pdf;
}

f16vec4 integrate_quad_half_precision(
    f16vec3 tangent,
    f16vec3 bitangent,
    f16vec3 normal,
    f16vec3 ltc_transform,
    float16_t inv_probe_resolution,
    f16vec2 base,
    float16_t flip_fresnel,
    float16_t specular_amplitude
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

    // Deal with rounding errors.
    //f16vec4 sum_contrib = flip_fresnel * max(diff_z, f16vec4(0.0)) + specular_amplitude * max(spec_z, f16vec4(0.0));
    f16vec4 sum_contrib = max(diff_z, f16vec4(0.0));

    // Mask out quads that are below the horizon.
    // These get included often due to rounding errors in our half-precision
    // approximate math.
    if (h11_h11_z.x == float16_t(0))
    {
        if (h00_h22_z.x == float16_t(0) && h10_h12_z.x == float16_t(0) && h21_h01_z.y == float16_t(0))
            sum_contrib.z = float16_t(0);
        if (h10_h12_z.x == float16_t(0) && h20_h02_z.x == float16_t(0) && h21_h01_z.x == float16_t(0))
            sum_contrib.x = float16_t(0);
        if (h21_h01_z.y == float16_t(0) && h20_h02_z.y == float16_t(0) && h10_h12_z.y == float16_t(0))
            sum_contrib.y = float16_t(0);
        if (h10_h12_z.y == float16_t(0) && h21_h01_z.x == float16_t(0) && h00_h22_z.y == float16_t(0))
            sum_contrib.w = float16_t(0);
    }

    return sum_contrib;
}

f16vec4 max_brdf_contrib_half_precision(
    f16vec3 tangent,
    f16vec3 bitangent,
    f16vec3 normal,
    f16vec3 ltc_transform,
    float16_t inv_probe_resolution,
    f16vec2 base,
    float16_t flip_fresnel,
    float16_t specular_amplitude
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

#if 1
    f16vec2 rx;
    f16vec2 ry;
    f16vec2 rz;
    maximal_dir_on_texel_half_precision(
        f16vec2[4](h00_h22_x, h10_h12_x, h11_h11_x, h21_h01_x.yx),
        f16vec2[4](h00_h22_y, h10_h12_y, h11_h11_y, h21_h01_y.yx),
        f16vec2[4](h00_h22_z, h10_h12_z, h11_h11_z, h21_h01_z.yx),
        f16vec2(0,0),
        f16vec2(0,0),
        f16vec2(1,1),
        rx, ry, rz
    );
    diff_z.zw = max(rz, f16vec2(0));
    maximal_dir_on_texel_half_precision(
        f16vec2[4](h10_h12_x, h20_h02_x, h21_h01_x, h11_h11_x),
        f16vec2[4](h10_h12_y, h20_h02_y, h21_h01_y, h11_h11_y),
        f16vec2[4](h10_h12_z, h20_h02_z, h21_h01_z, h11_h11_z),
        f16vec2(0,0),
        f16vec2(0,0),
        f16vec2(1,1),
        rx, ry, rz
    );
    diff_z.xy = max(rz, f16vec2(0));

    // TODO Specular. Use LTC max dir.
#else
    vec3 h00 = vec3(h00_h22_x[0], h00_h22_y[0], h00_h22_z[0]);
    vec3 h22 = vec3(h00_h22_x[1], h00_h22_y[1], h00_h22_z[1]);
    vec3 h10 = vec3(h10_h12_x[0], h10_h12_y[0], h10_h12_z[0]);
    vec3 h12 = vec3(h10_h12_x[1], h10_h12_y[1], h10_h12_z[1]);
    vec3 h20 = vec3(h20_h02_x[0], h20_h02_y[0], h20_h02_z[0]);
    vec3 h02 = vec3(h20_h02_x[1], h20_h02_y[1], h20_h02_z[1]);
    vec3 h21 = vec3(h21_h01_x[0], h21_h01_y[0], h21_h01_z[0]);
    vec3 h01 = vec3(h21_h01_x[1], h21_h01_y[1], h21_h01_z[1]);
    vec3 h11 = vec3(h11_h11_x[0], h11_h11_y[0], h11_h11_z[0]);

    diff_z.z = float16_t(max(maximal_dir_on_texel(vec3[4](
        h00,h10,h11,h01
    ), vec3(0,0,1)).z, 0.0f));
    diff_z.x = float16_t(max(maximal_dir_on_texel(vec3[4](
        h10,h20,h21,h11
    ), vec3(0,0,1)).z, 0.0f));
    diff_z.y = float16_t(max(maximal_dir_on_texel(vec3[4](
        h01,h11,h12,h02
    ), vec3(0,0,1)).z, 0.0f));
    diff_z.w = float16_t(max(maximal_dir_on_texel(vec3[4](
        h11,h21,h22,h12
    ), vec3(0,0,1)).z, 0.0f));
#endif

    // Deal with rounding errors.
    //f16vec4 sum_contrib = flip_fresnel * max(diff_z, f16vec4(0.0)) + specular_amplitude * max(spec_z, f16vec4(0.0));
    f16vec4 sum_contrib = max(diff_z, f16vec4(0.0));

    // Mask out quads that are below the horizon.
    // These get included often due to rounding errors in our half-precision
    // approximate math.
    if (h11_h11_z.x == float16_t(0))
    {
        if (h00_h22_z.x == float16_t(0) && h10_h12_z.x == float16_t(0) && h21_h01_z.y == float16_t(0))
            sum_contrib.z = float16_t(0);
        if (h10_h12_z.x == float16_t(0) && h20_h02_z.x == float16_t(0) && h21_h01_z.x == float16_t(0))
            sum_contrib.x = float16_t(0);
        if (h21_h01_z.y == float16_t(0) && h20_h02_z.y == float16_t(0) && h10_h12_z.y == float16_t(0))
            sum_contrib.y = float16_t(0);
        if (h10_h12_z.y == float16_t(0) && h21_h01_z.x == float16_t(0) && h00_h22_z.y == float16_t(0))
            sum_contrib.w = float16_t(0);
    }

    return sum_contrib;
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
        contrib += dot(integrate_quad_half_precision(
            tangent,
            bitangent,
            hnormal,
            ltc_transform,
            inv_probe_resolution,
            f16vec2(x,y) * inv_probe_resolution,
            flip_fresnel,
            specular_amplitude
        ), values);
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
    inout float16_t selected_value
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
    inout float16_t selected_value
){
    f16vec2 weight2 = weights.xy + weights.zw;
    float16_t weight = weight2.x + weight2.y;
    sum_weight += weight;
    ivec2 selected_offset = itex_coord - p;
    selected_weight = selected_offset.y == 0 ?
        (selected_offset.x == 0 ? weights.z : weights.x) :
        (selected_offset.x == 0 ? weights.y : weights.w);
    selected_value = selected_offset.y == 0 ?
        (selected_offset.x == 0 ? values.z : values.x) :
        (selected_offset.x == 0 ? values.y : values.w);
}

vec3 sample_radiance_cascades(uint seed, vec3 origin, vec3 normal, vec3 view, float roughness, float f0, out float pdf)
{
    vec3 aabb_min = radiance_cascade_metadata.aabb_min.xyz;
    vec3 aabb_max = radiance_cascade_metadata.aabb_max.xyz;

    // 0-1 inside cascade volume
    vec3 fcoord = (origin - aabb_min) / (aabb_max - aabb_min);
    int cascade_size = RC_C0_SPATIAL_RESOLUTION;
    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));

    float vdotn = dot(view, normal);
    f16vec3 ltc_transform = f16vec3(ltc_ggx_transform(vdotn, max(roughness, 0.01f)));
    mat3 tbn = create_tangent_space(normal, view);
    f16vec3 tangent = f16vec3(tbn[0]);
    f16vec3 bitangent = f16vec3(tbn[1]);
    f16vec3 hnormal = f16vec3(tbn[2]);
    float16_t fresnel = float16_t(f0 + (1.0 - f0) * ggx_fresnel(vdotn, roughness));
    float16_t specular_amplitude = fresnel + float16_t(f0 * ggx_albedo(vdotn, roughness) - f0);
    float16_t flip_fresnel = float16_t(1.0) - fresnel;

    ivec2 selected_cell = ivec2(0);
    float16_t selected_weight = float16_t(0);
    float16_t selected_value = float16_t(0.0f);
    float16_t sum_weight = float16_t(0.0f);

    float16_t inv_probe_resolution = float16_t(1.0f / RC_C0_ANGULAR_RESOLUTION);

    for(int x = 0; x < RC_C0_ANGULAR_RESOLUTION; x+=2)
    for(int y = 0; y < RC_C0_ANGULAR_RESOLUTION; y+=2)
    {
        ivec2 p = ivec2(x, y);
        ivec3 tex_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), p);

        f16vec4 values = f16vec4(textureGather(radiance_cascades[0], vec3(tex_coord.xy + 1.0, tex_coord.z)).zxwy);
        f16vec4 contrib = values * max_brdf_contrib_half_precision(
            tangent,
            bitangent,
            hnormal,
            ltc_transform,
            inv_probe_resolution,
            f16vec2(x,y) * inv_probe_resolution,
            flip_fresnel,
            specular_amplitude
        );
        rc_wrs_update(seed, p, contrib, sum_weight, selected_weight, selected_cell, values, selected_value);
    }

    ivec3 sel_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), selected_cell);
    float16_t total_visibility = float16_t(1.0 - texelFetch(radiance_cascades_visibility[0], sel_coord, 0).r);

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
        f16vec4 values = f16vec4(textureGather(radiance_cascades[cascade], vec3(tex_coord.xy + 1.0, tex_coord.z)).zxwy);
        values = max(selected_value - dot(values, f16vec4(0.25)) * total_visibility, float16_t(0)) + values * total_visibility;

        f16vec4 contrib = values * max_brdf_contrib_half_precision(
            tangent,
            bitangent,
            hnormal,
            ltc_transform,
            inv_probe_resolution,
            f16vec2(base_cell) * inv_probe_resolution,
            flip_fresnel,
            specular_amplitude
        );
        rc_wrs_update(seed, base_cell, contrib, sum_weight, selected_weight, selected_cell, values, selected_value);

        ivec3 sel_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION<<cascade, ivec3(cascade_coord), selected_cell);
        total_visibility *= float16_t(1.0 - texelFetch(radiance_cascades_visibility[cascade], sel_coord, 0).r);
        if (sum_weight != float16_t(0))
            pdf *= float(selected_weight) / float(sum_weight);
    }

    const int probe_resolution = RC_C0_ANGULAR_RESOLUTION<<(RC_CASCADE_COUNT-1);
    //const int probe_resolution = RC_C0_ANGULAR_RESOLUTION;
    return rc_texel_sample(seed, selected_cell, probe_resolution, inv_probe_resolution, tangent, bitangent, hnormal, ltc_transform, pdf);
}

float radiance_cascades_pdf(vec3 origin, vec3 normal, vec3 view, float roughness, float f0, vec3 dir)
{
    vec3 aabb_min = radiance_cascade_metadata.aabb_min.xyz;
    vec3 aabb_max = radiance_cascade_metadata.aabb_max.xyz;

    // 0-1 inside cascade volume
    vec3 fcoord = (origin - aabb_min) / (aabb_max - aabb_min);

    ivec3 cascade_size = radiance_cascade_metadata.size.xyz;
    int probe_resolution = radiance_cascade_metadata.c0_angular_resolution;

    vec3 cascade_coord = clamp(fcoord * cascade_size, vec3(0.5), vec3(cascade_size-0.5));
    vec2 tex_coord = radiance_cascade_probe_mapping_inverse(dir);

    float vdotn = dot(view, normal);
    f16vec3 ltc_transform = f16vec3(ltc_ggx_transform(vdotn, max(roughness, 0.01f)));
    mat3 tbn = create_tangent_space(normal, view);
    f16vec3 tangent = f16vec3(tbn[0]);
    f16vec3 bitangent = f16vec3(tbn[1]);
    f16vec3 hnormal = f16vec3(tbn[2]);

    float16_t fresnel = float16_t(f0 + (1.0 - f0) * ggx_fresnel(vdotn, roughness));
    float16_t specular_amplitude = fresnel + float16_t(f0 * ggx_albedo(vdotn, roughness) - f0);
    float16_t flip_fresnel = float16_t(1.0) - fresnel;

    float16_t selected_weight = float16_t(0);
    float16_t selected_value = float16_t(0);
    float16_t sum_weight = float16_t(0);

    float16_t inv_probe_resolution = float16_t(1.0f / RC_C0_ANGULAR_RESOLUTION);

    ivec2 itex_coord = ivec2(tex_coord * probe_resolution);

    ivec3 sel_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), itex_coord);
    float16_t total_visibility = float16_t(1.0 - texelFetch(radiance_cascades_visibility[0], sel_coord, 0).r);

    for(int x = 0; x < RC_C0_ANGULAR_RESOLUTION; x+=2)
    for(int y = 0; y < RC_C0_ANGULAR_RESOLUTION; y+=2)
    {
        ivec2 p = ivec2(x, y);
        ivec3 tex_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION, ivec3(cascade_coord), p);
        f16vec4 values = f16vec4(textureGather(radiance_cascades[0], vec3(tex_coord.xy + 1.0, tex_coord.z)).zxwy);

        f16vec4 contrib = values * max_brdf_contrib_half_precision(
            tangent,
            bitangent,
            hnormal,
            ltc_transform,
            inv_probe_resolution,
            f16vec2(x,y) * inv_probe_resolution,
            flip_fresnel,
            specular_amplitude
        );

        rc_wrs_pdf(p, itex_coord, contrib, sum_weight, selected_weight, values, selected_value);
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
        f16vec4 values = f16vec4(textureGather(radiance_cascades[cascade], vec3(tex_coord.xy + 1.0, tex_coord.z)).zxwy);
        values = max(selected_value - dot(values, f16vec4(0.25)) * total_visibility, float16_t(0)) + values * total_visibility;

        ivec3 sel_coord = get_cascade_layout(ivec3(cascade_size), RC_C0_ANGULAR_RESOLUTION<<cascade, ivec3(cascade_coord), itex_coord);
        total_visibility *= float16_t(1.0 - texelFetch(radiance_cascades_visibility[cascade], sel_coord, 0).r);

        f16vec4 contrib = values * max_brdf_contrib_half_precision(
            tangent,
            bitangent,
            hnormal,
            ltc_transform,
            inv_probe_resolution,
            f16vec2(base_cell) * inv_probe_resolution,
            flip_fresnel,
            specular_amplitude
        );
        rc_wrs_pdf(base_cell, itex_coord, contrib, sum_weight, selected_weight, values, selected_value);

        pdf *= sum_weight == float16_t(0) ? 1.0 : float(selected_weight) / float(sum_weight);
    }

    return rc_texel_pdf(itex_coord, tex_coord, dir*tbn, probe_resolution, inv_probe_resolution, tangent, bitangent, hnormal, ltc_transform, pdf);
}

#endif

#endif
