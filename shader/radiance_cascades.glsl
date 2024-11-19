#ifndef RADIANCE_CASCADES_GLSL
#define RADIANCE_CASCADES_GLSL
#include "math.glsl"

struct rc_spherical_triangle
{
    vec3 a, b, c;
    float cos_a, cos_b, cos_c;
    float sin_solid_angle;
    float cos_solid_angle;
};

void rc_reorder_triangle(inout rc_spherical_triangle v)
{
    if(v.cos_a < v.cos_b && v.cos_a < v.cos_c)
    { // This should be the hot path
        return;
    }
    else if(v.cos_b < v.cos_c)
    {
        v = rc_spherical_triangle(
            v.b, v.c, v.a,
            v.cos_b,
            v.cos_c,
            v.cos_a,
            v.sin_solid_angle,
            v.cos_solid_angle
        );
    }
    else
    {
        v = rc_spherical_triangle(
            v.c, v.a, v.b,
            v.cos_c,
            v.cos_a,
            v.cos_b,
            v.sin_solid_angle,
            v.cos_solid_angle
        );
    }
}

void rc_split_triangle_preamble(rc_spherical_triangle v, inout vec3 split_point, inout float cos_z, inout float cos_az)
{
    float sin_a2 = 1.0f - v.cos_a * v.cos_a;
    float sin_c2 = 1.0f - v.cos_c * v.cos_c;

    float cos_B = v.cos_b - v.cos_a * v.cos_c;
    float sin_B = sqrt(max(sin_a2 * sin_c2 - cos_B * cos_B, 0.0f));
    float cos_D = v.cos_solid_angle * cos_B + v.sin_solid_angle * sin_B;
    float cos_cp1 = 1.0f + v.cos_c;
    float sin_za = (cos_B - cos_D) * cos_cp1 / (cos_B * cos_D + sin_a2 * cos_cp1 * (v.cos_solid_angle * v.cos_c - 1.0f));
    cos_z = sqrt(1.0f - sin_za * sin_za * sin_a2);

    split_point = v.c * sin_za + v.b * (cos_z - v.cos_a * sin_za);
    cos_az = cos_z * v.cos_a + sin_za * sin_a2;
}

void rc_build_half_triangle(inout rc_spherical_triangle v, bool left, vec3 split_point, float cos_z, float cos_az)
{
    v.c = left ? v.c : v.b;
    v.b = v.a;
    v.a = split_point;

    v.cos_a = left ? v.cos_b : v.cos_c;
    v.cos_b = left ? cos_az : cos_z;
    v.cos_c = clamp(dot(v.a, v.b), -1.0f, 1.0f);
}

// In the result triangle, A and B are the splitting edge. 'left' controls
// whether the new C is the original C or B.
void rc_split_triangle(inout rc_spherical_triangle v, bool left)
{
    vec3 vD;
    float cos_z, cos_az;
    rc_split_triangle_preamble(v, vD, cos_z, cos_az);
    rc_build_half_triangle(v, left, vD, cos_z, cos_az);
}

bool rc_select_subtriangle(inout rc_spherical_triangle v, vec3 dir, float ccw)
{
    vec3 vD;
    float cos_z, cos_az;
    rc_split_triangle_preamble(v, vD, cos_z, cos_az);
    bool left = ccw * dot(cross(vD, v.a), dir) < 0.0f; // TODO: Check sign correctness
    rc_build_half_triangle(v, left, vD, cos_z, cos_az);
    return left;
}

void rc_mapping(ivec2 p, int cascade, inout rc_spherical_triangle t, inout rc_spherical_triangle prev_t, inout float solid_angle)
{
    ivec2 rounded = p >> cascade;
    p -= rounded << cascade;
    t.cos_a = -1.0f/3.0f;
    t.cos_b = -1.0f/3.0f;
    t.cos_c = -1.0f/3.0f;

    // TODO: Make this into math?
    // Make sure winding order is the same for all triangles!
    if(rounded.x == 0 && rounded.y == 0)
    {
        t.a = normalize(vec3(-1,-1,1));
        t.b = normalize(vec3(1,-1,-1));
        t.c = normalize(vec3(1,1,1));
    }
    else if(rounded.x == 1 && rounded.y == 0)
    {
        t.a = normalize(vec3(-1,1,-1));
        t.b = normalize(vec3(-1,-1,1));
        t.c = normalize(vec3(1,1,1));
    }
    else if(rounded.x == 0 && rounded.y == 1)
    {
        t.a = normalize(vec3(-1,-1,1));
        t.b = normalize(vec3(-1,1,-1));
        t.c = normalize(vec3(1,-1,-1));
    }
    else
    {
        t.a = normalize(vec3(1,1,1));
        t.b = normalize(vec3(1,-1,-1));
        t.c = normalize(vec3(-1,1,-1));
    }
    prev_t = t;

    solid_angle = M_PI;

    for(int i = cascade-1; i >= 0; --i)
    {
        prev_t = t;
        rounded = p >> i;
        p -= rounded << i;

        // TODO: Could tabulate these somehow?
        solid_angle *= 0.5f;
        t.sin_solid_angle = sin(solid_angle);
        t.cos_solid_angle = cos(solid_angle);
        rc_reorder_triangle(t);
        rc_split_triangle(t, rounded.x != 0);

        solid_angle *= 0.5f;
        t.sin_solid_angle = sin(solid_angle);
        t.cos_solid_angle = cos(solid_angle);
        rc_reorder_triangle(t);
        rc_split_triangle(t, rounded.y != 0);
    }
}

// Does cascade 0
void rc_inv_mapping_init(vec3 dir, out ivec2 p, out float ccw, out rc_spherical_triangle t, out float solid_angle)
{
    t.cos_a = -1.0f/3.0f;
    t.cos_b = -1.0f/3.0f;
    t.cos_c = -1.0f/3.0f;
    p = ivec2(0);

    if(dir.x + dir.z > 0.0 && dir.y - dir.x < 0.0 && dir.z - dir.y > 0.0)
    {
        t.a = normalize(vec3(-1,-1,1));
        t.b = normalize(vec3(1,-1,-1));
        t.c = normalize(vec3(1,1,1));
        p = ivec2(0,0);
    }
    else if(dir.y + dir.z > 0.0 && dir.x - dir.z < 0.0)
    {
        t.a = normalize(vec3(-1,1,-1));
        t.b = normalize(vec3(-1,-1,1));
        t.c = normalize(vec3(1,1,1));
        p = ivec2(1,0);
    }
    else if(dir.x + dir.y < 0.0)
    {
        t.a = normalize(vec3(-1,-1,1));
        t.b = normalize(vec3(-1,1,-1));
        t.c = normalize(vec3(1,-1,-1));
        p = ivec2(0,1);
    }
    else
    {
        t.a = normalize(vec3(1,1,1));
        t.b = normalize(vec3(1,-1,-1));
        t.c = normalize(vec3(-1,1,-1));
        p = ivec2(1,1);
    }

    solid_angle = M_PI;
    ccw = 1.0;
}

void rc_inv_mapping_step(vec3 dir, inout ivec2 p, inout float ccw, inout rc_spherical_triangle t, inout float solid_angle)
{
    p <<= 1;

    solid_angle *= 0.5f;
    t.sin_solid_angle = sin(solid_angle);
    t.cos_solid_angle = cos(solid_angle);
    rc_reorder_triangle(t);
    if(rc_select_subtriangle(t, dir, ccw))
    {
        p.x++;
        ccw = -ccw;
    }

    solid_angle *= 0.5f;
    t.sin_solid_angle = sin(solid_angle);
    t.cos_solid_angle = cos(solid_angle);
    rc_reorder_triangle(t);
    if(rc_select_subtriangle(t, dir, ccw))
    {
        p.y++;
        ccw = -ccw;
    }
}

void rc_inv_mapping(vec3 dir, int cascade, inout ivec2 p, inout rc_spherical_triangle t, inout float solid_angle)
{
    float ccw;
    rc_inv_mapping_init(dir, p, ccw, t, solid_angle);
    for(int i = cascade-1; i >= 0; --i)
        rc_inv_mapping_step(dir, p, ccw, t, solid_angle);
}

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
            // TODO: Maybe could fall back to envmap?
            return vec3(1);
        }
        origin = origin + dir * t;
    }

    // 0-1 inside cascade volume
    vec3 fcoord = (origin - aabb_min) / (aabb_max - aabb_min);

    ivec3 cascade_size = radiance_cascade_metadata.size.xyz;
    ivec2 probe_resolution = ivec2(2,2);
    // TODO: Maybe make this a specialization constant?
    int cascade_count = radiance_cascade_metadata.size.w;
    ivec3 cascade_coord = clamp(ivec3(
        fcoord * cascade_size
    ), ivec3(0), ivec3(cascade_size-1));

    vec4 sum = vec4(0,0,0,1);

    float ccw;
    ivec2 p;
    float solid_angle;
    rc_spherical_triangle st;
    rc_inv_mapping_init(dir, p, ccw, st, solid_angle);

    ivec3 tex_coord = cascade_coord * ivec3(probe_resolution, 1) + ivec3(p, 0);
    vec4 col = texelFetch(radiance_cascades[0], tex_coord, 0);
    sum.rgb += col.rgb * sum.a;
    sum.a *= 1.0f-col.a;

    for(int cascade = 1; sum.a > 0 && cascade < cascade_count; ++cascade)
    {
        probe_resolution *= 2;
        cascade_size /= 2;
        cascade_coord /= 2;
        rc_inv_mapping_step(dir, p, ccw, st, solid_angle);
        tex_coord = cascade_coord * ivec3(probe_resolution, 1) + ivec3(p, 0);
        vec4 col = texelFetch(radiance_cascades[cascade], tex_coord, 0);
        sum.rgb += col.rgb * sum.a;
        sum.a *= 1.0f-col.a;
    }
    return sum.rgb;
}
#endif

#endif
