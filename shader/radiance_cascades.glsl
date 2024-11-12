#ifndef RADIANCE_CASCADES_GLSL
#define RADIANCE_CASCADES_GLSL
#include "math.glsl"

#ifdef RADIANCE_CASCADES_SET
layout(set=RADIANCE_CASCADES_SET, binding = 0) uniform sampler2DArray radiance_cascades[];
#endif


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

void rc_mapping(ivec2 p, int cascade, inout rc_spherical_triangle t, inout float solid_angle)
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

    solid_angle = M_PI;

    for(int i = cascade-1; i >= 0; --i)
    {
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

void rc_inv_mapping(vec3 dir, int cascade, inout ivec2 p, inout rc_spherical_triangle t, inout float solid_angle)
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

    float ccw = 1.0;
    for(int i = cascade-1; i >= 0; --i)
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
}

#endif
