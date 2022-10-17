#ifndef ALIAS_TABLE_GLSL
#define ALIAS_TABLE_GLSL

#include "math.glsl"

// Based on CC0 code from https://gist.github.com/juliusikkala/6c8c186f0150fe877a55cee4d266b1b0
struct alias_table_entry
{
    uint alias_id;
    uint probability;
    float pdf;
    float alias_pdf;
};

float latlong_texel_solid_angle(ivec2 p, ivec2 size)
{
    float y0 = float(p.y) / size.y;
    float y1 = float(p.y+1) / size.y;
    return 2.0f * M_PI * (cos(M_PI * y0) - cos(M_PI * y1)) / float(size.x);
}

int latlong_direction_to_pixel_id(vec3 dir, ivec2 size)
{
    vec2 uv = vec2(atan(dir.z, dir.x) * 0.5f, asin(-dir.y)) / M_PI + 0.5f;
    ivec2 p = ivec2(uv * size + 0.5f);
    return p.x + p.y * size.x;
}

vec3 uv_to_latlong_direction(vec2 uv)
{
    uv = (uv - 0.5f) * M_PI;
    vec3 dir = vec3(cos(2.0f * uv.x), -sin(uv.y), sin(2.0f * uv.x));
    dir.xz *= sqrt(1 - dir.y*dir.y);
    return dir;
}

#endif
