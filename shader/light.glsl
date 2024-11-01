#ifndef LIGHT_GLSL
#define LIGHT_GLSL
#include "material.glsl"
#include "color.glsl"
#include "camera.glsl"

struct directional_light
{
    vec3 color;
    int shadow_map_index;
    vec3 dir;
    float dir_cutoff;
};

struct point_light
{
    vec3 color;
    vec3 dir;
    vec3 pos;
    float radius;
    float dir_cutoff;
    float dir_falloff;
    float cutoff_radius;
    float spot_radius;
    int shadow_map_index;
    int padding;
};

struct tri_light
{
    vec3 pos[3];
    uint emission_factor; // R9G9B9E5
    uint instance_id;
    uint primitive_id;
    uint uv[3];
    int emission_tex_id;
};

void random_sample_point_light(vec3 world_pos, float u, int item_count, out float selected_weight, out int selected_index)
{
    selected_index = clamp(int(u * item_count), 0, item_count-1);
    selected_weight = max(item_count, 1);
}

float get_spotlight_intensity(point_light l, vec3 dir)
{
    if(l.dir_falloff > 0)
    {
        float cutoff = dot(dir, -l.dir);
        cutoff = cutoff > l.dir_cutoff ?
            1.0f-pow(
                max(1.0f-cutoff, 0.0f)/(1.0f-l.dir_cutoff), l.dir_falloff
            ) : 0.0f;
        return cutoff;
    }
    else return 1.0f;
}

void get_point_light_info(
    point_light l,
    vec3 pos,
    out vec3 dir,
    out float dist,
    out vec3 color
){
    dir = l.pos - pos;
    float dist2 = dot(dir, dir);
    dist = sqrt(dist2);
    dir /= dist;

    color = get_spotlight_intensity(l, dir) * l.color/dist2;
}

void sample_point_light(
    point_light pl,
    vec2 u,
    vec3 pos,
    out vec3 out_dir,
    out float out_length,
    out vec3 color,
    out float pdf
){
    vec3 dir = pos - pl.pos;
    float dist2 = dot(dir, dir);
    float k = 1.0f - pl.radius * pl.radius / dist2;
    float dir_cutoff = k > 0 ? sqrt(k) : -1.0f;
    out_dir = sample_cone(u, -normalize(dir), dir_cutoff);

    float b = dot(dir, out_dir);
    out_length = -b - sqrt(max(b * b - dist2 + pl.radius * pl.radius, 0.0f));

    color = get_spotlight_intensity(pl, normalize(-dir)) * pl.color;

    if(pl.radius == 0.0f)
    {
        // We mark infinite PDFs with the minus sign on the NEE side.
        pdf = -dist2;
    }
    else
    {
        color /= pl.radius * pl.radius * M_PI;
        pdf = 1 / (2.0f * M_PI * (1.0f - dir_cutoff));
    }
}

float sample_point_light_pdf(point_light pl, vec3 pos)
{
    vec3 dir = pos - pl.pos;
    float dist2 = dot(dir, dir);
    float k = 1.0f - pl.radius * pl.radius / dist2;
    float dir_cutoff = k > 0 ? sqrt(k) : -1.0f;

    if(pl.radius == 0.0f) return 0;
    else return 1 / (2.0f * M_PI * (1.0f - dir_cutoff));
}

void sample_directional_light(
    directional_light dl,
    vec2 u,
    out vec3 out_dir,
    out vec3 color,
    out float pdf
){
    out_dir = sample_cone(u, -dl.dir, dl.dir_cutoff);
    pdf = dl.dir_cutoff >= 1.0f ? -1.0f : 1.0f / (2.0f * M_PI * (1.0f - dl.dir_cutoff));
    color = pdf > 0 ? dl.color * pdf : dl.color;
}

float sample_directional_light_pdf(directional_light dl)
{
    return dl.dir_cutoff >= 1.0f ? 0.0f : 1.0f / (2.0f * M_PI * (1.0f - dl.dir_cutoff));
}

#if defined(TRI_LIGHT_SAMPLE_AREA)
float sample_triangle_light_pdf(vec3 P, vec3 A, vec3 B, vec3 C)
{
    return triangle_area_pdf(P, A, B, C);
}

vec3 sample_triangle_light(vec2 u, vec3 A, vec3 B, vec3 C, out float pdf)
{
    vec3 P = sample_triangle_area(u, A, B, C);
    pdf = triangle_area_pdf(P, A, B, C);
    return normalize(P);
}
#elif defined(TRI_LIGHT_SAMPLE_SOLID_ANGLE)
float sample_triangle_light_pdf(vec3 P, vec3 A, vec3 B, vec3 C)
{
    return 1.0f / spherical_triangle_solid_angle(normalize(A), normalize(B), normalize(C));
}

vec3 sample_triangle_light(vec2 u, vec3 A, vec3 B, vec3 C, out float pdf)
{
    return sample_spherical_triangle(u, A, B, C, pdf);
}
#else
float sample_triangle_light_pdf(vec3 P, vec3 A, vec3 B, vec3 C)
{
    float solid_angle = spherical_triangle_solid_angle(normalize(A), normalize(B), normalize(C));
    return solid_angle > 1e-6 ? 1.0f / solid_angle : triangle_area_pdf(P, A, B, C);
}

vec3 sample_triangle_light(vec2 u, vec3 A, vec3 B, vec3 C, out float pdf)
{
    float solid_angle = spherical_triangle_solid_angle(normalize(A), normalize(B), normalize(C));
    if(solid_angle > 1e-6)
    {
        // Let's just hope that the optimizer spots us calculating the same solid
        // angle twice (that happens inside sample_spherical_triangle)
        return sample_spherical_triangle(u, A, B, C, pdf);
    }
    else
    {
        vec3 P = sample_triangle_area(u, A, B, C);
        pdf = triangle_area_pdf(P, A, B, C);
        return normalize(P);
    }
}
#endif

#endif
