// Code yoinked from Raybase with the author's permission (the author being
// me, check git blame or something :D)
#ifndef TAURAY_RAY_CONE_GLSL
#define TAURAY_RAY_CONE_GLSL

float get_specular_cone_angle(float roughness, float energy_percentage)
{
    return roughness * sqrt(energy_percentage / (1.0f - energy_percentage));
}

// Ray cone implementation based on:
// Improved Shader and Texture Level of Detail Using Ray Cones
// https://www.jcgt.org/published/0010/01/01/paper-lowres.pdf
struct ray_cone
{
    float radius;
    float angle;
};

ray_cone init_pixel_ray_cone(vec4 proj_info, ivec2 p, ivec2 resolution)
{
    if(proj_info.x < 0)
    { // Perspective
        vec2 step_size = proj_info.wz / resolution;
        vec2 a = abs(-proj_info.wz * 0.5f + step_size * vec2(p));
        vec2 b = abs(-proj_info.wz * 0.5f + step_size * vec2(p+1));

        vec2 low = min(a, b);
        vec2 high = max(a, b);

        vec2 angles = atan(high) - atan(low);

        return ray_cone(0, min(angles.x, angles.y));
    }
    else
    { // Ortho
        return ray_cone((proj_info.z / resolution.x + proj_info.w / resolution.y) * 0.25f, 0.0f);
    }
}

void ray_cone_apply_dist(float dist, inout ray_cone rc)
{
    rc.radius += rc.angle * dist;
}

void ray_cone_apply_curvature(float curvature, inout ray_cone rc)
{
    rc.angle += curvature;
}

void ray_cone_apply_roughness(float roughness, inout ray_cone rc)
{
    rc.angle += get_specular_cone_angle(roughness, 0.25f);
}

void ray_cone_gradients(
    ray_cone rc,
    vec3 ray_dir,
    vec3 hit_normal,
    vec3 hit_pos,
    vec2 hit_uv,
    vec3 tri_pos[3],
    vec2 tri_uv[3],
    out vec2 puvdx,
    out vec2 puvdy
){
    vec3 a1 = ray_dir - dot(hit_normal, ray_dir) * hit_normal;
    vec3 p1 = a1 - dot(ray_dir, a1) * ray_dir;
    a1 *= rc.radius / max(0.0001, length(p1));

    vec3 a2 = cross(hit_normal, a1);
    vec3 p2 = a2 - dot(ray_dir, a2) * ray_dir;
    a2 *= rc.radius / max(0.0001, length(p2));

    vec3 delta = hit_pos - tri_pos[0];
    vec3 e1 = tri_pos[1] - tri_pos[0];
    vec3 e2 = tri_pos[2] - tri_pos[0];
    float inv_area = 1.0f / dot(hit_normal, cross(e1, e2));

    vec3 eP = delta + a1;
    vec2 a1_bary = vec2(
        dot(hit_normal, cross(eP, e2)),
        dot(hit_normal, cross(e1, eP))
    ) * inv_area;

    puvdx = (1.0f - a1_bary.x - a1_bary.y) * tri_uv[0] + a1_bary.x * tri_uv[1] + a1_bary.y * tri_uv[2] - hit_uv;

    eP = delta + a2;
    vec2 a2_bary = vec2(
        dot(hit_normal, cross(eP, e2)),
        dot(hit_normal, cross(e1, eP))
    ) * inv_area;

    puvdy = (1.0f - a2_bary.x - a2_bary.y) * tri_uv[0] + a2_bary.x * tri_uv[1] + a2_bary.y * tri_uv[2] - hit_uv;
}

#endif
