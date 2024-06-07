#ifndef CAMERA_GLSL
#define CAMERA_GLSL
#include "math.glsl"

// This is so that basic raster-based things don't have to add the definition
// separately, since they can only support this camera type anyway.
#ifndef CAMERA_PROJECTION_TYPE
#define CAMERA_PROJECTION_TYPE 0
#endif

#if CAMERA_PROJECTION_TYPE == 0 || CAMERA_PROJECTION_TYPE == 1
// Matrix-based cameras go here.
struct camera_data
{
    mat4 view;
    mat4 view_inverse;
    mat4 view_proj;
    mat4 proj_inverse;
    vec4 origin;
    vec4 dof_params;
    vec4 projection_info;
    vec4 pan;
};

void get_camera_ray(
    in camera_data cam,
    vec2 pixel_coord,
    vec2 screen_size,
    vec2 dof_u,
    out vec3 origin,
    out vec3 dir
){
    vec2 uv = pixel_coord / screen_size;
    uv = uv * 2.0f - 1.0f;

#if CAMERA_PROJECTION_TYPE == 0
#ifdef USE_DEPTH_OF_FIELD
    vec2 aperture_offset = cam.dof_params.w == 0 ?
        sample_concentric_disk(dof_u) :
        sample_regular_polygon(dof_u, cam.dof_params.z, uint(cam.dof_params.w));
    vec3 view_origin = vec3(aperture_offset * cam.dof_params.y, 0);
    vec3 view_dir = (cam.proj_inverse * vec4(uv.xy, 1, 1)).xyz * cam.dof_params.x;
    view_dir = normalize(view_dir - view_origin);

    origin = (cam.view_inverse * vec4(view_origin, 1.0f)).xyz;
    dir = normalize(cam.view_inverse * vec4(view_dir.xyz, 0)).xyz;
#else
    // Pinhole camera, so origin is always the same.
    origin = cam.origin.xyz;

    // World-space direction just derived from the projection and view matrices.
    vec4 t = cam.proj_inverse * vec4(uv.xy, 1, 1);
    dir = normalize(cam.view_inverse * vec4(t.xyz, 0)).xyz;
#endif
#elif CAMERA_PROJECTION_TYPE == 1
    origin = (cam.view_inverse * cam.proj_inverse * vec4(uv.xy, 0, 1)).xyz;
    dir = normalize(cam.view_inverse * vec4(0, 0, -1, 0)).xyz;
#endif
}

vec3 get_camera_projection(in camera_data cam, vec3 world_pos)
{
    vec4 projected_pos = cam.view_proj * vec4(world_pos, 1.0f);
    vec3 uv = vec3(projected_pos.xy / projected_pos.w, projected_pos.w);
    uv.xy = uv.xy * 0.5 + 0.5;
    return uv;
}

vec3 get_camera_projected_direction(in camera_data cam, vec3 dir)
{
    vec4 projected_dir = cam.view_proj * vec4(dir, 0.0f);
    vec3 uv = vec3(projected_dir.xy / projected_dir.w, projected_dir.w);
    uv.xy = uv.xy * 0.5 + 0.5;
    return uv;
}

// Used for edge detection algorithms. Inverse size of frustum at the depth of
// a given point.
float get_frustum_size(camera_data cam, vec3 pos)
{
    float frustum_size = min(abs(cam.projection_info.z), abs(cam.projection_info.w));
    if(cam.projection_info.x < 0) // Perspective
        frustum_size *= abs(dot(cam.view_inverse[2].xyz, cam.view_inverse[3].xyz - pos));

    return frustum_size;
}

float get_frustum_size(camera_data cam, float view_z)
{
    float frustum_size = min(abs(cam.projection_info.z), abs(cam.projection_info.w));
    if(cam.projection_info.x < 0) // Perspective
        frustum_size *= abs(view_z);

    return frustum_size;
}

#elif CAMERA_PROJECTION_TYPE == 2
// Equirectangular cameras go here.
struct camera_data
{
    mat4 view;
    mat4 view_inverse;
    vec4 origin;
    vec2 fov;
};

void get_camera_ray(
    in camera_data cam,
    vec2 pixel_coord,
    vec2 screen_size,
    vec2 dof_u,
    out vec3 origin,
    out vec3 dir
){
    vec2 uv = pixel_coord / screen_size;
    uv = (uv * 2.0f - 1.0f) * cam.fov;

    vec2 c = cos(uv);
    vec2 s = sin(uv);
    vec3 t = vec3(s.x*c.y, s.y, -c.x*c.y);
    dir = normalize(cam.view_inverse * vec4(t, 0)).xyz;

    origin = cam.origin.xyz;
}

vec3 get_camera_projection(in camera_data cam, vec3 world_pos)
{
    vec3 t = (cam.view * vec4(world_pos, 1.0f)).xyz;
    float t_len = length(t);
    t /= t_len
    vec3 uv = vec2(atan(t.x, -t.z), asin(t.y), t_len);
    uv.xy = (uv.xy / cam.fov) * 0.5 + 0.5;
    return uv;
}

#endif

#endif

