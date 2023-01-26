#ifndef RT_GLSL
#define RT_GLSL
#include "scene.glsl"
#include "gbuffer.glsl"

#ifdef DISTRIBUTION_DATA_BINDING
layout(binding = DISTRIBUTION_DATA_BINDING, set = 0) uniform distribution_data_buffer
{
    uvec2 size;
    uint index;
    uint count;
    uint primary;
    uint samples_accumulated;
} distribution;
#endif

// I would prefer this to be INF, but it's hard to write in glsl.
#define RAY_MAX_DIST float(1e39)

#if defined(SCENE_DATA_BUFFER_BINDING) && defined(VERTEX_BUFFER_BINDING) && defined(INDEX_BUFFER_BINDING)
#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
vertex_data get_interpolated_vertex(vec3 view, vec2 barycentrics, int instance_id, int primitive_id, vec3 pos, out float solid_angle)
#else
vertex_data get_interpolated_vertex(vec3 view, vec2 barycentrics, int instance_id, int primitive_id)
#endif
{
    instance o = scene.o[instance_id];
    ivec3 i = ivec3(
        indices[nonuniformEXT(o.mesh_id)].i[3*primitive_id+0],
        indices[nonuniformEXT(o.mesh_id)].i[3*primitive_id+1],
        indices[nonuniformEXT(o.mesh_id)].i[3*primitive_id+2]
    );
    vertex v0 = vertices[nonuniformEXT(o.mesh_id)].v[i.x];
    vertex v1 = vertices[nonuniformEXT(o.mesh_id)].v[i.y];
    vertex v2 = vertices[nonuniformEXT(o.mesh_id)].v[i.z];

    vec3 b = vec3(1.0f - barycentrics.x - barycentrics.y, barycentrics);

    vec4 avg_tangent = v0.tangent * b.x + v1.tangent * b.y + v2.tangent * b.z;

    vertex_data interp;
#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
    if(o.light_base_id >= 0)
    {
        solid_angle = spherical_triangle_solid_angle(
            (o.model * vec4(v0.pos, 1)).xyz - pos,
            (o.model * vec4(v1.pos, 1)).xyz - pos,
            (o.model * vec4(v2.pos, 1)).xyz - pos
        );
    }
    else solid_angle = 0.0f;
#endif

    vec4 model_pos = vec4(v0.pos * b.x + v1.pos * b.y + v2.pos * b.z, 1);
    interp.pos = (o.model * model_pos).xyz;
#ifdef CALC_PREV_VERTEX_POS
    interp.prev_pos = (o.model_prev * model_pos).xyz;
#endif
    interp.smooth_normal = normalize(mat3(o.model_normal) *
        (v0.normal * b.x + v1.normal * b.y + v2.normal * b.z));
    interp.tangent = normalize(mat3(o.model_normal) * avg_tangent.xyz);
    interp.bitangent = normalize(cross(interp.smooth_normal, interp.tangent) * avg_tangent.w);
    interp.uv = v0.uv * b.x + v1.uv * b.y + v2.uv * b.z;

    // Flip normal if back-facing triangle
    interp.hard_normal = normalize(mat3(o.model_normal) * cross(v1.pos-v0.pos, v2.pos-v0.pos));
    interp.back_facing = dot(interp.hard_normal, view) > 0;
    if(interp.back_facing)
    {
        interp.smooth_normal = -interp.smooth_normal;
        interp.hard_normal = -interp.hard_normal;
    }
    interp.mapped_normal = interp.smooth_normal;
    interp.instance_id = instance_id;
    interp.primitive_id = primitive_id;

    return interp;
}

void get_interpolated_vertex_light(vec3 view, vec2 barycentrics, int instance_id, int primitive_id, out vec2 uv, out bool back_facing)
{
    instance o = scene.o[instance_id];
    ivec3 i = ivec3(
        indices[nonuniformEXT(o.mesh_id)].i[3*primitive_id+0],
        indices[nonuniformEXT(o.mesh_id)].i[3*primitive_id+1],
        indices[nonuniformEXT(o.mesh_id)].i[3*primitive_id+2]
    );
    vertex v0 = vertices[nonuniformEXT(o.mesh_id)].v[i.x];
    vertex v1 = vertices[nonuniformEXT(o.mesh_id)].v[i.y];
    vertex v2 = vertices[nonuniformEXT(o.mesh_id)].v[i.z];

    vec3 b = vec3(1.0f - barycentrics.x - barycentrics.y, barycentrics);
    uv = v0.uv * b.x + v1.uv * b.y + v2.uv * b.z;

    vec3 hard_normal = mat3(o.model_normal) * normalize(cross(v1.pos-v0.pos, v2.pos-v0.pos));
    back_facing = dot(hard_normal, view) > 0;
}

sampled_material sample_material(int instance_id, inout vertex_data v)
{
    instance o = scene.o[instance_id];
    material mat = o.mat;
    sampled_material res = sample_material(mat, v);
    res.shadow_terminator_mul = o.shadow_terminator_mul;
    return res;
}

bool is_material_skippable(int instance_id, vec2 uv, bool back_facing, float alpha_cutoff)
{
    material mat = scene.o[instance_id].mat;
    vec4 albedo = mat.albedo_factor;
    if(mat.albedo_tex_id >= 0)
        albedo *= texture(textures[nonuniformEXT(mat.albedo_tex_id)], uv);

    bool double_sided = mat.emission_factor_double_sided.a > 0.5f;
    return (albedo.a <= alpha_cutoff) || (back_facing && !double_sided);
}

#endif

// A workaround for the shadow terminator problem, the idea is from
// the Appleseed renderer. This is not physically based, so it's easy to
// disable. It shouldn't do anything if mul == 1, which should be the default
// for all objects.
void shadow_terminator_fix(inout vec3 diffuse, inout vec3 specular, float cos_l, in sampled_material mat)
{
#ifdef USE_SHADOW_TERMINATOR_FIX
    float s = (cos_l <= 0 || mat.shadow_terminator_mul == 1) ?
        1.0f : max(cos(acos(cos_l) * mat.shadow_terminator_mul)/cos_l, 0.0f);
    diffuse *= s;
    specular *= s;
#endif
}

#if defined(CAMERA_DATA_BINDING) && defined(DISTRIBUTION_DATA_BINDING)
camera_data get_camera()
{
    return camera.pairs[gl_LaunchIDEXT.z].current;
}

camera_data get_prev_camera()
{
    return camera.pairs[gl_LaunchIDEXT.z].previous;
}

#if DISTRIBUTION_STRATEGY == 2
//Permute region for the pixel i
uint permute_region_id(uint i)
{
    uint region_size = ((distribution.size.x * distribution.size.y) + (1<<distribution.count) - 1) >> distribution.count;
    uint region_id = i / region_size; //Get the id of the region
    uint k = bitfieldReverse(region_id) >> (32 - distribution.count);
    return k * region_size + i % region_size;
}
#endif

ivec2 get_pixel_pos()
{
#if DISTRIBUTION_STRATEGY == 0
    return ivec2(gl_LaunchIDEXT.xy);
#elif DISTRIBUTION_STRATEGY == 1
    return ivec2(
        gl_LaunchIDEXT.x,
        gl_LaunchIDEXT.y * distribution.count + distribution.index
    );
#elif DISTRIBUTION_STRATEGY == 2
    uint j = permute_region_id(distribution.index + gl_LaunchIDEXT.x);

    if(j < distribution.size.x * distribution.size.y)
        return ivec2(j % distribution.size.x, j / distribution.size.x);
#endif
}

ivec3 get_write_pixel_pos(in camera_data cam)
{
#if DISTRIBUTION_STRATEGY == 0
    return ivec3(gl_LaunchIDEXT.xyz);
#elif DISTRIBUTION_STRATEGY == 1
    uvec3 write_pos = gl_LaunchIDEXT.xyz;
    if(distribution.primary == 1)
        write_pos.y = write_pos.y * distribution.count + distribution.index;
    return ivec3(write_pos);
#elif DISTRIBUTION_STRATEGY == 2
    uvec3 write_pos = uvec3(
        gl_LaunchIDEXT.x%distribution.size.x,
        gl_LaunchIDEXT.x/distribution.size.x,
        gl_LaunchIDEXT.z
    );

    uint j = permute_region_id(distribution.index + gl_LaunchIDEXT.x);

    if(distribution.primary == 1)
        write_pos = uvec3(j % distribution.size.x, j / distribution.size.x, gl_LaunchIDEXT.z);

    if(j < distribution.size.x * distribution.size.y)
        return ivec3(write_pos);
#endif
}

uvec2 get_screen_size()
{
#if DISTRIBUTION_STRATEGY == 0
    return uvec2(gl_LaunchSizeEXT.xy);
#else
    return distribution.size;
#endif
}

// DoF version
void get_screen_camera_ray(in camera_data cam, vec2 pixel_offset, vec2 dof_u, out vec3 origin, out vec3 dir)
{
    vec2 p = vec2(get_pixel_pos()) + (pixel_offset*0.5f+0.5f);
    // Flip into OpenGL coordinates.
    uvec2 size = get_screen_size();
    p.y = size.y - p.y;
    get_camera_ray(cam, p, vec2(size), dof_u, origin, dir);
}

// Pinhole version
void get_screen_camera_ray(in camera_data cam, vec2 pixel_offset, out vec3 origin, out vec3 dir)
{
    get_screen_camera_ray(cam, pixel_offset, vec2(0.5), origin, dir);
}
#endif

#endif

