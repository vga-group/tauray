#ifndef RT_GLSL
#define RT_GLSL
#define RAY_TRACING
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

#ifdef PRE_TRANSFORMED_VERTICES
#define TRANSFORM_MAT(name, val) val
#else
#define TRANSFORM_MAT(name, val) (name * val)
#endif

// I would prefer this to be INF, but it's hard to write in glsl.
#define RAY_MAX_DIST float(1e39)

#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
vertex_data get_interpolated_vertex(vec3 view, vec2 barycentrics, int instance_id, int primitive_id, vec3 pos, out float pdf)
#else
vertex_data get_interpolated_vertex(vec3 view, vec2 barycentrics, int instance_id, int primitive_id)
#endif
{
    instance o = instances.o[instance_id];
    ivec3 i = ivec3(
        indices[nonuniformEXT(instance_id)].i[3*primitive_id+0],
        indices[nonuniformEXT(instance_id)].i[3*primitive_id+1],
        indices[nonuniformEXT(instance_id)].i[3*primitive_id+2]
    );
    vertex v0 = vertices[nonuniformEXT(instance_id)].v[i.x];
    vertex v1 = vertices[nonuniformEXT(instance_id)].v[i.y];
    vertex v2 = vertices[nonuniformEXT(instance_id)].v[i.z];

    vec3 b = vec3(1.0f - barycentrics.x - barycentrics.y, barycentrics);

    vec4 avg_tangent = v0.tangent * b.x + v1.tangent * b.y + v2.tangent * b.z;

    vertex_data interp;

    vec4 model_pos = vec4(v0.pos * b.x + v1.pos * b.y + v2.pos * b.z, 1);
    interp.pos = TRANSFORM_MAT(o.model, model_pos).xyz;

#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
    if(o.light_base_id >= 0)
    {
        pdf = sample_triangle_light_pdf(
            interp.pos - pos,
            TRANSFORM_MAT(o.model, vec4(v0.pos, 1)).xyz - pos,
            TRANSFORM_MAT(o.model, vec4(v1.pos, 1)).xyz - pos,
            TRANSFORM_MAT(o.model, vec4(v2.pos, 1)).xyz - pos
        );
    }
    else pdf = 0.0f;
#endif

#ifdef CALC_PREV_VERTEX_POS
#ifdef PRE_TRANSFORMED_VERTICES
    interp.prev_pos = (o.model_prev * inverse(o.model) * model_pos).xyz;
#else
    interp.prev_pos = (o.model_prev * model_pos).xyz;
#endif
#endif
    interp.smooth_normal = normalize(TRANSFORM_MAT(
        mat3(o.model_normal),
        (v0.normal * b.x + v1.normal * b.y + v2.normal * b.z)
    ));
    interp.tangent = normalize(TRANSFORM_MAT(mat3(o.model_normal), avg_tangent.xyz));
    interp.bitangent = normalize(cross(interp.smooth_normal, interp.tangent) * avg_tangent.w);
    interp.uv = v0.uv * b.x + v1.uv * b.y + v2.uv * b.z;

    // Flip normal if back-facing triangle
    interp.hard_normal = normalize(TRANSFORM_MAT(mat3(o.model_normal), cross(v1.pos-v0.pos, v2.pos-v0.pos)));
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

void get_interpolated_vertex_light(vec3 view, vec2 barycentrics, int instance_id, int primitive_id, out vec2 uv)
{
    instance o = instances.o[instance_id];
    ivec3 i = ivec3(
        indices[nonuniformEXT(instance_id)].i[3*primitive_id+0],
        indices[nonuniformEXT(instance_id)].i[3*primitive_id+1],
        indices[nonuniformEXT(instance_id)].i[3*primitive_id+2]
    );
    vertex v0 = vertices[nonuniformEXT(instance_id)].v[i.x];
    vertex v1 = vertices[nonuniformEXT(instance_id)].v[i.y];
    vertex v2 = vertices[nonuniformEXT(instance_id)].v[i.z];

    vec3 b = vec3(1.0f - barycentrics.x - barycentrics.y, barycentrics);
    uv = v0.uv * b.x + v1.uv * b.y + v2.uv * b.z;
}

sampled_material sample_material(int instance_id, inout vertex_data v)
{
    instance o = instances.o[instance_id];
    material mat = o.mat;
    sampled_material res = sample_material(mat, v);
    res.shadow_terminator_mul = o.shadow_terminator_mul;
    return res;
}

bool is_material_skippable(int instance_id, vec2 uv, float alpha_cutoff)
{
    material mat = instances.o[instance_id].mat;
    vec4 albedo = mat.albedo_factor;
    if(mat.albedo_tex_id >= 0)
        albedo *= texture(textures[nonuniformEXT(mat.albedo_tex_id)], uv);

    return (albedo.a <= alpha_cutoff);
}

// A workaround for the shadow terminator problem, the idea is from
// the Appleseed renderer. This is not physically based, so it's easy to
// disable. It shouldn't do anything if mul == 1, which should be the default
// for all objects.
void shadow_terminator_fix(inout vec3 contrib, float cos_l, in sampled_material mat)
{
#ifdef USE_SHADOW_TERMINATOR_FIX
    float s = (cos_l <= 0 || mat.shadow_terminator_mul == 1) ?
        1.0f : max(cos(acos(cos_l) * mat.shadow_terminator_mul)/cos_l, 0.0f);
    contrib *= s;
#endif
}

camera_data get_camera()
{
    return camera.pairs[gl_LaunchIDEXT.z].current;
}

camera_data get_prev_camera()
{
    return camera.pairs[gl_LaunchIDEXT.z].previous;
}

#ifdef DISTRIBUTION_DATA_BINDING
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
#if !defined(DISTRIBUTION_STRATEGY) || DISTRIBUTION_STRATEGY == 0
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
#if !defined(DISTRIBUTION_STRATEGY) || DISTRIBUTION_STRATEGY == 0
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
#if !defined(DISTRIBUTION_STRATEGY) || DISTRIBUTION_STRATEGY == 0
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

// Based on CC0 code from https://gist.github.com/juliusikkala/6c8c186f0150fe877a55cee4d266b1b0
vec3 sample_environment_map(
    uvec3 rand,
    out vec3 shadow_ray_direction,
    out float shadow_ray_length,
    out float pdf
){
    vec3 color = scene_metadata.environment_factor.rgb;
    if(scene_metadata.environment_proj >= 0)
    {
        uvec2 size = textureSize(environment_map_tex, 0).xy;
        const uint pixel_count = size.x * size.y;
        uvec2 ip = clamp(rand.xy / (0xFFFFFFFFu / size), uvec2(0), size-1u);
        int i = int(ip.x + ip.y * size.x);
        alias_table_entry at = environment_map_alias_table.entries[i];
        pdf = at.pdf;
        if(rand.z > at.probability)
        {
            i = int(at.alias_id);
            pdf = at.alias_pdf;
        }

        ivec2 p = ivec2(i % size.x, i / size.x);
        vec2 off = ldexp(vec2(uvec2(rand.xy*pixel_count)), ivec2(-32));
        vec2 uv = (vec2(p) + off)/vec2(size);

        shadow_ray_direction = uv_to_latlong_direction(uv);

        color *= texture(environment_map_tex, vec2(uv.x, uv.y)).rgb;
    }
    else
    {
        pdf = 1.0f / (4.0f * M_PI);
        shadow_ray_direction = sample_sphere(ldexp(vec2(rand.xy), ivec2(-32)));
    }
    shadow_ray_length = RAY_MAX_DIST;
    return color;
}

float sample_environment_map_pdf(vec3 dir)
{
    if(scene_metadata.environment_proj >= 0)
    {
        uvec2 size = textureSize(environment_map_tex, 0).xy;
        const uint pixel_count = size.x * size.y;
        uint i = latlong_direction_to_pixel_id(dir, ivec2(size));
        alias_table_entry at = environment_map_alias_table.entries[i];
        return at.pdf;
    }
    else return 1.0f / (4.0f * M_PI);
}

void get_nee_sampling_probabilities(out float point, out float triangle, out float directional, out float envmap)
{
#ifdef NEE_SAMPLE_POINT_LIGHTS
    if(scene_metadata.point_light_count > 0) point = NEE_SAMPLE_POINT_LIGHTS;
    else
#endif
    point = 0.0f;

#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
    if(scene_metadata.tri_light_count > 0) triangle = NEE_SAMPLE_EMISSIVE_TRIANGLES;
    else
#endif
    triangle = 0.0f;

#ifdef NEE_SAMPLE_DIRECTIONAL_LIGHTS
    if(scene_metadata.directional_light_count > 0) directional = NEE_SAMPLE_DIRECTIONAL_LIGHTS;
    else
#endif
    directional = 0.0f;

#ifdef NEE_SAMPLE_ENVMAP
    if(scene_metadata.environment_proj >= 0) envmap = NEE_SAMPLE_ENVMAP;
    else
#endif
    envmap = 0.0f;

    float sum = point + triangle + directional + envmap;
    float inv_sum = sum <= 0.0f ? 0.0f : (1.0f/sum + 1e-5f);

    point *= inv_sum;
    triangle *= inv_sum;
    directional *= inv_sum;
    envmap *= inv_sum;
}

#endif

