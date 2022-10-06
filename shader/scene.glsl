#ifndef SCENE_GLSL
#define SCENE_GLSL
#include "material.glsl"
#include "color.glsl"
#include "camera.glsl"

struct vertex
{
    vec3 pos;
    vec3 normal;
    vec2 uv;
    vec4 tangent;
};

struct vertex_data
{
    vec3 pos;
#ifdef CALC_PREV_VERTEX_POS
    vec3 prev_pos;
#endif
    vec3 hard_normal;
    vec3 smooth_normal;
    vec3 mapped_normal;
    vec2 uv;
    vec3 tangent;
    vec3 bitangent;
    bool back_facing;
    int instance_id;
    int primitive_id;
};

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

struct instance
{
    uint mesh_id;
    uint pad;
    int sh_grid_index;
    float shadow_terminator_mul;
    mat4 model;
    mat4 model_normal;
    mat4 model_prev;
    material mat;
};

#ifdef SCENE_DATA_BUFFER_BINDING
layout(binding = SCENE_DATA_BUFFER_BINDING, set = 0, scalar) buffer scene_data_buffer
{
    instance o[];
} scene;
#endif

#ifdef DIRECTIONAL_LIGHT_BUFFER_BINDING
layout(binding = DIRECTIONAL_LIGHT_BUFFER_BINDING, set = 0, scalar) buffer directional_light_buffer
{
    directional_light lights[];
} directional_lights;
#endif

#ifdef POINT_LIGHT_BUFFER_BINDING
layout(binding = POINT_LIGHT_BUFFER_BINDING, set = 0, scalar) buffer point_light_buffer
{
    point_light lights[];
} point_lights;
#endif

#ifdef TEXTURE_3D_ARRAY_BINDING
layout(binding = TEXTURE_3D_ARRAY_BINDING, set = 0) uniform sampler3D textures3d[];
#endif

#ifdef SH_GRID_BUFFER_BINDING
struct sh_grid
{
    mat4 pos_from_world;
    mat4 normal_from_world;
    vec3 grid_clamp;
    float pad0;
    vec3 grid_resolution;
    float pad1;
};

layout(binding = SH_GRID_BUFFER_BINDING, set = 0, scalar) buffer sh_grid_buffer
{
    sh_grid grids[];
} sh_grids;
#endif

#ifdef VERTEX_BUFFER_BINDING
layout(binding = VERTEX_BUFFER_BINDING, set = 0, scalar) buffer vertex_buffer
{
    vertex v[];
} vertices[];
#endif

#ifdef INDEX_BUFFER_BINDING
layout(binding = INDEX_BUFFER_BINDING, set = 0) buffer index_buffer
{
    uint i[];
} indices[];
#endif

#ifdef TLAS_BINDING
layout(binding = TLAS_BINDING, set = 0) uniform accelerationStructureEXT tlas;
#endif

#ifdef ENVIRONMENT_MAP_BINDING
layout(binding = ENVIRONMENT_MAP_BINDING, set = 0) uniform sampler2D environment_map_tex;
#endif

#ifdef SCENE_METADATA_BUFFER_BINDING
layout(binding = SCENE_METADATA_BUFFER_BINDING, set = 0, scalar) uniform scene_metadata_buffer
{
    uint point_light_count;
    uint directional_light_count;
} scene_metadata;
#endif

#define POINT_LIGHT_FOR_BEGIN(world_pos) \
    for(uint item_index = 0; item_index < scene_metadata.point_light_count; ++item_index) {
#define POINT_LIGHT_FOR_END }

void random_sample_point_light(vec3 world_pos, float u, int item_count, out float selected_weight, out int selected_index)
{
    selected_index = clamp(int(u * item_count), 0, item_count-1);
    selected_weight = item_count;
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

#ifdef TEXTURE_ARRAY_BINDING
layout(binding = TEXTURE_ARRAY_BINDING, set = 0) uniform sampler2D textures[];

sampled_material sample_material(material mat, inout vertex_data v)
{
    sampled_material res;
    res.albedo = mat.albedo_factor;
    if(mat.albedo_tex_id >= 0)
    {
        vec4 tex_col = texture(textures[nonuniformEXT(mat.albedo_tex_id)], v.uv);
        tex_col.rgb = inverse_srgb_correction(tex_col.rgb);
        res.albedo *= tex_col;
    }

    vec2 mr = mat.metallic_roughness_factor.xy;
    if(mat.metallic_roughness_tex_id >= 0)
        mr *= texture(
            textures[nonuniformEXT(mat.metallic_roughness_tex_id)], v.uv
        ).bg;
    res.metallic = mr.x;
    // Squaring the roughness is just a thing artists like for some reason,
    // which is why roughness textures are authored to expect that. Which is why
    // we square the roughness here.
    res.roughness = mr.y * mr.y;

    if(mat.normal_tex_id >= 0)
    {
        mat3 tbn = mat3(
            v.tangent,
            v.bitangent,
            v.smooth_normal
        );
        vec3 ts_normal = normalize(
            texture(textures[nonuniformEXT(mat.normal_tex_id)], v.uv).xyz * 2.0f - 1.0f
        );
        v.mapped_normal = normalize(tbn * (ts_normal * vec3(mat.normal_factor, mat.normal_factor, 1.0f)));
        // Sometimes annoying stuff happens and the normal is broken. This isn't
        // usually fatal in rasterization, but is one source of NaN values in
        // path tracing.
        v.mapped_normal = any(isnan(v.mapped_normal)) ?
            v.smooth_normal : v.mapped_normal;
    }

    res.emission = mat.emission_factor_double_sided.rgb;
    if(mat.emission_tex_id >= 0)
        res.emission *=
            texture(textures[nonuniformEXT(mat.emission_tex_id)], v.uv).rgb;

    res.transmittance = mat.transmittance;
    if(v.back_facing && res.transmittance > 0.0001f)
    {
        res.ior_in = mat.ior;
        res.ior_out = 1.0f;
    }
    else
    {
        res.ior_in = 1.0f;
        res.ior_out = mat.ior;
    }

    float f0 = (res.ior_out-res.ior_in)/(res.ior_out+res.ior_in);
    f0 *= f0;
    res.f0 = f0;
    res.double_sided = mat.emission_factor_double_sided.a > 0.5f;
    res.shadow_terminator_mul = 1.0f;
    return res;
}
#endif

#endif
