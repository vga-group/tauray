#ifndef SCENE_GLSL
#define SCENE_GLSL
#include "material.glsl"
#include "color.glsl"
#include "camera.glsl"
#include "light.glsl"

#ifndef SCENE_SET
#define SCENE_SET 1
#endif

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

struct instance
{
    int light_base_id;
    int sh_grid_index;
    uint pad;
    float shadow_terminator_mul;
    mat4 model;
    mat4 model_normal;
    mat4 model_prev;
    material mat;
};

layout(binding = 0, set = SCENE_SET, scalar) buffer instance_data_buffer
{
    instance o[];
} instances;

layout(binding = 1, set = SCENE_SET, scalar) buffer directional_light_buffer
{
    directional_light lights[];
} directional_lights;

layout(binding = 2, set = SCENE_SET, scalar) buffer point_light_buffer
{
    point_light lights[];
} point_lights;

layout(binding = 3, set = SCENE_SET, scalar) buffer tri_light_buffer
{
    tri_light lights[];
} tri_lights;

layout(binding = 4, set = SCENE_SET, scalar) buffer vertex_buffer
{
    vertex v[];
} vertices[];

layout(binding = 5, set = SCENE_SET) buffer index_buffer
{
    uint i[];
} indices[];

layout(binding = 6, set = SCENE_SET) uniform sampler2D textures[];

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

layout(binding = 7, set = SCENE_SET) uniform sampler2D environment_map_tex;

#include "alias_table.glsl"
layout(binding = 8, set = SCENE_SET) readonly buffer environment_map_alias_table_buffer
{
    alias_table_entry entries[];
} environment_map_alias_table;

layout(binding = 9, set = SCENE_SET, scalar) uniform scene_metadata_buffer
{
    uint point_light_count;
    uint directional_light_count;
    uint tri_light_count;
    int environment_proj;
    vec4 environment_factor;
} scene_metadata;

#define POINT_LIGHT_FOR_BEGIN(world_pos) \
    for(uint item_index = 0; item_index < scene_metadata.point_light_count; ++item_index) {
#define POINT_LIGHT_FOR_END }

struct camera_pair
{
    camera_data current;
    camera_data previous;
};

layout(binding = 10, set = SCENE_SET) readonly buffer camera_data_buffer
{
    camera_pair pairs[];
} camera;

#ifdef RAY_TRACING
layout(binding = 11, set = SCENE_SET) uniform accelerationStructureEXT tlas;
#endif

#endif
