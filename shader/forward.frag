#version 450
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_multiview : enable

#define SCENE_DATA_BUFFER_BINDING 0
#define TEXTURE_ARRAY_BINDING 1
#define DIRECTIONAL_LIGHT_BUFFER_BINDING 2
#define POINT_LIGHT_BUFFER_BINDING 3
#define CAMERA_DATA_BINDING 4
#define SHADOW_MAP_BUFFER_BINDING 5
#define SHADOW_MAP_CASCADE_BUFFER_BINDING 6
#define SHADOW_MAP_ATLAS_BINDING 7
#define SHADOW_MAP_ATLAS_TEST_BINDING 8
#define PCF_NOISE_VECTOR_2D_BINDING 9
#define PCF_NOISE_VECTOR_3D_BINDING 10
#define TEXTURE_3D_ARRAY_BINDING 11
#define SH_GRID_BUFFER_BINDING 12
#define BRDF_INTEGRATION_BINDING 13
#define SCENE_METADATA_BUFFER_BINDING 14
#define CALC_PREV_VERTEX_POS
#include "forward.glsl"
#include "ggx.glsl"
#include "spherical_harmonics.glsl"
#include "gbuffer.glsl"

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_prev_pos;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec2 in_uv;
layout(location = 4) in vec3 in_tangent;
layout(location = 5) in vec3 in_bitangent;

sampled_material sample_material(inout vertex_data v)
{
    material mat = scene.o[control.instance_id].mat;
    return sample_material(mat, v);
}

vertex_data get_vertex_data()
{
    vertex_data v;
    v.pos = in_pos;
    v.prev_pos = in_prev_pos;
    v.hard_normal = -normalize(cross(dFdxFine(v.pos), dFdyFine(v.pos)));
    v.smooth_normal = normalize(in_normal);
    v.mapped_normal = v.smooth_normal;
    v.uv = in_uv;
    v.tangent = in_tangent;
    v.bitangent = in_bitangent;
    v.back_facing = !gl_FrontFacing;
    if(v.back_facing)
    {
        v.smooth_normal = -v.smooth_normal;
        v.mapped_normal = -v.mapped_normal;
        v.tangent = -v.tangent;
        v.bitangent = -v.bitangent;
    }
    return v;
}

void eval_punctual_lights(
    mat3 tbn, vec3 shading_view, sampled_material mat, vertex_data v,
    out vec3 diffuse_contrib,
    out vec3 specular_contrib
){
    diffuse_contrib = vec3(0);
    specular_contrib = vec3(0);
    bool opaque = mat.transmittance < 0.0001f;
    for(uint i = 0; i < scene_metadata.directional_light_count; ++i)
    {
        vec3 d, s;
        directional_light dl = directional_lights.lights[i];
        ggx_brdf(-dl.dir * tbn, shading_view, mat, d, s);
        d *= dl.color;
        s *= dl.color;
        float shadow = 1.0f;
        if(dl.shadow_map_index >= 0 && dot(v.mapped_normal, -dl.dir) > 0)
            shadow = calc_directional_shadow(
                dl.shadow_map_index, v.pos, v.hard_normal, -dl.dir
            );
        d = dot(v.hard_normal, dl.dir) > 0 ? vec3(0) : d;
        s = dot(v.hard_normal, dl.dir) > 0 ? vec3(0) : s;
        diffuse_contrib += d * shadow;
        specular_contrib += s * shadow;
    }

    POINT_LIGHT_FOR_BEGIN(v.pos)
        vec3 d, s;
        point_light pl = point_lights.lights[item_index];
        vec3 light_dir;
        float light_dist;
        vec3 light_color;
        get_point_light_info(pl, v.pos, light_dir, light_dist, light_color);
        ggx_brdf(light_dir * tbn, shading_view, mat, d, s);
        d *= light_color;
        s *= light_color;
        float shadow = 1.0f;
        if(pl.shadow_map_index >= 0 && dot(v.mapped_normal, light_dir) > 0)
            shadow = calc_point_shadow(
                pl.shadow_map_index, v.pos, v.hard_normal, light_dir
            );
        d = dot(v.hard_normal, light_dir) < 0 ? vec3(0) : d;
        s = dot(v.hard_normal, light_dir) < 0 ? vec3(0) : s;
        diffuse_contrib += d * shadow;
        specular_contrib += s * shadow;
    POINT_LIGHT_FOR_END
}

float fresnel_schlick_attenuated(float cos_d, float f0, float roughness)
{
    return f0 + (max(1.0f - roughness, f0) - f0) * pow(1.0f - cos_d, 5.0f);
}

void brdf_indirect(
    vec3 incoming_diffuse,
    vec3 incoming_specular,
    vec3 view_dir,
    vec3 normal,
    sampled_material mat,
    out vec3 diffuse_weight,
    out vec3 specular_weight
){
    float cos_v = max(dot(normal, view_dir), 0.0f);

    // The fresnel value must be attenuated, because we are actually integrating
    // over all directions instead of just one specific direction here. This is
    // an approximated function, though.
    vec3 fresnel = mix(
        vec3(fresnel_schlick_attenuated(cos_v, mat.f0, mat.roughness)),
        mat.albedo.rgb,
        mat.metallic
    );

    vec3 kd = (vec3(1.0f) - fresnel) * (1.0f - mat.metallic) * (1.0f - mat.transmittance);
    vec3 diffuse = kd * incoming_diffuse;

    // The integration texture was generated with squared roughness, so we'll
    // have to make sure we don't square it twice!
    vec2 bi = texture(brdf_integration, vec2(cos_v, sqrt(mat.roughness))).xy;
    vec3 specular = incoming_specular * (fresnel * bi.x + bi.y);

    diffuse_weight = diffuse;
    specular_weight = specular;
}

void eval_indirect_light(
    vec3 view, sampled_material mat, vertex_data v,
    out vec3 diffuse_weight,
    out vec3 specular_weight
){
    vec3 ref_dir = reflect(view, v.mapped_normal);
    vec3 indirect_diffuse = vec3(0);
    vec3 indirect_specular = vec3(0);

    int sh_grid_index = scene.o[control.instance_id].sh_grid_index;
    if(sh_grid_index >= 0)
    {
        sh_grid sg = sh_grids.grids[sh_grid_index];
        vec3 sample_pos = (sg.pos_from_world * vec4(v.pos, 1)).xyz;
        vec3 sample_normal = normalize((sg.normal_from_world * vec4(v.smooth_normal, 0)).xyz);

        sh_probe sh = sample_sh_grid(
            textures3d[nonuniformEXT(sh_grid_index)], sg.grid_clamp,
            sample_pos, sample_normal
        );

        vec3 sh_normal = normalize((sg.normal_from_world * vec4(v.mapped_normal, 0)).xyz);
        indirect_diffuse = calc_sh_irradiance(sh, sh_normal);

        vec3 sh_ref = normalize((sg.normal_from_world * vec4(ref_dir, 0)).xyz);
        // The GGX specular function was fitted for squared roughness, so we'll
        // have to make sure we don't square it twice!
        indirect_specular = calc_sh_ggx_specular(sh, sh_ref, sqrt(mat.roughness));
    }
    else indirect_diffuse = indirect_specular = control.ambient_color;
    brdf_indirect(
        indirect_diffuse, indirect_specular, -view, v.mapped_normal, mat,
        diffuse_weight,
        specular_weight
    );
}

void main()
{
    vertex_data v = get_vertex_data();
    sampled_material mat = sample_material(v);
    vec3 view = normalize(v.pos - camera.pairs[gl_ViewIndex].current.origin.xyz);
    mat3 tbn = create_tangent_space(v.mapped_normal);
    vec3 shading_view = -view * tbn;
    vec3 direct_diffuse;
    vec3 direct_specular;
#if defined(COLOR_TARGET_LOCATION) || defined(DIRECT_TARGET_LOCATION) || defined(DIFFUSE_TARGET_LOCATION)
    eval_punctual_lights(tbn, shading_view, mat, v, direct_diffuse, direct_specular);
    direct_diffuse *= mat.albedo.rgb;
    direct_diffuse += mat.emission;
#endif
    vec3 indirect_diffuse;
    vec3 indirect_specular;
#if defined(COLOR_TARGET_LOCATION) || defined(DIFFUSE_TARGET_LOCATION)
    eval_indirect_light(view, mat, v, indirect_diffuse, indirect_specular);
    indirect_diffuse *= mat.albedo.rgb;
#endif

    write_gbuffer_color(vec4(direct_diffuse + direct_specular + indirect_diffuse + indirect_specular, mat.albedo.a));
    write_gbuffer_direct(vec4(direct_diffuse + direct_specular, mat.albedo.a));
    write_gbuffer_diffuse(vec4(direct_diffuse + indirect_diffuse, mat.albedo.a));
    write_gbuffer_albedo(vec4(mat.albedo));
    write_gbuffer_material(mat);
    write_gbuffer_normal(v.mapped_normal);
    write_gbuffer_pos(v.pos);
    write_gbuffer_screen_motion(get_camera_projection(camera.pairs[gl_ViewIndex].previous, v.prev_pos));
    write_gbuffer_instance_id(int(control.instance_id));
    write_gbuffer_linear_depth();
    write_gbuffer_flat_normal(normalize(cross(dFdy(v.pos), dFdx(v.pos))));
    write_gbuffer_emission(mat.emission);
}
