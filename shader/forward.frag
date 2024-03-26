#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_multiview : enable

#define CALC_PREV_VERTEX_POS
#include "forward.glsl"
#include "ggx.glsl"
#include "spherical_harmonics.glsl"
#include "gbuffer.glsl"
#include "shadow_mapping.glsl"

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_prev_pos;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec2 in_uv;
layout(location = 4) in vec3 in_tangent;
layout(location = 5) in vec3 in_bitangent;

sampled_material sample_material(inout vertex_data v)
{
    material mat = instances.o[control.instance_id].mat;
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
    inout vec3 diffuse,
    inout vec3 reflection
){
    bool opaque = mat.transmittance < 0.0001f;
    for(uint i = 0; i < scene_metadata.directional_light_count; ++i)
    {
        directional_light dl = directional_lights.lights[i];
        bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
        ggx_brdf(-dl.dir * tbn, shading_view, mat, lobes);
        float shadow = 1.0f;
        if(dl.shadow_map_index >= 0 && dot(v.mapped_normal, -dl.dir) > 0)
            shadow = calc_directional_shadow(
                dl.shadow_map_index, v.pos, v.hard_normal, -dl.dir
            );
        if(dot(v.hard_normal, dl.dir) > 0)
            shadow = 0.0f;

        add_demodulated_color(lobes, shadow * dl.color, diffuse, reflection);
    }

    POINT_LIGHT_FOR_BEGIN(v.pos)
        point_light pl = point_lights.lights[item_index];
        vec3 light_dir;
        float light_dist;
        vec3 light_color;
        get_point_light_info(pl, v.pos, light_dir, light_dist, light_color);
        bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
        ggx_brdf(light_dir * tbn, shading_view, mat, lobes);
        float shadow = 1.0f;
        if(pl.shadow_map_index >= 0 && dot(v.mapped_normal, light_dir) > 0)
            shadow = calc_point_shadow(
                pl.shadow_map_index, v.pos, v.hard_normal, light_dir
            );
        if(dot(v.hard_normal, light_dir) < 0)
            shadow = 0.0f;

        add_demodulated_color(lobes, shadow * light_color, diffuse, reflection);
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
    inout vec3 diffuse,
    inout vec3 reflection
){
    float cos_v = max(dot(normal, view_dir), 0.0f);

    // The fresnel value must be attenuated, because we are actually integrating
    // over all directions instead of just one specific direction here. This is
    // an approximated function, though.
    float fresnel = fresnel_schlick_attenuated(cos_v, mat.f0, mat.roughness);
    float kd = (1.0f - fresnel) * (1.0f - mat.metallic) * (1.0f - mat.transmittance);
    diffuse += kd * incoming_diffuse;

    // have to make sure we don't square it twice!
    vec2 bi = texture(brdf_integration, vec2(cos_v, sqrt(mat.roughness))).xy;
    reflection += incoming_specular * mix(fresnel * bi.x + bi.y, 1.0f, mat.metallic);
}

void eval_indirect_light(
    vec3 view,
    sampled_material mat,
    vertex_data v,
    inout vec3 diffuse,
    inout vec3 reflection
){
    vec3 ref_dir = reflect(view, v.mapped_normal);
    vec3 incoming_diffuse = vec3(0);
    vec3 incoming_reflection = vec3(0);

    int sh_grid_index = instances.o[control.instance_id].sh_grid_index;
    if(sh_grid_index >= 0)
    {
        sh_grid sg = sh_grids.grids[sh_grid_index];
        vec3 sample_pos = (sg.pos_from_world * vec4(v.pos, 1)).xyz;
        vec3 sample_normal = normalize((sg.normal_from_world * vec4(v.smooth_normal, 0)).xyz);

        sh_probe sh = sample_sh_grid(
            sh_grid_data[nonuniformEXT(sh_grid_index)], sg.grid_clamp,
            sample_pos, sample_normal
        );

        vec3 sh_normal = normalize((sg.normal_from_world * vec4(v.mapped_normal, 0)).xyz);
        incoming_diffuse = calc_sh_irradiance(sh, sh_normal);

        vec3 sh_ref = normalize((sg.normal_from_world * vec4(ref_dir, 0)).xyz);
        // The GGX specular function was fitted for squared roughness, so we'll
        // have to make sure we don't square it twice!
        incoming_reflection = calc_sh_ggx_specular(sh, sh_ref, sqrt(mat.roughness));
    }
    else incoming_diffuse = incoming_reflection = control.ambient_color;
    brdf_indirect(
        incoming_diffuse, incoming_reflection, -view, v.mapped_normal, mat,
        diffuse, reflection
    );
}

void main()
{
    vertex_data v = get_vertex_data();
    sampled_material mat = sample_material(v);
    vec3 view = normalize(v.pos - camera.pairs[control.base_camera_index + gl_ViewIndex].current.origin.xyz);
    mat3 tbn = create_tangent_space(v.mapped_normal);
    vec3 shading_view = -view * tbn;
    vec3 diffuse = vec3(0);
    vec3 reflection = vec3(0);
#if defined(COLOR_TARGET_LOCATION) || defined(REFLECTION_TARGET_LOCATION) || defined(DIFFUSE_TARGET_LOCATION)
    eval_punctual_lights(tbn, shading_view, mat, v, diffuse, reflection);
#endif
#if defined(COLOR_TARGET_LOCATION) || defined(DIFFUSE_TARGET_LOCATION)
    eval_indirect_light(view, mat, v, diffuse, reflection);
#endif

    write_gbuffer_color(vec4(modulate_color(mat, diffuse, reflection) + mat.emission, mat.albedo.a));
    write_gbuffer_diffuse(vec4(diffuse, mat.albedo.a));
    write_gbuffer_reflection(vec4(reflection, mat.albedo.a));
    write_gbuffer_albedo(vec4(mat.albedo));
    write_gbuffer_material(mat);
    write_gbuffer_normal(dot(view, v.mapped_normal) > 0 ? -v.mapped_normal : v.mapped_normal);
    write_gbuffer_pos(v.pos);
    write_gbuffer_screen_motion(get_camera_projection(camera.pairs[control.base_camera_index + gl_ViewIndex].previous, v.prev_pos));
    write_gbuffer_instance_id(int(control.instance_id));
    write_gbuffer_linear_depth();
    write_gbuffer_flat_normal(normalize(cross(dFdy(v.pos), dFdx(v.pos))));
    write_gbuffer_emission(mat.emission);
}
