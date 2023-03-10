#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

hitAttributeEXT vec2 attribs;

#define TLAS_BINDING 0
#define SCENE_DATA_BUFFER_BINDING 1
#define VERTEX_BUFFER_BINDING 2
#define INDEX_BUFFER_BINDING 3
#define TEXTURE_ARRAY_BINDING 4
#define DIRECTIONAL_LIGHT_BUFFER_BINDING 5
#define POINT_LIGHT_BUFFER_BINDING 6
#define SCENE_METADATA_BINDING 11
#define USE_PUSH_CONSTANTS
#include "whitted.glsl"

layout(location = 0) rayPayloadInEXT hit_payload payload;
layout(location = 1) rayPayloadEXT vec3 shadow_transmittance;

#include "ggx.glsl"

vec3 shadow_ray(vec3 pos, float min_dist, vec3 dir, float max_dist)
{
    shadow_transmittance = vec3(1.0f);
    traceRayEXT(
        tlas,
        gl_RayFlagsNoneEXT,
        0xFF,
        1,
        0,
        1,
        pos,
        min_dist,
        dir,
        max_dist,
        1
    );

    return shadow_transmittance;
}

vec4 secondary_ray(vec3 pos, float min_dist, vec3 dir, float max_dist)
{
    payload.depth++;
    vec4 refl_color = vec4(0,0,0,1);
    if(payload.depth < control.max_depth)
    {
        payload.self_instance_id = gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT;
        payload.self_primitive_id = gl_PrimitiveID;
        traceRayEXT(
            tlas,
            gl_RayFlagsNoneEXT,
            0xFF,
            0,
            0,
            0,
            pos,
            min_dist,
            dir,
            max_dist,
            0
        );
        refl_color = payload.color;
    }
    payload.depth--;

    return refl_color;
}

void main()
{
    vec3 view = gl_WorldRayDirectionEXT;
    int instance_id = gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT;
    vertex_data v = get_interpolated_vertex(view, attribs, instance_id, gl_PrimitiveID);

    sampled_material mat = sample_material(instance_id, v);

    mat3 tbn = create_tangent_space(v.mapped_normal);
    vec3 shading_view = -view * tbn;

    vec4 color = vec4(mat.emission, mat.albedo.a);

    for(uint i = 0; i < control.directional_light_count; ++i)
    {
        directional_light dl = directional_lights.lights[i];
        vec3 d, s;
        ggx_brdf(-dl.dir * tbn, shading_view, mat, d, s);
        d *= mat.albedo.rgb;
        vec3 c = (d + s) * dl.color;
        // This essentially increases shadow precision (but also further
        // enforces the shadow terminator issue :/) It helps to avoid some odd
        // light leaks in the terrain test scene.
        c = dot(v.hard_normal, dl.dir) > 0 ? vec3(0) : c;
        if(any(greaterThan(c, vec3(0.0001f))))
            color.rgb += c * shadow_ray(v.pos, control.min_ray_dist, -dl.dir, RAY_MAX_DIST);
    }

    for(uint i = 0; i < control.point_light_count; ++i)
    {
        point_light pl = point_lights.lights[i];
        vec3 light_dir;
        float light_dist;
        vec3 light_color;
        get_point_light_info(pl, v.pos, light_dir, light_dist, light_color);
        vec3 d, s;
        ggx_brdf(light_dir * tbn, shading_view, mat, d, s);
        d *= mat.albedo.rgb;
        vec3 c = (d + s) * light_color;
        c = dot(v.hard_normal, light_dir) > 0 ? c : vec3(0);
        if(any(greaterThan(c, vec3(0.0001f))))
            color.rgb += c * shadow_ray(v.pos, control.min_ray_dist, light_dir, light_dist);
    }

    // Reflection
    if(mat.roughness <= 0.5)
    {
        vec3 refl_dir = reflect(view, v.mapped_normal);
        vec4 refl_color = secondary_ray(
            v.pos + v.smooth_normal * 0.001f, control.min_ray_dist, refl_dir, RAY_MAX_DIST
        );
        vec3 d, s;
        sharp_brdf(refl_dir * tbn, shading_view, mat, d, s);
        d *= mat.albedo.rgb;
        color.rgb += (d + s) * refl_color.rgb;
    }

    // Refraction
    if(mat.transmittance > 0.0001f)
    {
        vec3 refr_dir = refract(view, v.mapped_normal, mat.ior_in/mat.ior_out);
        if(refr_dir != vec3(0))
        {
            refr_dir = normalize(refr_dir);
            vec4 refr_color = secondary_ray(v.pos-refr_dir*0.001f, 0.0f, refr_dir, RAY_MAX_DIST);
            vec3 d, s;
            sharp_btdf(refr_dir * tbn, shading_view, mat, d, s);
            d *= mat.albedo.rgb;
            color.rgb += (d + s) * refr_color.rgb;
        }
    }

    // Ambient light
    color.rgb += control.ambient.rgb * mat.albedo.rgb;

    // Alpha
    if(mat.albedo.a < 0.999f)
    {
        vec4 behind = secondary_ray(v.pos, control.min_ray_dist, view, RAY_MAX_DIST);
        color.rgb = mix(behind.rgb, color.rgb, mat.albedo.a);
    }

    // Toybox:
    // gl_WorldRayOriginEXT
    // gl_WorldRayDirectionEXT
    // gl_HitTEXT
    //
    // Add fog
    //color = mix(color, vec4(0.3f, 0.5f, 1.0f, 1.0f), 1.0f-exp(-gl_HitTEXT * 0.0005f));
    payload.color = color;
}
