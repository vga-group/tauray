#ifndef PATH_TRACER_GLSL
#define PATH_TRACER_GLSL

#define USE_RAY_QUERIES
#extension GL_EXT_ray_flags_primitive_culling : enable

#ifdef USE_SCREEN_MOTION_TARGET
#define CALC_PREV_VERTEX_POS
#endif

#if defined(DIFFUSE_TARGET_BINDING) && defined(REFLECTION_TARGET_BINDING) && defined(COLOR_TARGET_BINDING)
#error "Can only output color OR demodulated diffuse & reflection, not all at the same time."
#endif

#if defined(DIFFUSE_TARGET_BINDING) || defined(REFLECTION_TARGET_BINDING)
#define DEMODULATED_OUTPUT
#endif

#include "rt.glsl"
#include "rt_common.glsl"
#include "sampling.glsl"
#include "radiance_cascades.glsl"
#ifdef LIGHT_TREE_SET
#include "light_tree.glsl"
#endif

struct pt_vertex_data
{
    vec3 pos;
#ifdef CALC_PREV_VERTEX_POS
    vec3 prev_pos;
#endif
    vec3 hard_normal;
    vec3 smooth_normal;
    vec3 mapped_normal;
    int instance_id;
};

struct intersection_pdf
{
    float point_light_pdf;
    float directional_light_pdf;
    float tri_light_pdf;
    float envmap_pdf;
#ifdef LIGHT_TREE_SET
    uint instance_id;
    uint primitive_id;
#endif
};

#include "ggx.glsl"

float shadow_ray(vec3 pos, float min_dist, vec3 dir, float max_dist)
{
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq,
        tlas,
        gl_RayFlagsOpaqueEXT|gl_RayFlagsSkipAABBEXT|gl_RayFlagsTerminateOnFirstHitEXT,
        //0x02^0xFF, // Exclude lights from shadow rays
        0xFF,
        pos,
        min_dist,
        dir,
        max_dist
    );

    return trace_ray_query_visibility(rq);
}

float bsdf_mis_pdf(
    intersection_pdf nee_pdf,
    float bsdf_pdf,
    vec3 pos,
    vec3 normal
){
    if(bsdf_pdf == 0.0f) return 1.0f;

#ifdef LIGHT_TREE_SET
    float avg_nee_pdf =
        calculate_light_pdf(
            nee_pdf.instance_id,
            nee_pdf.primitive_id,
            nee_pdf.directional_light_pdf + nee_pdf.point_light_pdf + nee_pdf.tri_light_pdf,
            nee_pdf.envmap_pdf,
            pos,
            normal,
            1.0f
        );
#else
    float point_prob, triangle_prob, dir_prob, envmap_prob;
    get_nee_sampling_probabilities(point_prob, triangle_prob, dir_prob, envmap_prob);

    float avg_nee_pdf =
        nee_pdf.directional_light_pdf * dir_prob / max(scene_metadata.directional_light_count, 1) +
        nee_pdf.tri_light_pdf * triangle_prob / max(scene_metadata.tri_light_count, 1) +
        nee_pdf.envmap_pdf * envmap_prob +
        nee_pdf.point_light_pdf * point_prob / max(scene_metadata.point_light_count, 1);
#endif

#ifdef MIS_POWER_HEURISTIC
    return (avg_nee_pdf * avg_nee_pdf + bsdf_pdf * bsdf_pdf) / bsdf_pdf;
#elif defined(MIS_BALANCE_HEURISTIC)
    return avg_nee_pdf + bsdf_pdf;
#else
    return avg_nee_pdf > 0 ? 1.0f / 0.0f : bsdf_pdf;
#endif
}

float nee_mis_pdf(float nee_pdf, float bsdf_pdf)
{
    if(nee_pdf <= 0.0f) return -nee_pdf;

#ifdef MIS_POWER_HEURISTIC
    return (nee_pdf * nee_pdf + bsdf_pdf * bsdf_pdf) / nee_pdf;
#elif defined(MIS_BALANCE_HEURISTIC)
    return nee_pdf + bsdf_pdf;
#else
    return nee_pdf;
#endif
}

bool get_intersection_info(
    hit_info payload,
    vec3 origin,
    vec3 view,
    bool include_directional_lights,
    out pt_vertex_data v,
    out intersection_pdf nee_pdf,
    out sampled_material mat,
    out vec3 light
){
    nee_pdf.point_light_pdf = 0;
    nee_pdf.directional_light_pdf = 0;
    nee_pdf.tri_light_pdf = 0;
    nee_pdf.envmap_pdf = 0;
#ifdef LIGHT_TREE_SET
    nee_pdf.instance_id = NULL_INSTANCE_ID;
    nee_pdf.primitive_id = 0;
#endif
    mat.metallic = 1;
    mat.albedo = vec4(0);

    if(payload.instance_id >= 0)
    {
        float pdf = 0.0f;
        vertex_data vd = get_interpolated_vertex(
            view, payload.barycentrics,
            payload.instance_id,
            payload.primitive_id
#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
            , origin, pdf
#endif
        );
        mat = sample_material(payload.instance_id, vd);
        mat.albedo.a = 1.0; // Alpha blending was handled by the any-hit shader!
#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
        nee_pdf.tri_light_pdf = pdf == 0.0f ? 0.0f : pdf;
        light = mat.emission;
        mat.emission = vec3(0);
#else
        light = vec3(0);
#endif

#ifdef LIGHT_TREE_SET
        int light_base_id = instances.o[payload.instance_id].light_base_id;
        if (light_base_id >= 0)
        {
            nee_pdf.instance_id = payload.instance_id;
            nee_pdf.primitive_id = payload.primitive_id;
        }
#endif

        v.pos = vd.pos;
#ifdef CALC_PREV_VERTEX_POS
        v.prev_pos = vd.prev_pos;
#endif
        v.hard_normal = vd.hard_normal;
        v.smooth_normal = vd.smooth_normal;
        v.mapped_normal = vd.mapped_normal;
        v.instance_id = vd.instance_id;
        return true;
    }
    else if(payload.primitive_id >= 0)
    {
        point_light pl = point_lights.lights[payload.primitive_id];
        vec3 color = get_spotlight_intensity(pl, view) * pl.color / (pl.radius * pl.radius * M_PI);
#ifdef NEE_SAMPLE_POINT_LIGHTS
        mat.emission = vec3(0);
        light = color;
        nee_pdf.point_light_pdf = sample_point_light_pdf(pl, origin);
#else
        light = vec3(0);
        mat.emission = color;
#endif

#ifdef LIGHT_TREE_SET
        nee_pdf.instance_id = POINT_LIGHT_INSTANCE_ID;
        nee_pdf.primitive_id = payload.primitive_id;
#endif

        v.pos = origin + payload.barycentrics.x * view;
        #ifdef CALC_PREV_VERTEX_POS
        v.prev_pos = v.pos; // TODO?
        #endif
        v.mapped_normal = normalize(v.pos - pl.pos);
        v.instance_id = -1;
        mat.albedo = vec4(0,0,0,1);
        return false;
    }
    else
    {
        vec4 color = scene_metadata.environment_factor;
        if(scene_metadata.environment_proj >= 0)
        {
            vec2 uv = vec2(0);
            uv.y = asin(-view.y)/M_PI+0.5f;
            uv.x = atan(view.z, view.x)/(2*M_PI)+0.5f;
            color.rgb *= texture(environment_map_tex, uv).rgb;
        }

#ifdef LIGHT_TREE_SET
        nee_pdf.instance_id = ENVMAP_INSTANCE_ID;
        nee_pdf.primitive_id = packSnorm2x16(octahedral_pack(view));
#endif

        mat.emission = vec3(0);
        light = vec3(0);
        for(uint i = 0; include_directional_lights && i < scene_metadata.directional_light_count; ++i)
        {
            directional_light dl = directional_lights.lights[i];
            if(dl.dir_cutoff >= 1.0f)
                continue;
            float visible = step(dl.dir_cutoff, dot(view, -dl.dir));
            vec3 color = visible * dl.color / (2.0f * M_PI * (1.0f - dl.dir_cutoff));
#ifdef NEE_SAMPLE_DIRECTIONAL_LIGHTS
            light += color;
            nee_pdf.directional_light_pdf += visible * sample_directional_light_pdf(dl);
#else
            mat.emission += color;
#endif
        }
        v.instance_id = -1;
        v.pos = origin;
        #ifdef CALC_PREV_VERTEX_POS
        v.prev_pos = v.pos;
        #endif
        v.mapped_normal = -view;
        mat.albedo = vec4(0);

#ifdef NEE_SAMPLE_ENVMAP
        light += color.rgb;
        nee_pdf.envmap_pdf = scene_metadata.environment_proj >= 0 ? sample_environment_map_pdf(view) : 0.0f;
#else
        mat.emission += color.rgb;
#endif
        return false;
    }
}

vec3 sample_explicit_light(uvec4 rand_uint, vec3 pos, out vec3 out_dir, out float out_length, out float pdf, out int hit_type)
{
    float point_prob, triangle_prob, dir_prob, envmap_prob;
    get_nee_sampling_probabilities(point_prob, triangle_prob, dir_prob, envmap_prob);

    vec4 u = vec4(rand_uint) * INV_UINT32_MAX;

    hit_type = -1;
    if(false) {}
#ifdef NEE_SAMPLE_POINT_LIGHTS
    else if((u.w -= point_prob) < 0)
    { // Sample point light
        hit_type = 0;
        const int light_count = int(scene_metadata.point_light_count);
        int light_index = 0;
        float weight = 0;
        random_sample_point_light(pos, u.z, light_count, weight, light_index);

        point_light pl = point_lights.lights[light_index];
        vec3 color;
        sample_point_light(pl, u.xy, pos, out_dir, out_length, color, pdf);
        pdf *= point_prob / weight;
        return color;
    }
#endif
#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
    else if((u.w -= triangle_prob) < 0)
    { // Sample triangle light
        hit_type = 3;
        const int light_count = int(scene_metadata.tri_light_count);
        int light_index = clamp(int(u.z*light_count), 0, light_count-1);
        tri_light tl = tri_lights.lights[light_index];
        vec3 A = tl.pos[0]-pos;
        vec3 B = tl.pos[1]-pos;
        vec3 C = tl.pos[2]-pos;

        vec3 color = r9g9b9e5_to_rgb(tl.emission_factor);

        float tri_pdf = 0.0f;
        out_dir = sample_triangle_light(u.xy, A, B, C, tri_pdf);
        out_length = ray_plane_intersection_dist(out_dir, A, B, C);
        if(isinf(tri_pdf) || tri_pdf <= 0 || out_length <= control.min_ray_dist || any(isnan(out_dir)))
        { // Same triangle, trying to intersect itself... Or zero-area degenerate triangle.
            pdf = 1.0f;
            out_dir = vec3(0);
            return vec3(0);
        }

        if(tl.emission_tex_id >= 0)
        { // Textured emissive triangle, so read texture.
            vec3 bary = get_barycentric_coords(out_dir*out_length, A, B, C);
            vec2 uv =
                bary.x * unpackHalf2x16(tl.uv[0]) +
                bary.y * unpackHalf2x16(tl.uv[1]) +
                bary.z * unpackHalf2x16(tl.uv[2]);
            color *= texture(textures[nonuniformEXT(tl.emission_tex_id)], uv).rgb;
        }

        // Prevent shadow ray from intersecting with the target triangle
        out_length -= control.min_ray_dist;

        pdf = triangle_prob * tri_pdf / light_count;
        return color;
    }
#endif
#ifdef NEE_SAMPLE_ENVMAP
    else if((u.w -= envmap_prob) < 0)
    { // Sample envmap
        hit_type = 2;
        vec3 color = sample_environment_map(rand_uint.xyz, out_dir, out_length, pdf);
        pdf *= envmap_prob;
        return color;
    }
#endif
#ifdef NEE_SAMPLE_DIRECTIONAL_LIGHTS
    else if((u.w -= dir_prob) < 0)
    { // Sample directional light
        hit_type = 1;
        const int light_count = int(scene_metadata.directional_light_count);
        int light_index = clamp(int(u.z*light_count), 0, light_count-1);

        directional_light dl = directional_lights.lights[light_index];
        out_length = RAY_MAX_DIST;
        vec3 color;
        sample_directional_light(dl, u.xy, out_dir, color, pdf);
        pdf *= dir_prob / light_count;
        return color;
    }
#endif
    // Should never be reached, hopefully.
    return vec3(0);
}

void correct_lobes_for_normal_map(vec3 sample_dir, vec3 geometric_normal, inout bsdf_lobes lobes)
{
    if(dot(geometric_normal, sample_dir) < 0)
    {
        lobes.diffuse = 0;
        lobes.dielectric_reflection = 0;
        lobes.metallic_reflection = 0;
    }
    else lobes.transmission = 0;
}

vec3 next_event_estimation(
    uvec4 rand_uint,
    mat3 tbn, vec3 shading_view, vec3 view, sampled_material mat,
    pt_vertex_data v,
    inout bsdf_lobes lobes
){
#if defined(NEE_SAMPLE_POINT_LIGHTS) || defined(NEE_SAMPLE_DIRECTIONAL_LIGHTS) || defined(NEE_SAMPLE_EMISSIVE_TRIANGLES) || defined(NEE_SAMPLE_ENVMAP)
    vec3 out_dir;
    float out_length = 0.0f;
    float light_pdf;
    // Sample lights
    int hit_type = -1;
#ifdef LIGHT_TREE_SET
    light_sample s = sample_light(
        rand_uint,
        v.pos,
        v.mapped_normal,
        1.0f,
        control.min_ray_dist,
        RAY_MAX_DIST
    );
    vec3 contrib = s.color;
    light_pdf = s.pdf;
    out_dir = s.dir;
    out_length = s.dist;
    hit_type = (s.instance_id == DIRECTIONAL_LIGHT_INSTANCE_ID || s.instance_id == POINT_LIGHT_INSTANCE_ID) ? 0 : 2;
#else
    vec3 contrib = sample_explicit_light(rand_uint, v.pos, out_dir, out_length, light_pdf, hit_type);
#endif

    bool opaque = mat.transmittance < 0.0001f;
    if(dot(v.hard_normal, out_dir) < 0 && opaque) contrib = vec3(0);

    vec3 shading_light = out_dir * tbn;
    lobes = bsdf_lobes(0,0,0,0);
#ifdef RADIANCE_CASCADES_SET
    ggx_brdf(shading_light, shading_view, mat, lobes);
#else
    float bsdf_pdf = material_bsdf_pdf(shading_light, shading_view, mat, lobes);
#endif

    correct_lobes_for_normal_map(out_dir, v.hard_normal, lobes);

    if(any(greaterThan(contrib, vec3(0.0f))))
        contrib *= shadow_ray(v.pos, control.min_ray_dist, out_dir, out_length);

#ifdef RADIANCE_CASCADES_SET
#ifdef HAS_AREA_LIGHTS
    float rc_pdf = radiance_cascades_pdf(v.pos, v.mapped_normal, -view, 
        hit_type < 2,
        mat.roughness,
        mix(0.04, 1.0, mat.metallic),
        rgb_to_luminance(mat.albedo.rgb) * (1.0-mat.metallic),
        out_dir
    );
    contrib /= nee_mis_pdf(light_pdf, rc_pdf);
#else
    contrib /= abs(light_pdf);
#endif
#else
    contrib /= nee_mis_pdf(light_pdf, bsdf_pdf);
#endif

    return contrib;
#else
    return vec3(0);
#endif
}

// This is used to remove invalid ray directions, which are caused by normal
// mapping.
float ray_visibility(vec3 view, pt_vertex_data v)
{
    vec3 h = v.mapped_normal + v.smooth_normal;
    float vh = dot(view, h);
    float nm = dot(v.mapped_normal, v.smooth_normal);
    return step((1-nm) * dot(h, h), 2.0f * vh * vh);
}

float clamp_contribution_mul(vec3 contrib)
{
    if(control.indirect_clamping > 0.0f)
    {
        float m = rgb_to_luminance(contrib);
        if(m > control.indirect_clamping)
            return control.indirect_clamping / m;
    }
    return 1;
}

#ifdef DISTRIBUTION_DATA_BINDING
void write_color_outputs(
#ifdef DEMODULATED_OUTPUT
    vec4 diffuse,
    vec4 reflection
#else
    vec3 color
#endif
){
    // Write all outputs
    ivec3 p = ivec3(get_write_pixel_pos(get_camera()));
#if DISTRIBUTION_STRATEGY != 0
    if(p != ivec3(-1))
#endif
    {
        uint prev_samples = distribution.samples_accumulated + control.previous_samples;

#ifdef DEMODULATED_OUTPUT
        accumulate_gbuffer_diffuse(diffuse, p, control.samples, prev_samples);
        accumulate_gbuffer_reflection(reflection, p, control.samples, prev_samples);
#else
        // TODO: Support transparent backgrounds again, somehow.
        const float alpha = 1.0;
        accumulate_gbuffer_color(vec4(color, alpha), p, control.samples, prev_samples);
#endif
    }
}

void write_hit_outputs(
    pt_vertex_data first_hit_vertex,
    sampled_material first_hit_material
){
    // Write outputs
    ivec3 p = ivec3(get_write_pixel_pos(get_camera()));
#if DISTRIBUTION_STRATEGY != 0
    if(p != ivec3(-1))
#endif
    {
        write_gbuffer_albedo(first_hit_material.albedo, p);
        write_gbuffer_material(first_hit_material, p);
        write_gbuffer_normal(first_hit_vertex.mapped_normal, p);
        write_gbuffer_pos(first_hit_vertex.pos, p);
        #ifdef CALC_PREV_VERTEX_POS
        write_gbuffer_screen_motion(
            get_camera_projection(get_prev_camera(), first_hit_vertex.prev_pos),
            p
        );
        #endif
        write_gbuffer_instance_id(first_hit_vertex.instance_id, p);
    }
}
#endif

void evaluate_ray(
    inout local_sampler lsampler,
    vec3 pos,
    vec3 view,
#ifdef DEMODULATED_OUTPUT
    inout vec4 diffuse,
    inout vec4 reflection,
#else
    inout vec3 color,
#endif
    //vec4 cliprule,
    bool write_first_hit_info
){
    vec3 attenuation = vec3(1);

    float regularization = 1.0f;
    float bsdf_pdf = 0.0f;
#ifdef DEMODULATED_OUTPUT
    bsdf_lobes primary_lobes = bsdf_lobes(0,0,0,1);
#endif
    pcg4d(lsampler.rs.seed);
    vec3 prev_normal = vec3(0);
    for(uint bounce = 0; bounce < MAX_BOUNCES; ++bounce)
    {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq,
            tlas,
            gl_RayFlagsNoneEXT,
            //gl_RayFlagsCullNoOpaqueEXT,
            //gl_RayFlagsOpaqueEXT|gl_RayFlagsSkipAABBEXT,
            //gl_RayFlagsCullBackFacingTrianglesEXT,
#ifdef HIDE_LIGHTS
            bounce == 0 ? 0xFF^0x02 : 0xFF,
#else
            0xFF,
#endif
            pos,
            bounce == 0 ? 0.0f : control.min_ray_dist,
            view,
            RAY_MAX_DIST
        );

        hit_info payload = trace_ray_query(rq, lsampler.rs.seed.x);

        pt_vertex_data v;
        sampled_material mat;
        intersection_pdf nee_pdf;
        vec3 light;
        bool include_directional_lights = false;
#ifdef HIDE_LIGHTS
        if (bounce == 0) include_directional_lights = false;
#endif
        bool terminal = !get_intersection_info(payload, pos, view, include_directional_lights, v, nee_pdf, mat, light) || bounce == MAX_BOUNCES-1;

        //if (bounce == 0 && distance(cliprule.xyz, v.pos) > cliprule.w)
        //    break;

        // Get rid of the attenuation by multiplying with bsdf_pdf, and use
        // mis_pdf instead.
        float mis_pdf = bsdf_mis_pdf(nee_pdf, bsdf_pdf, pos, prev_normal);
        float mis_weight = 1.0f;
        if(bsdf_pdf != 0)
        {
            attenuation /= bsdf_pdf;
            mis_weight = bsdf_pdf / mis_pdf;
        }

        light = attenuation * mis_weight * (mat.emission + light);

#ifndef INDIRECT_CLAMP_FIRST_BOUNCE
        if(bounce != 0)
#endif
        {
            light *= clamp_contribution_mul(light);
        }

        if(bounce == 0 && write_first_hit_info)
        {
            mat.emission = light;
#ifdef DISTRIBUTION_DATA_BINDING
            write_hit_outputs(v, mat);
#endif
        }

#ifdef USE_WHITE_ALBEDO_ON_FIRST_BOUNCE
        mat.albedo.rgb = vec3(1);
#endif

#ifdef DEMODULATED_OUTPUT
        add_demodulated_color(primary_lobes, light, diffuse.rgb, reflection.rgb);
#else
        color.rgb += light;
#endif

#ifdef PATH_SPACE_REGULARIZATION
        // Regularization strategy inspired by "Optimised Path Space Regularisation", 2021 Weier et al.
        // I'm using the BSDF PDF instead of roughness, which seems to be more
        // effective at reducing fireflies.
        if(bsdf_pdf != 0.0f)
            regularization *= max(1 - control.regularization_gamma / pow(bsdf_pdf, 0.25f), 0.0f);
        mat.roughness = 1.0f - ((1.0f - mat.roughness) * regularization);
#endif

        mat3 tbn = create_tangent_space(v.mapped_normal);
        vec3 shading_view = view_to_tangent_space(view, tbn);

        if(!terminal)
        {
            // Do NEE ray
            bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
            vec3 radiance = attenuation * next_event_estimation(
                generate_ray_sample_uint(lsampler, bounce*2), tbn, shading_view, view,
                mat, v, lobes
            );
#ifdef DEMODULATED_OUTPUT
            if(bounce == 0) primary_lobes = lobes;
            else
#endif
                radiance *= modulate_bsdf(mat, lobes);
#ifdef INDIRECT_CLAMP_FIRST_BOUNCE
            if(bounce != 0)
#endif
                radiance *= clamp_contribution_mul(radiance);
#ifdef DEMODULATED_OUTPUT
            add_demodulated_color(primary_lobes, radiance, diffuse.rgb, reflection.rgb);
            if(bounce == 1)
                diffuse.a = reflection.a = 1.0f / length(v.pos - pos);
#else
            color.rgb += radiance;
#endif
        }

        if(terminal) break;

        // Lastly, figure out the next ray and assign proper attenuation for it.
        bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
#ifdef RADIANCE_CASCADES_SET
        uvec4 ray_sample = generate_ray_sample_uint(lsampler, bounce*2+1);
        view = sample_radiance_cascades(ray_sample.x, v.pos, tbn[2], -view, mat.roughness, mix(0.04, 1.0, mat.metallic),
            rgb_to_luminance(mat.albedo.rgb) * (1.0-mat.metallic), bsdf_pdf);
        ggx_bsdf(view * tbn, shading_view, mat, lobes);
#else
        vec4 ray_sample = generate_ray_sample(lsampler, bounce*2+1);
        material_bsdf_sample(ray_sample, shading_view, mat, view, lobes, bsdf_pdf);
        view = tbn * view;
#endif

        correct_lobes_for_normal_map(view, v.hard_normal, lobes);

        if(bsdf_pdf < 0)
            break;

#ifdef DEMODULATED_OUTPUT
        if(bounce == 0) primary_lobes = lobes;
        else
#endif
            attenuation *= modulate_bsdf(mat, lobes);

        float visibility = ray_visibility(view, v);
        pos = v.pos;
        prev_normal = v.mapped_normal;

#ifdef USE_RUSSIAN_ROULETTE
        // This condition is fairly arbitrary again.
        float qi = min(1.0f, 1.0f / control.russian_roulette_delta);
        if(ray_sample.w > qi) break;
        else visibility /= qi;
#endif
        if(max(attenuation.x, max(attenuation.y, attenuation.z)) <= 0.0f) break;
    }
}

void evaluate_ray_matched(
    inout local_sampler lsampler,
    vec3 pos,
    vec3 view,
#ifdef DEMODULATED_OUTPUT
    inout vec4 diffuse,
    inout vec4 reflection,
#else
    inout vec3 color,
#endif
    bool write_first_hit_info
){
    vec3 attenuation = vec3(1);

    float regularization = 1.0f;
#ifdef DEMODULATED_OUTPUT
    bsdf_lobes primary_lobes = bsdf_lobes(0,0,0,1);
#endif
    pcg4d(lsampler.rs.seed);
    for(uint bounce = 0; bounce < MAX_BOUNCES-1; ++bounce)
    {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq,
            tlas,
            //gl_RayFlagsNoneEXT,
            //gl_RayFlagsCullNoOpaqueEXT,
            gl_RayFlagsOpaqueEXT|gl_RayFlagsSkipAABBEXT,
            //gl_RayFlagsCullBackFacingTrianglesEXT,
#ifdef HIDE_LIGHTS
            bounce == 0 ? 0xFF^0x02 : 0xFF,
#else
            0xFF,
#endif
            pos,
            bounce == 0 ? 0.0f : control.min_ray_dist,
            view,
            RAY_MAX_DIST
        );

        hit_info payload = trace_ray_query(rq, lsampler.rs.seed.x);

        pt_vertex_data v;
        sampled_material mat;
        intersection_pdf nee_pdf;
        vec3 light;
        bool terminal = !get_intersection_info(payload, pos, view, true, v, nee_pdf, mat, light);

        /*
        if(bounce == 0)
        {
#ifdef RADIANCE_CASCADES_SET
            color = mat.albedo.rgb * eval_diffuse_radiance_cascades(
                v.pos,
                v.smooth_normal,
                -view,
                mat.roughness,
                mix(mat.f0, 1.0, mat.metallic)
            );
            return;
#endif
        }
        */

        /*
        light *= attenuation;

#ifdef DEMODULATED_OUTPUT
        add_demodulated_color(primary_lobes, light, diffuse.rgb, reflection.rgb);
#else
        color.rgb += light;
#endif
        */

        if(terminal) break;

        mat3 tbn = create_tangent_space(v.mapped_normal);
        vec3 shading_view = view_to_tangent_space(view, tbn);

        {
            // Do NEE ray
            bsdf_lobes lobes = bsdf_lobes(0,0,0,0);

            vec3 radiance = attenuation * next_event_estimation(
                generate_ray_sample_uint(lsampler, bounce*2), tbn, shading_view, view,
                mat, v, lobes
            );

#ifdef DEMODULATED_OUTPUT
            if(bounce == 0) primary_lobes = lobes;
            else
#endif
                radiance *= modulate_bsdf(mat, lobes);

#ifdef DEMODULATED_OUTPUT
            add_demodulated_color(primary_lobes, radiance, diffuse.rgb, reflection.rgb);
            if(bounce == 1)
                diffuse.a = reflection.a = 1.0f / length(v.pos - pos);
#else
            if (!any(isnan(radiance)))
                color.rgb += radiance;
#endif
        }

        // Only NEE contribution for last bounce, to match SIByl for measurements.
        if(bounce+1 < MAX_BOUNCES-1)
        {
            // Lastly, figure out the next ray and assign proper attenuation for it.
#ifdef RADIANCE_CASCADES_SET
    #if 1
            bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
            float pdf = 0.0f;
            uvec4 ray_sample = generate_ray_sample_uint(lsampler, bounce*2+1);
            view = sample_radiance_cascades(ray_sample.x, v.pos, tbn[2], -view, mat.roughness, mix(0.04, 1.0, mat.metallic),
                rgb_to_luminance(mat.albedo.rgb) * (1.0-mat.metallic), pdf);
            ggx_bsdf(view * tbn, shading_view, mat, lobes);
    #else // Single-sample MIS. Really slow and more noisy but with less fireflies.
            uvec4 ray_sample_u = generate_ray_sample_uint(lsampler, bounce*2+1);
            uint rnd = ray_sample_u.z;
            lcg(rnd);
            bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
            float pdf;
            if (rnd * INV_UINT32_MAX < 0.5)
            {
                float rc_pdf;
                view = sample_radiance_cascades(ray_sample_u.x, v.pos, tbn[2], -view, mat.roughness, mix(0.04, 1.0, mat.metallic),
                    rgb_to_luminance(mat.albedo.rgb) * (1.0-mat.metallic), rc_pdf);
                float bsdf_pdf = material_bsdf_pdf(view * tbn, shading_view, mat, lobes);
                pdf = 0.5 * (rc_pdf + bsdf_pdf);
            }
            else
            {
                vec3 prev_view = view;
                vec4 ray_sample = vec4(ray_sample_u) * INV_UINT32_MAX;
                float bsdf_pdf;
                material_bsdf_sample(ray_sample, shading_view, mat, view, lobes, bsdf_pdf);
                view = tbn * view;

                float rc_pdf = radiance_cascades_pdf(v.pos, v.mapped_normal, -prev_view,
                    true,
                    mat.roughness,
                    mix(0.04, 1.0, mat.metallic),
                    rgb_to_luminance(mat.albedo.rgb) * (1.0-mat.metallic),
                    view
                );
                pdf = 0.5 * (rc_pdf + bsdf_pdf);
            }
    #endif
#else
            bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
            float pdf = 0.0f;

            vec4 ray_sample = generate_ray_sample(lsampler, bounce*2+1);
            material_bsdf_sample(ray_sample, shading_view, mat, view, lobes, pdf);
            view = tbn * view;
#endif

            if(pdf <= 0)
                break;

            attenuation /= pdf;

#ifdef DEMODULATED_OUTPUT
            if(bounce == 0) primary_lobes = lobes;
            else
#endif
                attenuation *= modulate_bsdf(mat, lobes);

            pos = v.pos;
        }
    }
}

#endif

#ifdef DISTRIBUTION_DATA_BINDING
void get_world_camera_ray(inout local_sampler lsampler, out vec3 origin, out vec3 dir)
{
    vec2 cam_offset = vec2(0.0);
    if(control.antialiasing == 1)
    {
#if defined(USE_POINT_FILTER)
        cam_offset = vec2(0.0);
#elif defined(USE_BOX_FILTER)
        cam_offset = generate_film_sample(lsampler) * 2.0f - 1.0f;
#elif defined(USE_BLACKMAN_HARRIS_FILTER)
        cam_offset = sample_blackman_harris_concentric_disk(
            generate_film_sample(lsampler).xy
        ) * 2.0f;
#else
#error "Unknown filter type"
#endif
        cam_offset *= 2.0f * control.film_radius;
    }

    const camera_data cam = get_camera();
    get_screen_camera_ray(
        cam, cam_offset,
#ifdef USE_DEPTH_OF_FIELD
        generate_film_sample(lsampler),
#else
        vec2(0.5f),
#endif
        origin, dir
    );
}

#endif
