#ifndef PATH_TRACER_GLSL
#define PATH_TRACER_GLSL

#ifdef USE_SCREEN_MOTION_TARGET
#define CALC_PREV_VERTEX_POS
#endif

#include "rt.glsl"
#include "sampling.glsl"

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

struct hit_payload
{
    // Needed by anyhit alpha handling.
    uint random_seed;

    // Negative if the ray escaped the scene or otherwise died. Otherwise, it's
    // the index of the mesh instance, and primitive_id and barycentrics are
    // valid as well.
    int instance_id;
    // If instance_id is non-negative, this is the triangle index. Otherwise,
    // if primitive_id is non-negative, it is the light index. If it is negative,
    // that means that the ray escaped the scene and hit the environment map
    // instead.
    int primitive_id;

    // Barycentric coordinates to the triangle that was hit.
    vec2 barycentrics;
};

struct intersection_pdf
{
    float point_light_pdf;
    float directional_light_pdf;
    float tri_light_pdf;
    float envmap_pdf;
};

#ifdef TLAS_BINDING
#include "ggx.glsl"

layout(location = 0) rayPayloadEXT hit_payload payload;
layout(location = 1) rayPayloadEXT float shadow_visibility;

float shadow_ray(vec3 pos, float min_dist, vec3 dir, float max_dist)
{
    shadow_visibility = 1.0f;
    traceRayEXT(
        tlas,
        gl_RayFlagsNoneEXT,
        0x02^0xFF, // Exclude lights from shadow rays
        1,
        0,
        1,
        pos,
        min_dist,
        dir,
        max_dist,
        1
    );

    return shadow_visibility;
}

#ifdef ENVIRONMENT_MAP_ALIAS_TABLE_BINDING
// Based on CC0 code from https://gist.github.com/juliusikkala/6c8c186f0150fe877a55cee4d266b1b0
vec3 sample_environment_map(
    uvec3 rand,
    out vec3 shadow_ray_direction,
    out float shadow_ray_length,
    out float pdf
){
    vec3 color = control.environment_factor.rgb;
    if(control.environment_proj >= 0)
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
    if(control.environment_proj >= 0)
    {
        uvec2 size = textureSize(environment_map_tex, 0).xy;
        const uint pixel_count = size.x * size.y;
        uint i = latlong_direction_to_pixel_id(dir, ivec2(size));
        alias_table_entry at = environment_map_alias_table.entries[i];
        return at.pdf;
    }
    else return 1.0f / (4.0f * M_PI);
}
#endif

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
    if(control.environment_proj >= 0) envmap = NEE_SAMPLE_ENVMAP;
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

float bsdf_mis_pdf(
    intersection_pdf nee_pdf,
    float bsdf_pdf
){
    if(bsdf_pdf == 0.0f) return 1.0f;

    float point_prob, triangle_prob, dir_prob, envmap_prob;
    get_nee_sampling_probabilities(point_prob, triangle_prob, dir_prob, envmap_prob);

    float avg_nee_pdf =
        nee_pdf.directional_light_pdf * dir_prob / max(scene_metadata.directional_light_count, 1) +
        nee_pdf.tri_light_pdf * triangle_prob / max(scene_metadata.tri_light_count, 1) +
        nee_pdf.envmap_pdf * envmap_prob +
        nee_pdf.point_light_pdf * point_prob / max(scene_metadata.point_light_count, 1);

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
    vec3 origin,
    vec3 view,
    out pt_vertex_data v,
    out intersection_pdf nee_pdf,
    out sampled_material mat,
    out vec3 light
){
    nee_pdf.point_light_pdf = 0;
    nee_pdf.directional_light_pdf = 0;
    nee_pdf.tri_light_pdf = 0;
    nee_pdf.envmap_pdf = 0;
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
        vec4 color = control.environment_factor;
        if(control.environment_proj >= 0)
        {
            vec2 uv = vec2(0);
            uv.y = asin(-view.y)/M_PI+0.5f;
            uv.x = atan(view.z, view.x)/(2*M_PI)+0.5f;
            color.rgb *= texture(environment_map_tex, uv).rgb;
        }

        mat.emission = vec3(0);
        light = vec3(0);
        for(uint i = 0; i < scene_metadata.directional_light_count; ++i)
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
        nee_pdf.envmap_pdf = control.environment_proj >= 0 ? sample_environment_map_pdf(view) : 0.0f;
#else
        mat.emission += color.rgb;
#endif
        return false;
    }
}

vec3 sample_explicit_light(uvec4 rand_uint, vec3 pos, out vec3 out_dir, out float out_length, out float pdf)
{
    float point_prob, triangle_prob, dir_prob, envmap_prob;
    get_nee_sampling_probabilities(point_prob, triangle_prob, dir_prob, envmap_prob);

    vec4 u = ldexp(vec4(rand_uint), ivec4(-32));

    if(false) {}
#ifdef NEE_SAMPLE_POINT_LIGHTS
    else if((u.w -= point_prob) < 0)
    { // Sample point light
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
        const int light_count = int(scene_metadata.tri_light_count);
        int light_index = clamp(int(u.z*light_count), 0, light_count-1);
        tri_light tl = tri_lights.lights[light_index];
        vec3 A = tl.pos[0]-pos;
        vec3 B = tl.pos[1]-pos;
        vec3 C = tl.pos[2]-pos;

        vec3 color = tl.emission_factor;

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
            vec2 uv = bary.x * tl.uv[0] + bary.y * tl.uv[1] + bary.z * tl.uv[2];
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
        vec3 color = sample_environment_map(rand_uint.xyz, out_dir, out_length, pdf);
        pdf *= envmap_prob;
        return color;
    }
#endif
#ifdef NEE_SAMPLE_DIRECTIONAL_LIGHTS
    else if((u.w -= dir_prob) < 0)
    { // Sample directional light
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

void next_event_estimation(
    uvec4 rand_uint,
    mat3 tbn, vec3 shading_view, sampled_material mat,
    pt_vertex_data v,
    inout vec3 diffuse_radiance,
    inout vec3 specular_radiance
){
#if defined(NEE_SAMPLE_POINT_LIGHTS) || defined(NEE_SAMPLE_DIRECTIONAL_LIGHTS) || defined(NEE_SAMPLE_EMISSIVE_TRIANGLES) || defined(NEE_SAMPLE_ENVMAP)
    if(false
#ifdef NEE_SAMPLE_POINT_LIGHTS
        || scene_metadata.point_light_count > 0
#endif
#ifdef NEE_SAMPLE_DIRECTIONAL_LIGHTS
        || scene_metadata.directional_light_count > 0
#endif
#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
        || scene_metadata.tri_light_count > 0
#endif
#ifdef NEE_SAMPLE_ENVMAP
        || control.environment_proj >= 0
#endif
    ){
        vec3 out_dir;
        float out_length = 0.0f;
        float light_pdf;
        // Sample lights
        vec3 contrib = sample_explicit_light(rand_uint, v.pos, out_dir, out_length, light_pdf);

        vec3 shading_light = out_dir * tbn;
        vec3 d, s;
        float bsdf_pdf = material_bsdf_pdf(shading_light, shading_view, mat, d, s);
        bool opaque = mat.transmittance < 0.0001f;
        d = dot(v.hard_normal, out_dir) < 0 && opaque ? vec3(0) : d;
        s = dot(v.hard_normal, out_dir) < 0 && opaque ? vec3(0) : s;
        shadow_terminator_fix(d, s, shading_light.z, mat);

        // TODO: Check if this conditional just hurts performance
        if(any(greaterThan((d+s) * contrib, vec3(0.0001f))))
            contrib *= shadow_ray(v.pos, control.min_ray_dist, out_dir, out_length);

        contrib /= nee_mis_pdf(light_pdf, bsdf_pdf);
        diffuse_radiance += d * contrib;
        specular_radiance += s * contrib;
    }
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

void evaluate_ray(
    inout local_sampler lsampler,
    vec3 pos,
    vec3 view,
    out vec3 color,
    out vec3 direct,
    out vec3 diffuse,
    out pt_vertex_data first_hit_vertex,
    out sampled_material first_hit_material
){
    vec3 diffuse_attenuation_ratio = vec3(1);
    vec3 attenuation = vec3(1);
    color = vec3(0);
    direct = vec3(0);
    diffuse = vec3(0);

    float regularization = 1.0f;
    float bsdf_pdf = 0.0f;
    payload.random_seed = pcg4d(lsampler.rs.seed).x;
    for(uint bounce = 0; bounce < MAX_BOUNCES; ++bounce)
    {
        traceRayEXT(
            tlas,
            gl_RayFlagsNoneEXT,
#ifdef HIDE_LIGHTS
            bounce == 0 ? 0xFF^0x02 : 0xFF,
#else
            0xFF,
#endif
            0,
            0,
            0,
            pos,
            bounce == 0 ? 0.0f : control.min_ray_dist,
            view,
            RAY_MAX_DIST,
            0
        );

        pt_vertex_data v;
        sampled_material mat;
        intersection_pdf nee_pdf;
        vec3 light;
        bool terminal = !get_intersection_info(pos, view, v, nee_pdf, mat, light) || bounce == MAX_BOUNCES-1;


        // Get rid of the attenuation by multiplying with bsdf_pdf, and use
        // mis_pdf instead.
        float mis_pdf = bsdf_mis_pdf(nee_pdf, bsdf_pdf);
        if(bsdf_pdf != 0)
        {
            attenuation /= bsdf_pdf;
            light = light / mis_pdf * bsdf_pdf;
        }

        vec3 diffuse_radiance = vec3(0);
        vec3 specular_radiance = mat.emission + light;

        if(bounce == 0)
        {
            first_hit_vertex = v;
            first_hit_material = mat;
        }

#ifdef PATH_SPACE_REGULARIZATION
        // Regularization strategy inspired by "Optimised Path Space Regularisation", 2021 Weier et al.
        // I'm using the BSDF PDF instead of roughness, which seems to be more
        // effective at reducing fireflies.
        if(bsdf_pdf != 0.0f)
            regularization *= max(1 - control.regularization_gamma / pow(bsdf_pdf, 0.25f), 0.0f);
        mat.roughness = 1.0f - ((1.0f - mat.roughness) * regularization);
#endif

        mat3 tbn = create_tangent_space(v.mapped_normal);
        vec3 shading_view = -view * tbn;

        // A lot of the stuff below assumes that the view direction is on the same
        // side as the normal. If not, everything breaks. Which is why this check
        // exists. Normal maps can cause these degenerate cases.
        if(shading_view.z < 0.00001f)
            shading_view = vec3(shading_view.xy, max(shading_view.z, 0.00001f));

        shading_view = normalize(shading_view);

        if(!terminal)
        {
            // Do NEE ray
            next_event_estimation(
                generate_ray_sample_uint(lsampler, bounce*2), tbn, shading_view,
                mat, v, diffuse_radiance, specular_radiance
            );
        }

        // Then, calculate contribution to pixel color from current bounce.
        vec3 contribution = attenuation * (diffuse_radiance * mat.albedo.rgb + specular_radiance);
        if(
#ifndef INDIRECT_CLAMP_FIRST_BOUNCE
            bounce != 0 &&
#endif
            control.indirect_clamping > 0.0f
        ){
            contribution = min(contribution, vec3(control.indirect_clamping));
        }

        color += contribution;
        diffuse += bounce == 0 ? diffuse_radiance : diffuse_attenuation_ratio * contribution;
        if(bounce == 0) direct += contribution;
        if(terminal) break;

        // Lastly, figure out the next ray and assign proper attenuation for it.
        vec3 diffuse_weight = vec3(1.0f);
        vec3 specular_weight = vec3(1.0f);
        vec4 ray_sample = generate_ray_sample(lsampler, bounce*2+1);
        material_bsdf_sample(ray_sample, shading_view, mat, view, diffuse_weight, specular_weight, bsdf_pdf);
        view = tbn * view;

        shadow_terminator_fix(diffuse_weight, specular_weight, dot(view, v.mapped_normal), mat);

        float visibility = ray_visibility(view, v);
        pos = v.pos;
#ifdef USE_RUSSIAN_ROULETTE
        // This condition is fairly arbitrary again.
        float qi = min(1.0f, 1.0f / control.russian_roulette_delta);
        if(ray_sample.w > qi) break;
        else visibility /= qi;
#endif
        attenuation *= (diffuse_weight * mat.albedo.rgb + specular_weight) * visibility;
        if(bounce == 0)
        {
            diffuse_attenuation_ratio = diffuse_weight / (diffuse_weight * mat.albedo.rgb + specular_weight);
            if(diffuse_weight.r < 1e-7) diffuse_attenuation_ratio.r = 0;
            if(diffuse_weight.g < 1e-7) diffuse_attenuation_ratio.g = 0;
            if(diffuse_weight.b < 1e-7) diffuse_attenuation_ratio.b = 0;
        }

        if(max(attenuation.x, max(attenuation.y, attenuation.z)) <= 0.0f) break;
    }

#ifdef USE_WHITE_ALBEDO_ON_FIRST_BOUNCE
    vec3 specular = color - diffuse * first_hit_material.albedo.rgb;
    specular /= mix(vec3(1), first_hit_material.albedo.rgb, first_hit_material.metallic);
    color = diffuse + specular;
#endif
}

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

void write_all_outputs(
    vec3 color,
    vec3 direct,
    vec3 diffuse,
    pt_vertex_data first_hit_vertex,
    sampled_material first_hit_material
){
    // Write all outputs
    ivec3 p = ivec3(get_write_pixel_pos(get_camera()));
#if DISTRIBUTION_STRATEGY != 0
    if(p != ivec3(-1))
#endif
    {
        uint prev_samples = distribution.samples_accumulated + control.previous_samples;

#ifdef USE_TRANSPARENT_BACKGROUND
        const float alpha = first_hit_material.albedo.a;
#else
        const float alpha = 1.0;
#endif

        accumulate_gbuffer_color(vec4(color, alpha), p, 1, prev_samples);
        accumulate_gbuffer_direct(vec4(direct, alpha), p, 1, prev_samples);
        accumulate_gbuffer_diffuse(vec4(diffuse, alpha), p, 1, prev_samples);

        if(prev_samples == 0)
        { // Only write gbuffer for the first sample.
            ivec3 p = ivec3(get_write_pixel_pos(get_camera()));
            write_gbuffer_albedo(first_hit_material.albedo, p);
            write_gbuffer_material(
                vec2(first_hit_material.metallic, first_hit_material.roughness), p
            );
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
}

#endif

#endif

