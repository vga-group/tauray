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

#ifdef USE_PUSH_CONSTANTS
layout(push_constant, scalar) uniform push_constant_buffer
{
    uint samples;
    uint previous_samples;
    float min_ray_dist;
    float indirect_clamping;
    float film_radius;
    float russian_roulette_delta;
    int antialiasing;
    int environment_proj;
    vec4 environment_factor;
} control;
#endif

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
    uvec2 rand,
    out vec3 shadow_ray_direction,
    out float shadow_ray_length,
    out float pdf
){
    vec3 color = control.environment_factor.rgb;
    if(control.environment_proj >= 0)
    {
        uvec2 size = textureSize(environment_map_tex, 0).xy;
        const uint pixel_count = size.x * size.y;
        // Assuming pixel_count is a power of two or fairly small, this should
        // be okay-ish to do.
        int i = int(rand.x % pixel_count);
        alias_table_entry at = environment_map_alias_table.entries[i];
        pdf = at.pdf;
        if(rand.y > at.probability)
        {
            i = int(at.alias_id);
            pdf = at.alias_pdf;
        }

        // TODO: Uniformly sample texel instead of picking center like this!
        // ... But that might not be worth the effort, since the individual texels
        // would only be visible in very smooth surfaces, which this function can't
        // sample well anyway (they need BRDF sampling instead).
        shadow_ray_direction = pixel_id_to_latlong_direction(i, ivec2(size));

        ivec2 p = ivec2(i % size.x, i / size.x);
        vec2 uv = (vec2(p) + 0.5f)/vec2(size);

        color *= texture(environment_map_tex, vec2(uv.x, uv.y)).rgb;
    }
    else
    {
        pdf = 4.0f * M_PI;
        shadow_ray_direction = sample_sphere(ldexp(vec2(rand), ivec2(-32)));
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
    else return 4.0f * M_PI;
}
#endif


bool get_intersection_info(
    vec3 origin,
    vec3 view,
    out pt_vertex_data v,
    out sampled_material mat,
    out vec3 light
){
    if(payload.instance_id >= 0)
    {
        vertex_data vd = get_interpolated_vertex(
            view, payload.barycentrics,
            payload.instance_id,
            payload.primitive_id
        );
        light = vec3(0);
        mat = sample_material(payload.instance_id, vd);
        mat.albedo.a = 1.0; // Alpha blending was handled by the any-hit shader!
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
        light = get_spotlight_intensity(pl, view) * pl.color / (M_PI * pl.radius * pl.radius);

        v.pos = origin + payload.barycentrics.x * view;
        #ifdef CALC_PREV_VERTEX_POS
        v.prev_pos = v.pos; // TODO?
        #endif
        v.mapped_normal = normalize(v.pos - pl.pos);
        v.instance_id = -1;
        mat.albedo = vec4(0,0,0,1);
        mat.emission = vec3(0);
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

        light = vec3(0);
        for(uint i = 0; i < scene_metadata.directional_light_count; ++i)
        {
            directional_light dl = directional_lights.lights[i];
            if(dl.dir_cutoff >= 1.0f)
                continue;
            light += step(dl.dir_cutoff, dot(view, -dl.dir)) * dl.color /
                (2 * M_PI * (1 - dl.dir_cutoff));
        }
        v.instance_id = -1;
        v.pos = origin;
        #ifdef CALC_PREV_VERTEX_POS
        v.prev_pos = v.pos;
        #endif
        v.mapped_normal = -view;
        mat.albedo = vec4(0);
#ifdef IMPORTANCE_SAMPLE_ENVMAP
        mat.emission = vec3(0);
        light += color.rgb;
#else
        mat.emission = color.rgb;
#endif
        return false;
    }
}

vec3 sample_explicit_light(inout local_sampler lsampler, vec3 pos, out vec3 out_dir, out float out_length, inout float ratio)
{
#ifdef IMPORTANCE_SAMPLE_ENVMAP
    const float point_directional_split =
        (scene_metadata.directional_light_count > 0 || control.environment_proj >= 0) ?
        (scene_metadata.point_light_count > 0 ? 0.5f : 0.0f) :
        1.00001f;

    const float envmap_directional_split =
        scene_metadata.directional_light_count > 0 ?
        (control.environment_proj >= 0 ? 0.5f : 0.0f) :
        1.00001f;
#else
    const float point_directional_split = scene_metadata.directional_light_count > 0 ?
        (scene_metadata.point_light_count > 0 ? 0.5f : 0.0f) :
        1.00001f;

    const float envmap_directional_split = 0.0f;
#endif
    vec4 u = generate_uniform_random(lsampler.rs);

    if(u.z < point_directional_split)
    {
        // Sample point light
        u.z = u.z / point_directional_split;
        const int light_count = int(scene_metadata.point_light_count);
        int light_index = 0;
        float weight = 0;
        random_sample_point_light(pos, u.z, light_count, weight, light_index);
        weight /= point_directional_split;

        point_light pl = point_lights.lights[light_index];
        vec3 dir = pos - pl.pos;
        float dist2 = dot(dir, dir);
        vec3 ndir = dir * inversesqrt(dist2);
        float k = 1.0f - pl.radius*pl.radius/dist2;
        float dir_cutoff = k > 0 ? sqrt(k) : -1;
        out_dir = sample_cone(u.xy, -ndir, dir_cutoff);

        float b = dot(dir, out_dir);
        out_length = -b - sqrt(max(b * b - dist2 + pl.radius * pl.radius, 0.0f));

        vec3 color = get_spotlight_intensity(pl, out_dir) * pl.color;
        color /= pl.radius == 0.0f ? dist2 : pl.radius * pl.radius;
        if(pl.radius != 0.0f) color *= (1.0f - dir_cutoff) * 2.0f;
        else ratio = 1.0f; // Punctual lights can only be sampled through NEE.
        return color * weight;
    }
    else
    {
        // Sample directional light or envmap
        u.z = (u.z - point_directional_split)/(1-point_directional_split);

#ifdef IMPORTANCE_SAMPLE_ENVMAP
        if(u.z < envmap_directional_split)
        {
            float pdf = 1.0f;
            vec3 color = sample_environment_map(lsampler.rs.seed.xy, out_dir, out_length, pdf);
            pdf *= (1.0f-point_directional_split) * envmap_directional_split;
            return color / pdf;
        }
        else
#endif
        {
            const int light_count = int(scene_metadata.directional_light_count);
            int light_index = clamp(int(u.z*light_count), 0, light_count-1);
            float weight = light_count / ((1.0f - point_directional_split) * (1.0f - envmap_directional_split));

            directional_light dl = directional_lights.lights[light_index];
            vec3 dir = -dl.dir;
            out_length = RAY_MAX_DIST;
            out_dir = sample_cone(u.xy, dir, dl.dir_cutoff);
            if(dl.dir_cutoff >= 1.0f)
                ratio = 1.0f; // Punctual lights can only be sampled through NEE.
            return dl.color * weight;
        }
    }
}

void eval_explicit_lights(
    inout local_sampler lsampler,
    mat3 tbn, vec3 shading_view, sampled_material mat,
    pt_vertex_data v,
    float ratio,
    inout vec3 diffuse_radiance,
    inout vec3 specular_radiance
){
    vec3 out_dir;
    float out_length = 0.0f;
    // Sample lights
    vec3 contrib = sample_explicit_light(lsampler, v.pos, out_dir, out_length, ratio);

    vec3 shading_light = out_dir * tbn;
    vec3 d, s;
    ggx_bsdf(shading_light, shading_view, mat, d, s);
    bool opaque = mat.transmittance < 0.0001f;
    d = dot(v.hard_normal, out_dir) < 0 && opaque ? vec3(0) : d;
    s = dot(v.hard_normal, out_dir) < 0 && opaque ? vec3(0) : s;
    shadow_terminator_fix(d, s, shading_light.z, mat);

    // TODO: Check if this conditional just hurts performance
    if(any(greaterThan((d+s) * contrib, vec3(0.0001f))))
        contrib *= shadow_ray(v.pos, control.min_ray_dist, out_dir, out_length);

    diffuse_radiance += ratio * d * contrib;
    specular_radiance += ratio * s * contrib;
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

    // Used for implementing NEE
    float nee_light_ratio = 1.0f;

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
        vec3 light;
        bool terminal = !get_intersection_info(pos, view, v, mat, light);
#ifdef HIDE_LIGHTS
        if(bounce == 0) light = vec3(0);
#endif

        vec3 diffuse_radiance = vec3(0);
        vec3 specular_radiance = mat.emission + nee_light_ratio * light;

        if(bounce == 0)
        {
            first_hit_vertex = v;
            first_hit_material = mat;
        }

        mat3 tbn = create_tangent_space(v.mapped_normal);
        vec3 shading_view = -view * tbn;

        // A lot of the stuff below assumes that the view direction is on the same
        // side as the normal. If not, everything breaks. Which is why this check
        // exists. Normal maps can cause these degenerate cases.
        if(shading_view.z < 0.00001f)
            shading_view = vec3(shading_view.xy, max(shading_view.z, 0.00001f));

        shading_view = normalize(shading_view);

        // This heuristic lets NEE rays be less weighted for extremely smooth and
        // metallic materials.
        nee_light_ratio = mix(0.0, mix(0.5, 1.0, mat.metallic), pow(1.0f-mat.roughness, 200.0f));

        // Calculate radiance from the intersection point (i.e. direct lighting
        // mostly, whenever that is applicable.)
        if(
            !terminal &&
            (scene_metadata.directional_light_count > 0 ||
            scene_metadata.point_light_count > 0
#ifdef IMPORTANCE_SAMPLE_ENVMAP
            || control.environment_proj >= 0
#endif
            )
        ){
            // Do NEE ray
            eval_explicit_lights(
                lsampler, tbn, shading_view, mat, v, 1.0f - nee_light_ratio,
                diffuse_radiance, specular_radiance
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
        vec4 ray_sample = generate_ray_sample(lsampler, bounce);
        vec3 diffuse_weight = vec3(1.0f);
        vec3 specular_weight = vec3(1.0f);
        ggx_bsdf_sample(ray_sample.xyz, shading_view, mat, view, diffuse_weight, specular_weight);
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
}

#endif

#endif

