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
    float envmap_pdf;
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
    float regularization_gamma;
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

        ivec2 p = ivec2(i % size.x, i / size.x);
        vec2 off = ldexp(vec2(uvec2(rand*pixel_count)), ivec2(-32));
        vec2 uv = (vec2(p) + off)/vec2(size);

        shadow_ray_direction = uv_to_latlong_direction(uv);

        color *= texture(environment_map_tex, vec2(uv.x, uv.y)).rgb;
    }
    else
    {
        pdf = 1.0f / (4.0f * M_PI);
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
    else return 1.0f / (4.0f * M_PI);
}
#endif

void sample_point_light(
    point_light pl,
    vec2 u,
    vec3 pos,
    out vec3 out_dir,
    out float out_length,
    out vec3 color,
    out float pdf
){
    vec3 dir = pos - pl.pos;
    float dist2 = dot(dir, dir);
    float k = 1.0f - pl.radius * pl.radius / dist2;
    float dir_cutoff = k > 0 ? sqrt(k) : -1.0f;
    out_dir = sample_cone(u, -normalize(dir), dir_cutoff);

    float b = dot(dir, out_dir);
    out_length = -b - sqrt(max(b * b - dist2 + pl.radius * pl.radius, 0.0f));

    color = get_spotlight_intensity(pl, out_dir) * pl.color;

    if(pl.radius == 0.0f)
    {
        // We mark infinite PDFs with the minus sign on the NEE side.
        pdf = -dist2;
    }
    else
    {
        color /= pl.radius * pl.radius * M_PI;
        pdf = 1 / (2.0f * M_PI * (1.0f - dir_cutoff));
    }
}

float sample_point_light_pdf(point_light pl, vec3 pos)
{
    vec3 dir = pos - pl.pos;
    float dist2 = dot(dir, dir);
    float k = 1.0f - pl.radius * pl.radius / dist2;
    float dir_cutoff = k > 0 ? sqrt(k) : -1.0f;

    if(pl.radius == 0.0f) return 0;
    else return 1 / (2.0f * M_PI * (1.0f - dir_cutoff));
}

float sample_directional_light_pdf(directional_light dl)
{
    return dl.dir_cutoff >= 1.0f ? 0.0f : 1.0f / (2.0f * M_PI * (1.0f - dl.dir_cutoff));
}

void sample_directional_light(
    directional_light dl,
    vec2 u,
    out vec3 out_dir,
    out vec3 color,
    out float pdf
){
    out_dir = sample_cone(u, -dl.dir, dl.dir_cutoff);
    pdf = dl.dir_cutoff >= 1.0f ? -1.0f : 1.0f / (2.0f * M_PI * (1.0f - dl.dir_cutoff));
    color = pdf > 0 ? dl.color * pdf : dl.color;
}

float bsdf_mis_pdf(
    intersection_pdf nee_pdf,
    float bsdf_pdf
){
    if(bsdf_pdf == 0.0f) return 1.0f;

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

    float avg_nee_pdf = mix(
        mix(
            nee_pdf.directional_light_pdf / max(scene_metadata.directional_light_count, 1),
            nee_pdf.envmap_pdf,
            envmap_directional_split
        ),
        nee_pdf.point_light_pdf / max(scene_metadata.point_light_count, 1),
        point_directional_split
    );
    return (bsdf_pdf * bsdf_pdf + avg_nee_pdf * avg_nee_pdf) / bsdf_pdf;
}

float nee_mis_pdf(float nee_pdf, float bsdf_pdf)
{
    if(nee_pdf <= 0.0f) return -nee_pdf;

    return (nee_pdf * nee_pdf + bsdf_pdf * bsdf_pdf) / (nee_pdf);
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
    nee_pdf.envmap_pdf = 0;
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
        light = get_spotlight_intensity(pl, view) * pl.color / (pl.radius * pl.radius * M_PI);

        nee_pdf.point_light_pdf = sample_point_light_pdf(pl, origin);

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
            float visible = step(dl.dir_cutoff, dot(view, -dl.dir));
            light += visible * dl.color;
            nee_pdf.directional_light_pdf += visible * sample_directional_light_pdf(dl);
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
        nee_pdf.envmap_pdf = control.environment_proj >= 0 ? sample_environment_map_pdf(view) : 0.0f;
#else
        mat.emission = color.rgb;
#endif
        return false;
    }
}

vec3 sample_explicit_light(uvec3 rand_uint, vec3 pos, out vec3 out_dir, out float out_length, out float pdf)
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

    vec3 u = ldexp(vec3(rand_uint), ivec3(-32));

    if(u.z < point_directional_split)
    {
        // Sample point light
        u.z = u.z / point_directional_split;
        const int light_count = int(scene_metadata.point_light_count);
        int light_index = 0;
        float weight = 0;
        random_sample_point_light(pos, u.z, light_count, weight, light_index);

        point_light pl = point_lights.lights[light_index];
        vec3 color;
        sample_point_light(pl, u.xy, pos, out_dir, out_length, color, pdf);
        pdf *= point_directional_split / weight;
        return color;
    }
    else
    {
        // Sample directional light or envmap
        u.z = (u.z - point_directional_split)/(1-point_directional_split);

#ifdef IMPORTANCE_SAMPLE_ENVMAP
        if(u.z < envmap_directional_split)
        {
            vec3 color = sample_environment_map(rand_uint.xy, out_dir, out_length, pdf);
            pdf *= (1.0f-point_directional_split) * envmap_directional_split;
            return color;
        }
        else
#endif
        {
            const int light_count = int(scene_metadata.directional_light_count);
            int light_index = clamp(int(u.z*light_count), 0, light_count-1);

            directional_light dl = directional_lights.lights[light_index];
            out_length = RAY_MAX_DIST;
            vec3 color;
            sample_directional_light(dl, u.xy, out_dir, color, pdf);
            pdf *= ((1.0f - point_directional_split) * (1.0f - envmap_directional_split)) / light_count;
            return color;
        }
    }
}

void eval_explicit_lights(
    uvec3 rand_uint,
    mat3 tbn, vec3 shading_view, sampled_material mat,
    pt_vertex_data v,
    inout vec3 diffuse_radiance,
    inout vec3 specular_radiance
){
    vec3 out_dir;
    float out_length = 0.0f;
    float light_pdf;
    // Sample lights
    vec3 contrib = sample_explicit_light(rand_uint, v.pos, out_dir, out_length, light_pdf);

    vec3 shading_light = out_dir * tbn;
    vec3 d, s;
    float bsdf_pdf = ggx_bsdf_pdf(shading_light, shading_view, mat, d, s);
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
        bool terminal = !get_intersection_info(pos, view, v, nee_pdf, mat, light);

        attenuation /= bsdf_mis_pdf(nee_pdf, bsdf_pdf);

#ifdef HIDE_LIGHTS
        if(bounce == 0) light = vec3(0);
#endif

        vec3 diffuse_radiance = vec3(0);
        vec3 specular_radiance = mat.emission + light;

        if(bounce == 0)
        {
            first_hit_vertex = v;
            first_hit_material = mat;
        }

#ifdef PATH_SPACE_REGULARIZATION
        // Regularization strategy inspired by "Optimised Path Space Regularisation", 2021 Weier et al.
        float original_roughness = mat.roughness;
        mat.roughness = 1.0f - ((1.0f - mat.roughness) * regularization);
        regularization *= max(1 - control.regularization_gamma * original_roughness, 0.0f);
#endif

        mat3 tbn = create_tangent_space(v.mapped_normal);
        vec3 shading_view = -view * tbn;

        // A lot of the stuff below assumes that the view direction is on the same
        // side as the normal. If not, everything breaks. Which is why this check
        // exists. Normal maps can cause these degenerate cases.
        if(shading_view.z < 0.00001f)
            shading_view = vec3(shading_view.xy, max(shading_view.z, 0.00001f));

        shading_view = normalize(shading_view);

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
                generate_ray_sample_uint(lsampler, bounce).xyz, tbn, shading_view,
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
        vec4 ray_sample = generate_ray_sample(lsampler, bounce);
        ggx_bsdf_sample(ray_sample.xyz, shading_view, mat, view, diffuse_weight, specular_weight, bsdf_pdf);
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

