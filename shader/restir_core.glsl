#ifndef RESTIR_CORE_GLSL
#define RESTIR_CORE_GLSL

#ifndef TR_RESTIR
#define TR_RESTIR pc.config
#endif

#define SHADOW_MAPPING_PARAMS pc.config.sm_params

layout(binding = 0) uniform sampler2D depth_or_position_tex;
layout(binding = 1) uniform sampler2D normal_tex;
layout(binding = 2) uniform sampler2D flat_normal_tex;
layout(binding = 3) uniform sampler2D albedo_tex;
layout(binding = 4) uniform sampler2D curvature_tex;
layout(binding = 5) uniform sampler2D material_tex;


layout(binding = 6, rgba32ui) readonly uniform uimage2D in_reservoir_ris_data_tex;
layout(binding = 7, rgba32ui) readonly uniform uimage2D in_reservoir_reconnection_data_tex;
layout(binding = 8, rgba32f) readonly uniform image2D in_reservoir_reconnection_radiance_tex;
layout(binding = 9, rgba32ui) readonly uniform uimage2D in_reservoir_rng_seeds_tex;

layout(binding = 10, rgba32ui) uniform uimage2D out_reservoir_ris_data_tex;
layout(binding = 11, rgba32ui) uniform uimage2D out_reservoir_reconnection_data_tex;
layout(binding = 12, rgba32f) uniform image2D out_reservoir_reconnection_radiance_tex;
layout(binding = 13, rgba32ui) uniform uimage2D out_reservoir_rng_seeds_tex;

#define SH_INTERPOLATION_TRILINEAR
#include "alias_table.glsl"
#include "math.glsl"
#include "random_sampler.glsl"
#include "ggx.glsl"
#include "spherical_harmonics.glsl"
#include "ray_cone.glsl"

#ifdef RESTIR_TEMPORAL
layout(binding = 14) uniform sampler2D prev_depth_or_position_tex;
layout(binding = 15) uniform sampler2D prev_normal_tex;
layout(binding = 16) uniform sampler2D prev_flat_normal_tex;
layout(binding = 17) uniform sampler2D prev_albedo_tex;
layout(binding = 18) uniform sampler2D prev_curvature_tex;
layout(binding = 19) uniform sampler2D prev_material_tex;
layout(binding = 20) uniform sampler2D motion_tex;

#include "temporal_tables.glsl"
#endif

#include "gbuffer.glsl"
#include "projection.glsl"

struct domain
{
    sampled_material mat;
    vec3 pos;
    mat3 tbn;
    vec3 flat_normal;
    vec3 view;
    vec3 tview;
    ray_cone rc;
};

sampled_material get_material(ivec2 p)
{
    sampled_material mat = sample_gbuffer_material(
        albedo_tex, material_tex, p
    );
    return mat;
}

bool get_pos(ivec2 p, camera_data cam, out vec3 pos)
{
#ifdef USE_POSITION
    pos = sample_gbuffer_position(depth_or_position_tex, p);
    return any(isnan(pos));
#else
    ivec2 size = textureSize(depth_or_position_tex, 0);
    float depth = texelFetch(depth_or_position_tex, p, 0).r;
    float linear_depth = linearize_depth(depth * 2.0f - 1.0f, cam.projection_info);

    vec2 uv = (vec2(p)+0.5f)/vec2(size);
    pos = (cam.view_inverse * vec4(unproject_position(linear_depth, vec2(uv.x, 1-uv.y), cam.projection_info, cam.pan.xy), 1)).xyz;
    return depth == 1.0f;
#endif
}

bool read_domain(camera_data cam, ivec2 p, out domain d)
{
    vec3 origin = cam.origin.xyz;

    d.mat = get_material(p);
    d.tbn = create_tangent_space(sample_gbuffer_normal(normal_tex, p));
#ifdef USE_FLAT_NORMAL
    d.flat_normal = sample_gbuffer_normal(flat_normal_tex, p);
#else
    d.flat_normal = d.tbn[2];
#endif
    bool miss = get_pos(p, cam, d.pos);

    float len = length(d.pos - origin);
    d.view = (d.pos - origin) / len;
    d.tview = view_to_tangent_space(d.view, d.tbn);

    float curvature = sample_gbuffer_curvature(curvature_tex, p);
    d.rc = init_pixel_ray_cone(cam.projection_info, p, ivec2(TR_RESTIR.display_size));
    ray_cone_apply_dist(len, d.rc);
    ray_cone_apply_curvature(curvature, d.rc);

    return miss;
}

// Used to correct for inaccurate depth buffer in the first bounce.
void bias_ray_origin(inout vec3 pos, bool exit_below, in domain d)
{
#ifndef USE_POSITION
    float s = sign(dot(d.view, d.flat_normal));
    if(exit_below) s = -s;
    camera_data cam = camera.pairs[pc.camera_index].current;
    pos -= s * d.flat_normal * TR_RESTIR.min_ray_dist * distance(d.pos, cam.origin.xyz);
#endif
}

void bias_ray(inout vec3 pos, inout vec3 dir, inout float len, in domain d)
{
#ifndef USE_POSITION
    bool exit_below = dot(dir, d.flat_normal) < 0;
    vec3 target = pos + dir * len;
    bias_ray_origin(pos, exit_below, d);
    len = distance(target, pos);
    dir = (target - pos) / len;
#endif
}

#ifdef RESTIR_TEMPORAL
sampled_material get_prev_material(ivec2 p)
{
    sampled_material mat = sample_gbuffer_material(
        prev_albedo_tex, prev_material_tex, p
    );
    return mat;
}

bool get_prev_pos(ivec2 p, camera_data cam, out vec3 pos)
{
#ifdef USE_POSITION
    pos = sample_gbuffer_position(prev_depth_or_position_tex, p);
    return any(isnan(pos));
#else
    ivec2 size = textureSize(prev_depth_or_position_tex, 0);
    float depth = texelFetch(prev_depth_or_position_tex, p, 0).r;
    float linear_depth = linearize_depth(depth * 2.0f - 1.0f, cam.projection_info);

    vec2 uv = (vec2(p)+0.5f)/vec2(size);
    pos = (cam.view_inverse * vec4(unproject_position(linear_depth, vec2(uv.x, 1-uv.y), cam.projection_info, cam.pan.xy), 1)).xyz;
    return depth == 1.0f;
#endif
}

bool read_prev_domain(camera_data cam, ivec2 p, domain cur_domain, out domain d)
{
    vec3 origin = cam.origin.xyz;

#ifdef ASSUME_SAME_MATERIAL_IN_TEMPORAL
    d.mat = cur_domain.mat;
#else
    d.mat = get_prev_material(p);
#endif

    d.tbn = create_tangent_space(sample_gbuffer_normal(prev_normal_tex, p));
#ifdef USE_FLAT_NORMAL
    d.flat_normal = sample_gbuffer_normal(prev_flat_normal_tex, p);
#else
    d.flat_normal = d.tbn[2];
#endif
    bool miss = get_prev_pos(p, cam, d.pos);

    float len = length(d.pos - origin);
    d.view = (d.pos - origin) / len;
    d.tview = view_to_tangent_space(d.view, d.tbn);

    float curvature = sample_gbuffer_curvature(prev_curvature_tex, p);
    d.rc = init_pixel_ray_cone(cam.projection_info, p, ivec2(TR_RESTIR.display_size));
    ray_cone_apply_dist(len, d.rc);
    ray_cone_apply_curvature(curvature, d.rc);

    return miss;
}
#endif

reservoir unpack_reservoir(
    uvec4 ris_data,
    uvec4 reconnection_data,
    vec4 reconnection_radiance,
    uvec4 rng_seeds
){
    reservoir r;

    r.target_function_value = uintBitsToFloat(ris_data.r);
    r.ucw = uintBitsToFloat(ris_data.g);
    r.output_sample.base_path_jacobian_part = uintBitsToFloat(ris_data.b);

    uint confidence_path_length = ris_data[3];
    r.confidence = bitfieldExtract(confidence_path_length, 0, 15);
    r.output_sample.nee_terminal = bitfieldExtract(confidence_path_length, 15, 1) == 1 ? true : false;
    r.output_sample.head_lobe = bitfieldExtract(confidence_path_length, 16, 2);
    r.output_sample.tail_lobe = bitfieldExtract(confidence_path_length, 18, 2);
    r.output_sample.head_length = bitfieldExtract(confidence_path_length, 20, 6);
    r.output_sample.tail_length = bitfieldExtract(confidence_path_length, 26, 6);

    r.output_sample.vertex.hit_info.x = uintBitsToFloat(reconnection_data[0]);
    r.output_sample.vertex.hit_info.y = uintBitsToFloat(reconnection_data[1]);
    r.output_sample.vertex.instance_id = r.ucw <= 0 ? NULL_INSTANCE_ID : reconnection_data[2];
    r.output_sample.vertex.primitive_id = reconnection_data[3];

    r.output_sample.vertex.radiance_estimate = reconnection_radiance.rgb;
    r.output_sample.radiance_luminance = reconnection_radiance.a;

    vec2 incident_dir = vec2(
        uintBitsToFloat(rng_seeds[2]),
        uintBitsToFloat(rng_seeds[3])
    );
    r.output_sample.vertex.incident_direction = octahedral_unpack(incident_dir);

    r.output_sample.head_rng_seed = rng_seeds[0];
    r.output_sample.tail_rng_seed = rng_seeds[1];

    r.sum_weight = 0.0f;

    return r;
}

reservoir read_reservoir(ivec2 p, uvec2 size)
{
    return unpack_reservoir(
        imageLoad(in_reservoir_ris_data_tex, p),
        RESTIR_HAS_RECONNECTION_DATA ?
            imageLoad(in_reservoir_reconnection_data_tex, p) : uvec4(0),
        RESTIR_HAS_RECONNECTION_DATA ?
            imageLoad(in_reservoir_reconnection_radiance_tex, p) : uvec4(0),
        RESTIR_HAS_SEEDS ? imageLoad(in_reservoir_rng_seeds_tex, p) : uvec4(0)
    );
}

float read_confidence(ivec2 p, uvec2 size)
{
    uint ris = imageLoad(in_reservoir_ris_data_tex, p).w;
    return bitfieldExtract(ris, 0, 15);
}

reservoir read_out_reservoir(ivec2 p, uvec2 size)
{
    return unpack_reservoir(
        imageLoad(out_reservoir_ris_data_tex, p),
        RESTIR_HAS_RECONNECTION_DATA ?
            imageLoad(out_reservoir_reconnection_data_tex, p) : uvec4(0),
        RESTIR_HAS_RECONNECTION_DATA ?
            imageLoad(out_reservoir_reconnection_radiance_tex, p) : uvec4(0),
        RESTIR_HAS_SEEDS ? imageLoad(out_reservoir_rng_seeds_tex, p) : uvec4(0)
    );
}

void write_reservoir(reservoir r, ivec2 p, uvec2 size)
{
    r.confidence = min(r.confidence, TR_RESTIR.max_confidence);
    uint reconnection_info = 0;
    reconnection_info = bitfieldInsert(reconnection_info, int(r.confidence), 0, 15);
    reconnection_info = bitfieldInsert(reconnection_info, r.output_sample.nee_terminal ? 1 : 0, 15, 1);
    reconnection_info = bitfieldInsert(reconnection_info, r.output_sample.head_lobe, 16, 2);
    reconnection_info = bitfieldInsert(reconnection_info, r.output_sample.tail_lobe, 18, 2);
    reconnection_info = bitfieldInsert(reconnection_info, r.output_sample.head_length, 20, 6);
    reconnection_info = bitfieldInsert(reconnection_info, r.output_sample.tail_length, 26, 6);
    if(isnan(r.ucw) || isinf(r.ucw) || r.ucw < 0)
        r.ucw = 0;

    uvec4 ris_data = uvec4(
        floatBitsToUint(r.target_function_value),
        floatBitsToUint(r.ucw),
        floatBitsToUint(r.output_sample.base_path_jacobian_part),
        reconnection_info
    );
    imageStore(out_reservoir_ris_data_tex, p, ris_data);

    if(RESTIR_HAS_RECONNECTION_DATA)
    {
        uvec4 reconnection_data = uvec4(
            floatBitsToUint(r.output_sample.vertex.hit_info.x),
            floatBitsToUint(r.output_sample.vertex.hit_info.y),
            r.output_sample.vertex.instance_id,
            r.output_sample.vertex.primitive_id
        );
        imageStore(out_reservoir_reconnection_data_tex, p, reconnection_data);
        imageStore(out_reservoir_reconnection_radiance_tex, p, vec4(
            r.output_sample.vertex.radiance_estimate, r.output_sample.radiance_luminance
        ));
    }

    if(RESTIR_HAS_SEEDS)
    {
        uvec2 rng_seeds = uvec2(
            r.output_sample.head_rng_seed,
            r.output_sample.tail_rng_seed
        );
        vec2 incident_dir = octahedral_pack(r.output_sample.vertex.incident_direction);
        imageStore(out_reservoir_rng_seeds_tex, p, uvec4(rng_seeds,
            floatBitsToUint(incident_dir.x),
            floatBitsToUint(incident_dir.y)
        ));
    }
}

int self_shadow(vec3 view, vec3 flat_normal, vec3 light_dir, sampled_material mat)
{
    return dot(flat_normal, light_dir) * dot(view, flat_normal) < 0 && mat.transmittance == 0.0f ? 0 : 1;
}

void update_regularization(float bsdf_pdf, inout float regularization)
{
#ifdef PATH_SPACE_REGULARIZATION
    // Regularization strategy inspired by "Optimised Path Space Regularisation", 2021 Weier et al.
    // I'm using the BSDF PDF instead of roughness, which seems to be more
    // effective at reducing fireflies.
    if(bsdf_pdf != 0.0f)
        regularization *= max(1 - PATH_SPACE_REGULARIZATION / pow(bsdf_pdf, 0.25f), 0.0f);
#endif
}

void apply_regularization(float regularization, inout sampled_material mat)
{
#ifdef PATH_SPACE_REGULARIZATION
    mat.roughness = 1.0f - ((1.0f - mat.roughness) * regularization);
#endif
}

vec3 get_bounce_throughput(
    uint bounce_index,
    domain src,
    vec3 dir,
    bsdf_lobes lobes,
    inout vec4 primary_bsdf
){
    vec3 bsdf = modulate_bsdf(src.mat, lobes);
#ifdef DEMODULATE_OUTPUT
    if(bounce_index == 0)
    {
        float transmission = lobes.transmission + lobes.diffuse;
        float reflection = lobes.dielectric_reflection + lobes.metallic_reflection;
        float scale = transmission + reflection;
        float inv_scale = scale == 0 ? 0 : 1.0f / scale;

        primary_bsdf.rgb = vec3(scale);
        primary_bsdf.a = scale <= 0 ? 0 : reflection * inv_scale;
        bsdf = vec3(1.0f);
    }
#endif
    return bsdf * self_shadow(-src.view, src.flat_normal, dir, src.mat);
}

struct sum_contribution
{
    vec3 diffuse;
#ifdef DEMODULATE_OUTPUT
    vec3 reflection;
#endif
};

sum_contribution init_sum_contribution(vec3 emission)
{
    sum_contribution sc;
#ifdef DEMODULATE_OUTPUT
    sc.diffuse = vec3(0);
    sc.reflection = vec3(0);
#else
    sc.diffuse = emission;
#endif
    return sc;
}

void add_contribution(inout sum_contribution sc, vec4 contrib, float weight)
{
#ifdef DEMODULATE_OUTPUT
    vec3 color = contrib.rgb * weight;
    sc.diffuse += color * (1.0f - contrib.a);
    sc.reflection += color * contrib.a;
#else
    sc.diffuse += contrib.rgb * weight;
#endif
}

#ifdef RAY_TRACING
#include "rt_common.glsl"

#ifdef SHADE_ALL_EXPLICIT_LIGHTS
#define SHADOW_MAPPING_SCREEN_COORD vec2(gl_GlobalInvocationID.xy)
#include "shadow_mapping.glsl"
vec3 shade_explicit_lights(
    int instance_id,
    vec3 view,
    vertex_data vd,
    sampled_material mat,
    uint bounce_index
){
    mat3 tbn = create_tangent_space(vd.mapped_normal);
    vec3 tview = view_to_tangent_space(view, tbn);

    vec3 contrib = vec3(0);
#ifdef SHADE_FAKE_INDIRECT
    // Only last bounce gets fake indirect light.
    if(bounce_index == MAX_BOUNCES-1)
    {
        vec3 ref_dir = reflect(view, vd.mapped_normal);
        vec3 incoming_diffuse = vec3(0);
        vec3 incoming_reflection = vec3(0);

        int sh_grid_index = instances.o[instance_id].sh_grid_index;
        if(sh_grid_index >= 0)
        {
            sh_grid sg = sh_grids.grids[sh_grid_index];
            vec3 sample_pos = (sg.pos_from_world * vec4(vd.pos, 1)).xyz;
            vec3 sh_normal = normalize((mat3(sg.normal_from_world) * vd.mapped_normal).xyz);
            vec3 sh_ref = normalize((mat3(sg.normal_from_world) * ref_dir).xyz);

            sample_and_filter_sh_grid(
                sh_grid_data[nonuniformEXT(sh_grid_index)], sg.grid_clamp,
                sample_pos, sh_normal, sh_ref, sqrt(mat.roughness),
                incoming_diffuse, incoming_reflection
            );
        }
        float cos_v = max(dot(vd.mapped_normal, -view), 0.0f);

        // The fresnel value must be attenuated, because we are actually integrating
        // over all directions instead of just one specific direction here. This is
        // an approximated function, though.
        float fresnel = fresnel_schlick_attenuated(cos_v, mat.f0, mat.roughness);
        float kd = (1.0f - fresnel) * (1.0f - mat.metallic) * (1.0f - mat.transmittance);
        contrib += kd * incoming_diffuse * mat.albedo.rgb;

        vec2 bi = texture(brdf_integration, vec2(cos_v, sqrt(mat.roughness))).xy;
        contrib += incoming_reflection * mix(vec3(mat.f0 * bi.x + bi.y), mat.albedo.rgb, mat.metallic);
    }
#endif

    for(uint i = 0; i < scene_metadata.point_light_count; ++i)
    {
        point_light pl = point_lights.lights[i];
        vec3 light_dir;
        float light_dist;
        vec3 light_color;
        get_point_light_info(pl, vd.pos, light_dir, light_dist, light_color);
        bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
        ggx_brdf(light_dir * tbn, tview, mat, lobes);
        float shadow = 1.0f;
        if(pl.shadow_map_index >= 0 && dot(vd.mapped_normal, light_dir) > 0)
            shadow = calc_point_shadow(
                pl.shadow_map_index, vd.pos, vd.hard_normal, light_dir
            );
        if(dot(vd.hard_normal, light_dir) < 0)
            shadow = 0.0f;

        contrib += light_color * shadow * modulate_bsdf(mat, lobes);
    }

    for(uint i = 0; i < scene_metadata.directional_light_count; ++i)
    {
        directional_light dl = directional_lights.lights[i];
        bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
        ggx_brdf(-dl.dir * tbn, tview, mat, lobes);

        float shadow = 1.0f;
        if(dl.shadow_map_index >= 0 && dot(vd.mapped_normal, -dl.dir) > 0)
            shadow = calc_directional_shadow(
                dl.shadow_map_index, vd.pos, vd.hard_normal, -dl.dir
            );
        if(dot(vd.hard_normal, dl.dir) > 0)
            shadow = 0.0f;

        contrib += dl.color * shadow * modulate_bsdf(mat, lobes);
    }
    return contrib;
}
#endif

float test_visibility(uint seed, vec3 pos, vec3 dir, float dist, vec3 flat_normal)
{
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq,
        tlas,
#ifdef STOCHASTIC_ALPHA_BLENDING
        gl_RayFlagsTerminateOnFirstHitEXT,
#else
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
#endif
        VISIBILITY_RAY_MASK,
        pos,
        TR_RESTIR.min_ray_dist,
        dir,
        dist-TR_RESTIR.min_ray_dist*2
    );

    return trace_ray_query_visibility(rq);
}

#ifdef RESTIR_TEMPORAL
float test_prev_visibility(uint seed, vec3 pos, vec3 dir, float dist, vec3 flat_normal)
{
    rayQueryEXT rq;

#ifdef ASSUME_UNCHANGED_ACCELERATION_STRUCTURES
    rayQueryInitializeEXT(rq,
        tlas,
#ifdef STOCHASTIC_ALPHA_BLENDING
        gl_RayFlagsTerminateOnFirstHitEXT,
#else
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
#endif
        VISIBILITY_RAY_MASK,
        pos,
        TR_RESTIR.min_ray_dist,
        dir,
        dist-TR_RESTIR.min_ray_dist*2
    );
#else
    rayQueryInitializeEXT(rq,
        prev_tlas,
#ifdef STOCHASTIC_ALPHA_BLENDING
        gl_RayFlagsTerminateOnFirstHitEXT,
#else
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
#endif
        VISIBILITY_RAY_MASK,
        pos,
        TR_RESTIR.min_ray_dist,
        dir,
        dist-TR_RESTIR.min_ray_dist*2
    );
#endif

    return trace_ray_query_visibility_prev(rq);
}
#endif

struct resolved_vertex
{
    vec3 dir;
    float dist;
    vec3 normal; // 0 for zero-radius point light sources
    // If the vertex is emissive, this holds the emission. If not, treat the value as undefined.
    vec3 emission;

    bsdf_lobes lobes;
    float bsdf_pdf;
};

bool resolve_reconnection_vertex(
    inout restir_sample rs,
    bool rv_is_in_past,
    domain to_domain,
    inout float regularization,
    out resolved_vertex to
){
    uint bounce_index = rs.head_length;

    // if rv_is_in_past, to is cur. Otherwise, to is prev.
    if(rs.vertex.instance_id == NULL_INSTANCE_ID)
        return false;

    if(rs.vertex.instance_id == POINT_LIGHT_INSTANCE_ID)
    {
#ifdef RESTIR_TEMPORAL
        uint primitive_id;
        if(rv_is_in_past) primitive_id = point_light_forward_map.array[rs.vertex.primitive_id];
        else primitive_id = point_light_backward_map.array[rs.vertex.primitive_id];

        if(primitive_id >= scene_metadata.point_light_count)
            return false;

        const bool prev = !rv_is_in_past;
#else
        const bool prev = false;
#endif
#ifdef RESTIR_TEMPORAL
        point_light pl;
        if(prev) pl = prev_point_lights.lights[rs.vertex.primitive_id];
        else pl = point_lights.lights[rs.vertex.primitive_id];
#else
        point_light pl = point_lights.lights[rs.vertex.primitive_id];
#endif
        vec3 pos = pl.pos;

        vec3 center_dir = pos - to_domain.pos;
        float dist2 = dot(center_dir, center_dir);

        float radius = pl.radius;

        if(radius != 0)
        {
            to.normal = octahedral_unpack(rs.vertex.hit_info * 2.0f - 1.0f);
            pos.xyz += radius * to.normal;
        }
        else to.normal = vec3(0);

        to.emission = pl.color;
        if(radius != 0.0f) to.emission /= M_PI * radius * radius;
        to.emission *= get_spotlight_intensity(pl, normalize(center_dir));
        if(radius == 0) to.emission /= dist2;

        to.dir = normalize(pos - to_domain.pos);
        to.dist = length(pos - to_domain.pos);
    }
    else if(
        rs.vertex.instance_id == ENVMAP_INSTANCE_ID
#ifdef SHADE_ALL_EXPLICIT_LIGHTS
        || rs.vertex.instance_id == MISS_INSTANCE_ID
#endif
    )
    {
        to.dir = octahedral_unpack(rs.vertex.hit_info * 2.0f - 1.0f);
        to.dist = TR_RESTIR.max_ray_dist;
        to.normal = vec3(0);
        to.emission = rs.vertex.radiance_estimate;
    }
    else if(rs.vertex.instance_id == DIRECTIONAL_LIGHT_INSTANCE_ID)
    {
        to.dir = octahedral_unpack(rs.vertex.hit_info * 2.0f - 1.0f);
        to.dist = TR_RESTIR.max_ray_dist;
        to.normal = vec3(0);

#if defined(RESTIR_TEMPORAL) && !defined(ASSUME_UNCHANGED_RECONNECTION_RADIANCE)
        to.emission = vec3(0);
        for(uint i = 0; i < scene_metadata.directional_light_count; ++i)
        {
            directional_light dl = directional_lights.lights[i];
            float visible = step(dl.dir_cutoff, dot(to.dir, -dl.dir));
            to.emission += visible * dl.color / (2.0f * M_PI * (1.0f - dl.dir_cutoff));
        }
#else
        to.emission = rs.vertex.radiance_estimate;
#endif
    }
#ifndef SHADE_ALL_EXPLICIT_LIGHTS
    else if(rs.vertex.instance_id == MISS_INSTANCE_ID)
    {
        to.dir = octahedral_unpack(rs.vertex.hit_info * 2.0f - 1.0f);
        to.dist = TR_RESTIR.max_ray_dist;
        to.normal = vec3(0);

#if defined(RESTIR_TEMPORAL) && !defined(ASSUME_UNCHANGED_RECONNECTION_RADIANCE)
        to.emission = scene_metadata.environment_factor.rgb;
        if(scene_metadata.environment_proj >= 0)
        {
            vec2 uv = vec2(0);
            uv.y = asin(-to.dir.y)/M_PI+0.5f;
            uv.x = atan(to.dir.z, to.dir.x)/(2*M_PI)+0.5f;
            to.emission *= texture(environment_map_tex, uv).rgb;
        }

        for(uint i = 0; i < scene_metadata.directional_light_count; ++i)
        {
            directional_light dl = directional_lights.lights[i];
            float visible = step(dl.dir_cutoff, dot(to.dir, -dl.dir));
            to.emission += visible * dl.color / (2.0f * M_PI * (1.0f - dl.dir_cutoff));
        }
#else
        to.emission = rs.vertex.radiance_estimate;
#endif
    }
#endif
    else
    { // Regular mesh triangle
#ifdef RESTIR_TEMPORAL
        if(rv_is_in_past)
            rs.vertex.instance_id = instance_forward_map.array[rs.vertex.instance_id];
        if(rs.vertex.instance_id >= scene_metadata.instance_count)
            return false;
#endif
        float pdf;
        vertex_data vd = get_interpolated_vertex(
            vec3(0),
            rs.vertex.hit_info,
            int(rs.vertex.instance_id),
            int(rs.vertex.primitive_id)
#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
            , to_domain.pos, pdf
#endif
        );
        vec3 cur_pos = vd.pos;
#ifdef RESTIR_TEMPORAL
        if(!rv_is_in_past)
            vd.pos = vd.prev_pos;
#endif

        to.dir = normalize(vd.pos-to_domain.pos);
        to.dist = length(vd.pos-to_domain.pos);
        to.normal = vd.hard_normal;

#if !defined(SHADE_ALL_EXPLICIT_LIGHTS) || defined(ASSUME_UNCHANGED_RECONNECTION_RADIANCE)
        // Normal operation: just find the emissiveness of the vertex.
        // TODO: Allow mesh triangle material changes?
        to.emission = rs.vertex.radiance_estimate;
#else
        vd.back_facing = dot(vd.hard_normal, to.dir) > 0;
        if(vd.back_facing)
        {
            vd.smooth_normal = -vd.smooth_normal;
            vd.hard_normal = -vd.hard_normal;
        }
        vd.mapped_normal = vd.smooth_normal;

        to.lobes = bsdf_lobes(0,0,0,0);
        vec3 tdir = to.dir * to_domain.tbn;
        to.bsdf_pdf = ggx_bsdf_lobe_pdf(rs.head_lobe, tdir, to_domain.tview, to_domain.mat, to.lobes);
        update_regularization(to.bsdf_pdf, regularization);

        ray_cone rc = to_domain.rc;
        ray_cone_apply_dist(to.dist, rc);
        vec2 puvdx;
        vec2 puvdy;
        ray_cone_gradients(
            rc,
            to.dir,
            vd.hard_normal,
            cur_pos,
            vd.uv.xy,
            vd.triangle_pos,
            vd.triangle_uv,
            puvdx, puvdy
        );

        sampled_material mat = sample_material(int(rs.vertex.instance_id), vd, puvdx, puvdy);
        apply_regularization(regularization, mat);
        to.emission = mat.emission + shade_explicit_lights(int(rs.vertex.instance_id), to.dir, vd, mat, bounce_index);
        return true;
#endif
    }

    to.lobes = bsdf_lobes(0,0,0,0);
    vec3 tdir = to.dir * to_domain.tbn;
    to.bsdf_pdf = ggx_bsdf_lobe_pdf(rs.head_lobe, tdir, to_domain.tview, to_domain.mat, to.lobes);
    update_regularization(to.bsdf_pdf, regularization);
    return true;
}
#endif

bool allow_initial_reconnection(sampled_material mat)
{
#ifdef USE_RECONNECTION_SHIFT
    return true;
#elif defined(USE_RANDOM_REPLAY_SHIFT)
    return false;
#elif defined(USE_HYBRID_SHIFT)
    // TODO: Better reconnection heuristics, but I'm not sure what's legal to
    // take into account...
    return mat.roughness > 0.05f;
#endif
}

bool allow_reconnection(
    float dist,
    sampled_material mat,
    bool bounces,
    inout bool as_head
){
#ifdef USE_RECONNECTION_SHIFT
    as_head = true;
    return true;
#elif defined(USE_RANDOM_REPLAY_SHIFT)
    as_head = false;
    return false;
#elif defined(USE_HYBRID_SHIFT)
    bool head = as_head;
    as_head = bounces ? mat.roughness > 0.05f : true;
    // Don't allow reconnection to transient objects.
    if(bounces && (mat.flags & MATERIAL_FLAG_TRANSIENT) != 0) as_head = false;
    return head && as_head && (dist > TR_RESTIR.reconnection_scale || !bounces);
#endif
}

#ifdef RAY_TRACING
bool get_intersection_info(
    vec3 ray_origin,
    vec3 ray_direction,
    ray_cone rc,
    hit_info payload,
    uint bounce_index,
    bool in_past,
    float regularization,
    out intersection_info info,
    out reconnection_vertex candidate
){
#if defined(RESTIR_TEMPORAL) && defined(ASSUME_UNCHANGED_ACCELERATION_STRUCTURES)
    in_past = false;
#endif

    info.light = vec3(0);
    info.envmap_pdf = 0.0f;
    info.local_pdf = 0.0f;

    if(payload.instance_id >= 0)
    { // Mesh
#ifdef RESTIR_TEMPORAL
        if(in_past)
        {
            // Translate previous mesh to current frame
            uint new_id = instance_forward_map.array[payload.instance_id];
            if(new_id == 0xFFFFFFFFu)
            {
                candidate.instance_id = NULL_INSTANCE_ID;
                candidate.primitive_id = 0;
                candidate.hit_info = vec2(0);
                candidate.radiance_estimate = vec3(0);
                return false;
            }
            payload.instance_id = int(new_id);
        }
#endif

        float pdf = 0.0f;
        info.vd = get_interpolated_vertex(
            ray_direction, payload.barycentrics,
            payload.instance_id,
            payload.primitive_id
#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
            , ray_origin, pdf
#endif
        );
        vec3 cur_pos = info.vd.pos;
#ifdef RESTIR_TEMPORAL
        if(in_past) info.vd.pos = info.vd.prev_pos;
#endif

        ray_cone_apply_dist(distance(ray_origin, info.vd.pos), rc);
        vec2 puvdx;
        vec2 puvdy;
        ray_cone_gradients(
            rc,
            ray_direction,
            info.vd.hard_normal,
            cur_pos,
            info.vd.uv.xy,
            info.vd.triangle_pos,
            info.vd.triangle_uv,
            puvdx, puvdy
        );
        info.mat = sample_material(payload.instance_id, info.vd, puvdx, puvdy);

        apply_regularization(regularization, info.mat);

        info.local_pdf = any(greaterThan(info.mat.emission, vec3(0))) ? pdf : 0;

        candidate.instance_id = payload.instance_id;
        candidate.primitive_id = payload.primitive_id;
        candidate.hit_info = payload.barycentrics;
        candidate.radiance_estimate = info.mat.emission;

#ifdef SHADE_ALL_EXPLICIT_LIGHTS
        candidate.radiance_estimate += shade_explicit_lights(
            payload.instance_id, ray_direction, info.vd, info.mat, bounce_index
        );
#endif

        return true;
    }
#ifndef SHADE_ALL_EXPLICIT_LIGHTS
    else if(payload.primitive_id >= 0)
    { // Point light
        // Point lights are unhittable with SHADE_ALL_EXPLICIT_LIGHTS, as ray
        // flags cull them.
        point_light pl;
#ifdef RESTIR_TEMPORAL
        if(in_past)
        {
            pl = prev_point_lights.lights[payload.primitive_id];
            uint new_id = point_light_forward_map.array[payload.primitive_id];
            if(new_id == 0xFFFFFFFFu)
            {
                candidate.instance_id = NULL_INSTANCE_ID;
                candidate.primitive_id = 0;
                candidate.hit_info = vec2(0);
                candidate.radiance_estimate = vec3(0);
                return false;
            }
            payload.primitive_id = int(new_id);
        }
        else
#endif
        pl = point_lights.lights[payload.primitive_id];

        vec3 hit_pos = ray_origin + ray_direction * payload.barycentrics.x;
        // No worries, radius cannot be zero -- we couldn't hit the light here
        // if it was!
        info.light = get_spotlight_intensity(pl, normalize(pl.pos-ray_origin)) * pl.color / (pl.radius * pl.radius * M_PI);
        info.local_pdf = sample_point_light_pdf(pl, ray_origin);
        info.vd.hard_normal = normalize(hit_pos-pl.pos);
        info.vd.pos = hit_pos;

        candidate.instance_id = POINT_LIGHT_INSTANCE_ID;
        candidate.primitive_id = payload.primitive_id;
        candidate.hit_info = octahedral_pack(info.vd.hard_normal) * 0.5f + 0.5f;
        candidate.radiance_estimate = info.light;
        return false;
    }
#endif
    else
    { // Miss
        vec4 color = scene_metadata.environment_factor;
        if(scene_metadata.environment_proj >= 0)
        {
            vec2 uv = vec2(0);
            uv.y = asin(-ray_direction.y)/M_PI+0.5f;
            uv.x = atan(ray_direction.z, ray_direction.x)/(2*M_PI)+0.5f;
            color.rgb *= texture(environment_map_tex, uv).rgb;
        }
        info.light += color.rgb;

        info.envmap_pdf = scene_metadata.environment_proj >= 0 ? sample_environment_map_pdf(ray_direction) : 0.0f;
        info.local_pdf = 0;
#ifndef SHADE_ALL_EXPLICIT_LIGHTS
        for(uint i = 0; i < scene_metadata.directional_light_count; ++i)
        {
            directional_light dl = directional_lights.lights[i];
            if(dl.dir_cutoff >= 1.0f)
                continue;
            float visible = step(dl.dir_cutoff, dot(ray_direction, -dl.dir));
            vec3 color = visible * dl.color / (2.0f * M_PI * (1.0f - dl.dir_cutoff));
            info.light += color;
            info.local_pdf += visible * sample_directional_light_pdf(dl);
        }
#endif

        candidate.instance_id = MISS_INSTANCE_ID;
        candidate.primitive_id = 0;
        candidate.hit_info = octahedral_pack(ray_direction) * 0.5f + 0.5f;
        candidate.radiance_estimate = info.light;
        info.vd.pos = ray_origin + TR_RESTIR.max_ray_dist * ray_direction;
        info.vd.hard_normal = vec3(0);
        return false;
    }
}

hit_info trace_ray(
    uint seed,
    vec3 origin,
    vec3 dir
){
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq,
        tlas,
#ifdef STOCHASTIC_ALPHA_BLENDING
        gl_RayFlagsNoneEXT,
#else
        gl_RayFlagsOpaqueEXT,
#endif
        RAY_MASK,
        origin,
        TR_RESTIR.min_ray_dist,
        dir,
        TR_RESTIR.max_ray_dist
    );

    return trace_ray_query(rq, seed);
}

#ifdef RESTIR_TEMPORAL
hit_info trace_prev_ray(
    uint seed,
    vec3 origin,
    vec3 dir
){
    rayQueryEXT rq;

#ifdef ASSUME_UNCHANGED_ACCELERATION_STRUCTURES
    rayQueryInitializeEXT(rq,
        tlas,
#ifdef STOCHASTIC_ALPHA_BLENDING
        gl_RayFlagsNoneEXT,
#else
        gl_RayFlagsOpaqueEXT,
#endif
        RAY_MASK,
        origin,
        TR_RESTIR.min_ray_dist,
        dir,
        TR_RESTIR.max_ray_dist
    );
    return trace_ray_query(rq, seed);
#else
    rayQueryInitializeEXT(rq,
        prev_tlas,
#ifdef STOCHASTIC_ALPHA_BLENDING
        gl_RayFlagsNoneEXT,
#else
        gl_RayFlagsOpaqueEXT,
#endif
        RAY_MASK,
        origin,
        TR_RESTIR.min_ray_dist,
        dir,
        TR_RESTIR.max_ray_dist
    );
    return trace_ray_query_prev(rq, seed);
#endif
}
#endif

struct light_sample
{
    bool infinitesimal;
    vec3 color;
    vec3 dir;
    float dist;
    float pdf;
    uint instance_id;
    uint primitive_id;
    vec2 hit_info;
    vec3 normal;
};

// Warning: does NOT update the seed! You need to do that yourself.
light_sample sample_light(
    uvec4 rand32,
    ray_cone rc,
    vec3 pos,
    float min_dist,
    float max_dist
){
    float point_prob, triangle_prob, dir_prob, envmap_prob;
    get_nee_sampling_probabilities(point_prob, triangle_prob, dir_prob, envmap_prob);

    light_sample ls;
    ls.color = vec3(0);

    vec4 u = vec4(rand32) * INV_UINT32_MAX;

    ls.instance_id = NULL_INSTANCE_ID;
    ls.normal = vec3(0);
    ls.dist = max_dist;

    float local_pdf = 0.0f;
    if((u.x -= point_prob) < 0)
    { // Point light
        ls.instance_id = POINT_LIGHT_INSTANCE_ID;
        float weight = 0;
        random_sample_point_light(pos, u.y, int(scene_metadata.point_light_count), weight, ls.primitive_id);

        ls.pdf = point_prob / weight;

        point_light pl = point_lights.lights[ls.primitive_id];
        sample_point_light(pl, u.zw, pos, ls.dir, ls.dist, ls.color, local_pdf);

        vec3 p = pos + ls.dir * ls.dist;
        ls.normal = normalize(p - pl.pos);
        ls.hit_info = octahedral_pack(ls.normal) * 0.5f + 0.5f;
        if(local_pdf <= 0.0f) ls.normal = vec3(0);
    }
    else if((u.x -= triangle_prob) < 0)
    { // triangle light
        int selected_index = clamp(int(u.y*scene_metadata.tri_light_count), 0, int(scene_metadata.tri_light_count-1));

        ls.pdf = triangle_prob / max(scene_metadata.tri_light_count, 1);

        tri_light tl = tri_lights.lights[selected_index];

        ls.instance_id = tl.instance_id;
        ls.primitive_id = tl.primitive_id;

        vec3 A = tl.pos[0]-pos;
        vec3 B = tl.pos[1]-pos;
        vec3 C = tl.pos[2]-pos;

        ls.dir = sample_triangle_light(u.zw, A, B, C, local_pdf);
        ls.dist = ray_plane_intersection_dist(ls.dir, A, B, C);
        vec3 bary = get_barycentric_coords(ls.dir*ls.dist, A, B, C);
        ls.hit_info = bary.yz;
        ls.dist -= min_dist;

        ls.normal = normalize(cross(
            tl.pos[2] - tl.pos[0],
            tl.pos[1] - tl.pos[0]
        ));

        ls.color = r9g9b9e5_to_rgb(tl.emission_factor);
        if(tl.emission_tex_id >= 0)
        { // Textured emissive triangle, so read texture.
            vec2 uvs[3] = {
                unpackHalf2x16(tl.uv[0]),
                unpackHalf2x16(tl.uv[1]),
                unpackHalf2x16(tl.uv[2])
            };
            vec2 uv = bary.x * uvs[0] + bary.y * uvs[1] + bary.z * uvs[2];
            ray_cone_apply_dist(ls.dist+min_dist, rc);
            vec2 puvdx;
            vec2 puvdy;
            ray_cone_gradients(
                rc,
                ls.dir,
                ls.normal,
                pos + ls.dir * (ls.dist + min_dist),
                uv,
                tl.pos,
                uvs,
                puvdx,
                puvdy
            );
            ls.color *= textureGrad(textures[nonuniformEXT(tl.emission_tex_id)], uv, puvdx, puvdy).rgb;
        }

        // TODO: Check this condition if ReSTIR is doing NaNs with triangle
        // lights! May still need the flat normal check or something else
        // that is equivalent! ls.normal?
        if(
            isinf(local_pdf) || local_pdf <= 0 || any(isnan(ls.dir)) ||
            ls.dist < min_dist // || abs(dot(ls.dir, d.flat_normal)) < 1e-4f
        ){
            ls.instance_id = NULL_INSTANCE_ID;
        }
    }
    else if((u.x -= dir_prob) < 0)
    { // Directional light
        int selected_index = clamp(
            int(u.y * scene_metadata.directional_light_count),
            0, int(scene_metadata.directional_light_count)-1
        );
        directional_light dl = directional_lights.lights[selected_index];

        sample_directional_light(dl, u.zw, ls.dir, ls.color, local_pdf);
        ls.instance_id = DIRECTIONAL_LIGHT_INSTANCE_ID;
        ls.primitive_id = selected_index;
        ls.hit_info = octahedral_pack(ls.dir) * 0.5f + 0.5f;

        ls.pdf = dir_prob / scene_metadata.directional_light_count;
    }
    else if((u.x -= envmap_prob) < 0)
    { // Envmap
        rand32 += uvec4(12); // Make 'values' not correlate with future RNG samples
        pcg4d(rand32);
        ls.color = sample_environment_map(rand32.xyz, ls.dir, ls.dist, local_pdf);

        ls.instance_id = ENVMAP_INSTANCE_ID;
        ls.primitive_id = 0;
        ls.hit_info = octahedral_pack(ls.dir) * 0.5f + 0.5f;

        ls.pdf = envmap_prob;
    }

    ls.infinitesimal = local_pdf <= 0;
    if(!ls.infinitesimal) ls.pdf *= local_pdf;

    return ls;
}

float calculate_light_pdf(
    uint instance_id,
    uint primitive_id,
    float local_pdf,
    float envmap_pdf
){
    float point_prob, triangle_prob, dir_prob, envmap_prob;
    get_nee_sampling_probabilities(point_prob, triangle_prob, dir_prob, envmap_prob);

    if(instance_id == MISS_INSTANCE_ID)
    {
        return local_pdf * dir_prob / max(scene_metadata.directional_light_count, 1u) + envmap_pdf * envmap_prob;
    }
    else if(instance_id == NULL_INSTANCE_ID || local_pdf == 0)
        return 0;
    else if(instance_id == POINT_LIGHT_INSTANCE_ID)
    {
        return local_pdf * point_prob / max(scene_metadata.point_light_count, 1u);
    }
    else
    { // Tri light
        return local_pdf * triangle_prob / max(scene_metadata.tri_light_count, 1u);
    }
    return 0;
}

bool generate_nee_vertex(
    uvec4 rand32,
    domain d,
    out reconnection_vertex vertex,
    out vec3 vertex_normal,
    out vec3 vertex_dir,
    out float vertex_dist,
    out float nee_pdf
){
    light_sample ls = sample_light(
        rand32,
        d.rc,
        d.pos,
        TR_RESTIR.min_ray_dist,
        TR_RESTIR.max_ray_dist
    );

    if(ls.instance_id == POINT_LIGHT_INSTANCE_ID && ls.infinitesimal)
        ls.color /= ls.dist * ls.dist;

    vertex.instance_id = ls.instance_id;
    vertex.primitive_id = ls.primitive_id;
    vertex.hit_info = ls.hit_info;
    vertex.radiance_estimate = ls.color;
    // Incident direction is not reported from here, it's filled by the caller
    // afterwards.
    vertex.incident_direction = vec3(0);
    vertex_normal = ls.normal;
    vertex_dir = ls.dir;
    vertex_dist = ls.dist;
    nee_pdf = ls.pdf;

    return ls.infinitesimal;
}

bool generate_bsdf_vertex(
    uvec4 rand32,
    vec3 dir,
    bool in_past,
    domain cur_domain,
    uint bounce_index,
    float regularization,
    out domain next_domain, // Only valid if true is returned.
    out reconnection_vertex vertex,
    out float nee_pdf
){
    intersection_info info;
    hit_info hi;
#ifdef RESTIR_TEMPORAL
    if(in_past) hi = trace_prev_ray(rand32.w, cur_domain.pos, dir);
    else
#endif
        hi = trace_ray(rand32.w, cur_domain.pos, dir);

#if !defined(ASSUME_UNCHANGED_ACCELERATION_STRUCTURES) && defined(RESTIR_TEMPORAL)
    const bool get_in_past = in_past;
#else
    const bool get_in_past = false;
#endif
    bool bounces = get_intersection_info(cur_domain.pos, dir, cur_domain.rc, hi, bounce_index, get_in_past, regularization, info, vertex);

#if defined(NEE_SAMPLE_POINT_LIGHTS) || defined(NEE_SAMPLE_EMISSIVE_TRIANGLES) || defined(NEE_SAMPLE_DIRECTIONAL_LIGHTS) || defined(NEE_SAMPLE_ENVMAP)
    nee_pdf = calculate_light_pdf(
        vertex.instance_id, vertex.primitive_id, info.local_pdf, info.envmap_pdf
    );
#else
    nee_pdf = 0;
#endif

    next_domain.mat = info.mat;
    next_domain.mat.emission = vertex.radiance_estimate;
    next_domain.pos = info.vd.pos;
    next_domain.rc = cur_domain.rc;
    ray_cone_apply_dist(distance(cur_domain.pos, next_domain.pos), next_domain.rc);
    // TODO: Apply curvature?
    next_domain.tbn = create_tangent_space(info.vd.mapped_normal);
    next_domain.flat_normal = info.vd.hard_normal;
    next_domain.view = dir;
    next_domain.tview = view_to_tangent_space(next_domain.view, next_domain.tbn);

    if(bounce_index == MAX_BOUNCES-1)
        bounces = false;

    return bounces;
}

bool replay_path_nee_leaf(
    int bounce_index,
    inout uint path_seed,
    inout vec3 path_throughput,
    inout vec4 primary_bsdf,
    inout domain src,
    bool in_past,
    inout bool head_allows_reconnection,
    inout bool reconnected
){
    ray_cone_apply_roughness(src.mat.roughness, src.rc);

    vec3 candidate_dir = vec3(0);
    vec3 candidate_normal = vec3(0);
    float candidate_dist = 0.0f;
    float nee_pdf = 0.0f;
    uvec4 rand32 = pcg1to4(path_seed);
    reconnection_vertex vertex;
    bool extremity = generate_nee_vertex(rand32, src, vertex, candidate_normal, candidate_dir, candidate_dist, nee_pdf);

    if(bounce_index == 0)
        bias_ray(src.pos, candidate_dir, candidate_dist, src);

    if(vertex.instance_id == NULL_INSTANCE_ID)
        return false;

    reconnected = allow_reconnection(candidate_dist, src.mat, false, head_allows_reconnection);

    rand32.w += 7u;
    vec3 tdir = candidate_dir * src.tbn;
    bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
    float bsdf_pdf = ggx_bsdf_pdf(tdir, src.tview, src.mat, lobes);

    path_throughput *= get_bounce_throughput(
        bounce_index, src, candidate_dir, lobes, primary_bsdf
    ) / nee_pdf;
    path_throughput *=
#ifdef RESTIR_TEMPORAL
        in_past ? test_prev_visibility(rand32.w, src.pos, candidate_dir, candidate_dist, src.flat_normal) :
#endif
        test_visibility(rand32.w, src.pos, candidate_dir, candidate_dist, src.flat_normal);

    src.mat.emission = vertex.radiance_estimate;
    return true;
}

bool replay_path_bsdf_bounce(
    int bounce_index,
    inout uint path_seed,
    inout vec3 path_throughput,
    inout vec4 primary_bsdf,
    inout float regularization,
    inout domain src,
    bool in_past,
    inout bool head_allows_reconnection,
    inout bool reconnected
){
    uvec4 rand32 = pcg1to4(path_seed);

    vec4 u = vec4(rand32) * INV_UINT32_MAX;
    vec3 tdir = vec3(0,0,1);
    bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
    uint sampled_lobe = 0;
    float bsdf_pdf = 0;
    ggx_bsdf_sample_lobe(u, src.tview, src.mat, tdir, lobes, bsdf_pdf, sampled_lobe);

    if(bounce_index == 0)
        bias_ray_origin(src.pos, sampled_lobe == MATERIAL_LOBE_TRANSMISSION, src);

    update_regularization(bsdf_pdf, regularization);
    ray_cone_apply_roughness(sampled_lobe == MATERIAL_LOBE_DIFFUSE ? 1.0f : src.mat.roughness, src.rc);
    if(bsdf_pdf == 0) bsdf_pdf = 1;
    vec3 dir = src.tbn * tdir;

    path_throughput *= get_bounce_throughput(
        bounce_index, src, dir, lobes, primary_bsdf
    ) / bsdf_pdf;

    float nee_pdf;
    domain dst;
    reconnection_vertex vertex;

    bool bounces = generate_bsdf_vertex(rand32, dir, in_past, src, bounce_index, regularization, dst, vertex, nee_pdf);
    reconnected = allow_reconnection(distance(src.pos, dst.pos), dst.mat, bounces, head_allows_reconnection);
    src = dst;
    return bounces;
}

// Will return false if the path ends before the given bounce count was reached.
// Set end_nee = true if the last bounce is selected via next event estimation
// instead of BSDF sampling.
//
// Return value:
// 0 - ended in a normal vertex
// 1 - ended in a terminal vertex
// 2 - ended prematurely
//
// If you expect reconnection after replay_path, only 0 is a valid path. If you
// expect to trace the entire path, both 0 and 1 are valid paths.
int replay_path(
    inout domain src,
    uint seed,
    int path_length,
    bool end_nee,
    bool fail_on_reconnect,
    bool fail_on_last_unconnected,
    bool in_past,
    inout float regularization,
    out vec3 throughput,
    out vec4 primary_bsdf
){
    throughput = vec3(1.0f);
    primary_bsdf = vec4(0);

    int end_state = 0;

    bool head_allows_reconnection = allow_initial_reconnection(src.mat);
    for(int bounce = 0; bounce < min(end_nee ? path_length-1 : path_length, MAX_BOUNCES); ++bounce)
    {
        // Eat NEE sample
        if(!RESTIR_DI)
            pcg1to4(seed);

        bool reconnected = false;
        bool bounces = replay_path_bsdf_bounce(bounce, seed, throughput, primary_bsdf, regularization, src, in_past, head_allows_reconnection, reconnected);
        apply_regularization(regularization, src.mat);

        if(fail_on_reconnect && reconnected)
            return 2;

        if(!bounces)
        {
            if(bounce == path_length-1)
                end_state = 1;
            else end_state = 2;
            break;
        }
    }

    if(fail_on_last_unconnected && !head_allows_reconnection)
        return 2;

    if(end_nee && end_state == 0)
    {
        // NEE paths are always terminal.
        bool reconnected = false;
        bool success = replay_path_nee_leaf(path_length-1, seed, throughput, primary_bsdf, src, in_past, head_allows_reconnection, reconnected);
        if(!success || (fail_on_reconnect && reconnected))
            end_state = 2;
        else
            end_state = 1;
    }
    return end_state;
}

void update_tail_radiance(domain tail_domain, float regularization, bool end_nee, uint tail_length, uint tail_rng_seed, inout vec3 radiance, inout vec3 tail_dir)
{
    vec3 path_throughput = vec3(1.0f);

    for(int bounce = 0; bounce < min(end_nee ? tail_length-1 : tail_length, MAX_BOUNCES); ++bounce)
    {
        // Eat NEE sample
        pcg1to4(tail_rng_seed);

        uvec4 rand32 = pcg1to4(tail_rng_seed);

        vec4 u = vec4(rand32) * INV_UINT32_MAX;
        vec3 tdir = vec3(0,0,1);
        bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
        uint sampled_lobe = 0;
        float bsdf_pdf = 0.0f;
        ggx_bsdf_sample_lobe(u, tail_domain.tview, tail_domain.mat, tdir, lobes, bsdf_pdf, sampled_lobe);
        vec3 dir = tail_domain.tbn * tdir;
        update_regularization(bsdf_pdf, regularization);

        if(bsdf_pdf == 0) bsdf_pdf = 1;

        if(bounce != 0)
        {
            path_throughput *= modulate_bsdf(tail_domain.mat, lobes);
#ifdef USE_PRIMARY_SAMPLE_SPACE
            path_throughput /= bsdf_pdf;
#endif
        }
        //else tail_dir = tail_domain.tbn * tdir;

        domain dst;
        reconnection_vertex vertex;
        float nee_pdf;
        bool bounces = generate_bsdf_vertex(rand32, dir, false, tail_domain, bounce, regularization, dst, vertex, nee_pdf);

        tail_domain = dst;

        radiance = path_throughput * dst.mat.emission;
        if(!bounces)
            return;
    }

    if(end_nee)
    {
        // NEE paths are always terminal.
        vec3 candidate_dir = vec3(0);
        vec3 candidate_normal = vec3(0);
        float candidate_dist = 0.0f;
        float nee_pdf = 0.0f;
        uvec4 rand32 = pcg1to4(tail_rng_seed);
        reconnection_vertex vertex;
        bool extremity = generate_nee_vertex(rand32, tail_domain, vertex, candidate_normal, candidate_dir, candidate_dist, nee_pdf);

        if(vertex.instance_id == NULL_INSTANCE_ID)
            return;

        rand32.w += 7u;

        float visibility =
            self_shadow(-tail_domain.view, tail_domain.flat_normal, candidate_dir, tail_domain.mat) *
            test_visibility(rand32.w, tail_domain.pos, candidate_dir, candidate_dist, tail_domain.flat_normal);

        path_throughput *= visibility;
        if(tail_length <= 1)
            ; //tail_dir = candidate_dir;
        else
        {
            vec3 tdir = candidate_dir * tail_domain.tbn;
            bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
            ggx_bsdf(tdir, tail_domain.tview, tail_domain.mat, lobes);
            path_throughput *= modulate_bsdf(tail_domain.mat, lobes);
#ifdef USE_PRIMARY_SAMPLE_SPACE
            path_throughput /= nee_pdf;
#endif
        }
        radiance = path_throughput * vertex.radiance_estimate;

    }
}

bool reconnection_shift_map(
    inout uint seed,
    inout restir_sample rs,
#ifdef RESTIR_TEMPORAL
    // If true, rs is defined in the current frame and is shifted into the
    // previous frame.
    // If false, rs is defined in the previous frame and is shifted into the
    // current frame.
    bool cur_to_prev,
#endif
    domain to_domain,
    bool do_visibility_test,
    bool update_sample,
    out float jacobian,
    out vec3 contribution,
    out vec4 primary_bsdf
){
#ifndef RESTIR_TEMPORAL
    const bool cur_to_prev = false;
#endif

    jacobian = 1;
    contribution = vec3(0);
    primary_bsdf = vec4(0);

    if(rs.vertex.instance_id == NULL_INSTANCE_ID || rs.vertex.instance_id == UNCONNECTED_PATH_ID)
        return false;

    ray_cone_apply_roughness(rs.head_lobe == MATERIAL_LOBE_DIFFUSE ? 1.0f : to_domain.mat.roughness, to_domain.rc);

    float regularization = 1;
    resolved_vertex to;
    if(!resolve_reconnection_vertex(rs, !cur_to_prev, to_domain, regularization, to))
    {
        // Failed to reconnect, vertex may have ceased to exist.
        return false;
    }

    bias_ray(to_domain.pos, to.dir, to.dist, to_domain);

    vec3 emission = to.emission;
    if(rs.tail_length >= 1)
    {
        // The reconnection vertex wasn't the last path vertex, so we need the
        // full material data too now.
        float nee_pdf;
        vertex_data vd = get_interpolated_vertex(
            to.dir,
            rs.vertex.hit_info,
            int(rs.vertex.instance_id),
            int(rs.vertex.primitive_id)
#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
            , to_domain.pos, nee_pdf
#endif
        );

        ray_cone rc = to_domain.rc;
        ray_cone_apply_dist(to.dist, rc);
        vec2 puvdx;
        vec2 puvdy;
        ray_cone_gradients(
            rc,
            to.dir,
            vd.hard_normal,
            vd.pos,
            vd.uv.xy,
            vd.triangle_pos,
            vd.triangle_uv,
            puvdx, puvdy
        );

        sampled_material mat = sample_material(int(rs.vertex.instance_id), vd, puvdx, puvdy);
        apply_regularization(regularization, mat);
        ray_cone_apply_roughness(rs.tail_lobe == MATERIAL_LOBE_DIFFUSE ? 1.0f : mat.roughness, rc);

        mat3 tbn = create_tangent_space(vd.mapped_normal);
        vec3 tview = -to.dir * tbn;
        vec3 incident_dir = rs.vertex.incident_direction * tbn;

        // Turn radiance into emission
        bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
        float pdf = ggx_bsdf_lobe_pdf(rs.tail_lobe, incident_dir, tview, mat, lobes);
        update_regularization(pdf, regularization);
        vec3 bsdf = modulate_bsdf(mat, lobes);

        // Optionally, update radiance based on tail path in the timeframe of
        // to_domain. This is only needed for temporal reuse.
#if defined(RESTIR_TEMPORAL) && !defined(ASSUME_UNCHANGED_RECONNECTION_RADIANCE)
        if(!cur_to_prev)
        {
            domain tail_domain;
            tail_domain.mat = mat;
            tail_domain.pos = vd.pos;
            tail_domain.tbn = tbn;
            tail_domain.flat_normal = vd.hard_normal;
            tail_domain.view = to.dir;
            tail_domain.tview = tview;
            tail_domain.rc = rc;
            update_tail_radiance(tail_domain, regularization, rs.nee_terminal, rs.tail_length, rs.tail_rng_seed, rs.vertex.radiance_estimate, rs.vertex.incident_direction);
        }
#endif

        emission = bsdf * rs.vertex.radiance_estimate;
    }

    contribution =
        get_bounce_throughput(0, to_domain, to.dir, to.lobes, primary_bsdf) * emission;

    if(do_visibility_test)
    {
#ifdef RESTIR_TEMPORAL
        contribution *= cur_to_prev ?
            test_prev_visibility(seed, to_domain.pos, to.dir, to.dist, to_domain.flat_normal) :
            test_visibility(seed, to_domain.pos, to.dir, to.dist, to_domain.flat_normal);
#else
        contribution *= test_visibility(seed, to_domain.pos, to.dir, to.dist, to_domain.flat_normal);
#endif
    }

    float half_jacobian = reconnection_shift_half_jacobian(to.dir, to.dist, to.normal);

    jacobian = half_jacobian / rs.base_path_jacobian_part;
    if(isinf(jacobian) || isnan(jacobian)) jacobian = 0;

    if(update_sample)
    {
        rs.base_path_jacobian_part = half_jacobian;
        rs.radiance_luminance = rgb_to_luminance(contribution);
    }
    return true;
}

bool random_replay_shift_map(
    inout uint seed,
    inout restir_sample rs,
#ifdef RESTIR_TEMPORAL
    // If true, rs is defined in the current frame and is shifted into the
    // previous frame.
    // If false, rs is defined in the previous frame and is shifted into the
    // current frame.
    bool cur_to_prev,
#endif
    domain to_domain,
    bool update_sample,
    out float jacobian,
    out vec3 contribution,
    out vec4 primary_bsdf
){
#ifndef RESTIR_TEMPORAL
    const bool cur_to_prev = false;
#endif

    jacobian = 1;
    contribution = vec3(0);
    primary_bsdf = vec4(0);

    if(rs.vertex.instance_id == NULL_INSTANCE_ID)
        return false;

    vec3 throughput = vec3(1);
    float regularization = 1.0f;
    int replay_status = replay_path(
        to_domain,
        rs.head_rng_seed,
        int(rs.head_length + 1 + rs.tail_length),
        rs.nee_terminal,
        false,
        false,
        cur_to_prev,
        regularization,
        throughput,
        primary_bsdf
    );
    if(replay_status == 2)
    {
        // Failed to replay path here, so contribution is zero.
        return false;
    }
    contribution = throughput * to_domain.mat.emission;
    if(update_sample)
        rs.radiance_luminance = rgb_to_luminance(contribution);
    return true;
}

bool hybrid_shift_map(
    inout uint seed,
    inout restir_sample rs,
#ifdef RESTIR_TEMPORAL
    // If true, rs is defined in the current frame and is shifted into the
    // previous frame.
    // If false, rs is defined in the previous frame and is shifted into the
    // current frame.
    bool cur_to_prev,
#endif
    domain to_domain,
    bool do_visibility_test,
    bool update_sample,
    out float jacobian,
    out vec3 contribution,
    out vec4 primary_bsdf
){
#ifndef RESTIR_TEMPORAL
    const bool cur_to_prev = false;
#endif

    jacobian = 1;
    contribution = vec3(0);
    primary_bsdf = vec4(0);

    if(rs.vertex.instance_id == NULL_INSTANCE_ID)
        return false;

    vec3 throughput = vec3(1);
    float regularization = 1.0f;
    bool has_reconnection = rs.vertex.instance_id != UNCONNECTED_PATH_ID;
    if(!has_reconnection || rs.head_length != 0)
    {
        int replay_status = replay_path(
            to_domain,
            rs.head_rng_seed,
            int(rs.head_length) + (has_reconnection ? 0 : 1),
            !has_reconnection && rs.nee_terminal,
            true,
            has_reconnection,
            cur_to_prev,
            regularization,
            throughput,
            primary_bsdf
        );

        if(!has_reconnection)
        { // This path has no reconnection so it's just a random replay path.
            if(replay_status == 2) // Failed to replay path
                return false;
            contribution = throughput * to_domain.mat.emission;
            if(update_sample)
                rs.radiance_luminance = rgb_to_luminance(contribution);
            return true;
        }
        else if(replay_status != 0) // Failed to replay path
            return false;
    }

    if(!allow_initial_reconnection(to_domain.mat))
        return false;

    ray_cone_apply_roughness(rs.head_lobe == MATERIAL_LOBE_DIFFUSE ? 1.0f : to_domain.mat.roughness, to_domain.rc);

    resolved_vertex to;
    if(!resolve_reconnection_vertex(rs, !cur_to_prev, to_domain, regularization, to))
    {
        // Failed to reconnect, vertex may have ceased to exist.
        return false;
    }

    if(rs.head_length == 0)
        bias_ray(to_domain.pos, to.dir, to.dist, to_domain);

    float v0_pdf = to.bsdf_pdf;
    float v1_pdf = 1.0f;

    vec3 emission = to.emission;
    if(rs.tail_length >= 1)
    {
        // The reconnection vertex wasn't the last path vertex, so we need the
        // full material data too now.
        float nee_pdf;
        vertex_data vd = get_interpolated_vertex(
            to.dir,
            rs.vertex.hit_info,
            int(rs.vertex.instance_id),
            int(rs.vertex.primitive_id)
#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
            , to_domain.pos, nee_pdf
#endif
        );

        ray_cone rc = to_domain.rc;
        ray_cone_apply_dist(to.dist, rc);
        vec2 puvdx;
        vec2 puvdy;
        ray_cone_gradients(
            rc,
            to.dir,
            vd.hard_normal,
            vd.pos,
            vd.uv.xy,
            vd.triangle_pos,
            vd.triangle_uv,
            puvdx, puvdy
        );

        sampled_material mat = sample_material(int(rs.vertex.instance_id), vd, puvdx, puvdy);
        apply_regularization(regularization, mat);
        ray_cone_apply_roughness(rs.tail_lobe == MATERIAL_LOBE_DIFFUSE ? 1.0f : mat.roughness, rc);
        mat3 tbn = create_tangent_space(vd.mapped_normal);

        bool allowed = true;
        if(!allow_reconnection(to.dist, mat, true, allowed))
            return false;

        vec3 tview = -to.dir * tbn;
        vec3 incident_dir = rs.vertex.incident_direction * tbn;

        // Turn radiance into to.emission
        bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
        v1_pdf = ggx_bsdf_lobe_pdf(rs.tail_lobe, incident_dir, tview, mat, lobes);
        update_regularization(v1_pdf, regularization);
        vec3 bsdf = modulate_bsdf(mat, lobes);

        // Optionally, update radiance based on tail path in the timeframe of
        // to_domain. This is only needed for temporal reuse.
#if defined(RESTIR_TEMPORAL) && !defined(ASSUME_UNCHANGED_RECONNECTION_RADIANCE)
        if(!cur_to_prev)
        {
            domain tail_domain;
            tail_domain.mat = mat;
            tail_domain.pos = vd.pos;
            tail_domain.tbn = tbn;
            tail_domain.flat_normal = vd.hard_normal;
            tail_domain.view = to.dir;
            tail_domain.tview = tview;
            tail_domain.rc = rc;
            update_tail_radiance(tail_domain, regularization, rs.nee_terminal, rs.tail_length, rs.tail_rng_seed, rs.vertex.radiance_estimate, rs.vertex.incident_direction);
        }
#endif

        emission = bsdf * rs.vertex.radiance_estimate;
        if(isnan(v1_pdf) || isinf(v1_pdf) || v1_pdf == 0)
        {
            v1_pdf = 0;
            emission = vec3(0);
        }
        else emission /= v1_pdf;
    }
    else
    {
        bool allowed = true;
        if(!allow_reconnection(to.dist, to_domain.mat, false, allowed))
            return false;
    }

    contribution = throughput * get_bounce_throughput(rs.head_length, to_domain, to.dir, to.lobes, primary_bsdf) * emission;

    if(isnan(v0_pdf) || isinf(v0_pdf) || v0_pdf == 0)
        contribution = vec3(0);
    else contribution /= v0_pdf;

    if(do_visibility_test)
    {
#ifdef RESTIR_TEMPORAL
        contribution *= cur_to_prev && rs.head_length == 0 ?
            test_prev_visibility(seed, to_domain.pos, to.dir, to.dist, to_domain.flat_normal) :
            test_visibility(seed, to_domain.pos, to.dir, to.dist, to_domain.flat_normal);
#else
        contribution *= test_visibility(seed, to_domain.pos, to.dir, to.dist, to_domain.flat_normal);
#endif
    }

    float half_jacobian = v0_pdf * v1_pdf * reconnection_shift_half_jacobian(to.dir, to.dist, to.normal);

    jacobian = half_jacobian / rs.base_path_jacobian_part;
    if(isinf(jacobian) || isnan(jacobian))
        jacobian = 0;

    if(update_sample)
    {
        rs.base_path_jacobian_part = half_jacobian;
        rs.radiance_luminance = rgb_to_luminance(contribution);
    }
    return true;
}

bool shift_map(
    inout uint seed,
    inout restir_sample rs,
#ifdef RESTIR_TEMPORAL
    // If true, rs is defined in the current frame and is shifted into the
    // previous frame.
    // If false, rs is defined in the previous frame and is shifted into the
    // current frame.
    bool cur_to_prev,
#endif
    domain to_domain,
    bool do_visibility_test,
    bool update_sample,
    out float jacobian,
    out vec3 contribution,
    out vec4 primary_bsdf
){
#ifdef USE_RECONNECTION_SHIFT
    return reconnection_shift_map(
        seed, rs,
#ifdef RESTIR_TEMPORAL
        cur_to_prev,
#endif
        to_domain, do_visibility_test, update_sample,
        jacobian, contribution, primary_bsdf
    );
#elif defined(USE_RANDOM_REPLAY_SHIFT)
    return random_replay_shift_map(
        seed, rs,
#ifdef RESTIR_TEMPORAL
        cur_to_prev,
#endif
        to_domain, update_sample,
        jacobian, contribution, primary_bsdf
    );
#else
    return hybrid_shift_map(
        seed, rs,
#ifdef RESTIR_TEMPORAL
        cur_to_prev,
#endif
        to_domain, do_visibility_test, update_sample,
        jacobian, contribution, primary_bsdf
    );
#endif
}
#endif

struct spatial_candidate_sampler
{
    vec3 view_pos;
    vec3 view_tangent;
    vec3 view_bitangent;
    vec4 proj_info;
};

spatial_candidate_sampler init_spatial_candidate_sampler(
    camera_data cam, vec3 pos, mat3 tbn, float max_plane_dist
){
    spatial_candidate_sampler scs;
    scs.view_pos = (cam.view * vec4(pos, 1)).xyz;
    scs.view_tangent = (mat3(cam.view) * tbn[0]) * max_plane_dist * 2.0f;
    scs.view_bitangent = (mat3(cam.view) * tbn[1]) * max_plane_dist * 2.0f;
    scs.proj_info.xy = cam.projection_info.zw;
    scs.proj_info.y = -scs.proj_info.y;
    scs.proj_info.zw = cam.pan.xy;
    return scs;
}

ivec2 spatial_candidate_pos(
    ivec2 p,
    ivec2 size,
    spatial_candidate_sampler scs,
    uint camera_index,
    uint sample_index,
    uint candidate_index,
    uint attempt_index
){
    uvec4 seed = uvec4(p, pc.camera_index, pc.sample_index) + attempt_index;
    vec2 u = generate_single_uniform_random(seed).xy;
    u = weyl(u.xy, int(candidate_index));

#ifdef NEIGHBOR_SAMPLE_ORIENTED_DISKS
    vec2 range = vec2(
        TR_RESTIR.min_spatial_radius,
        TR_RESTIR.max_spatial_radius - TR_RESTIR.min_spatial_radius
    );
    vec2 s = sample_ring(u, range.x, range.y);
    vec3 q = scs.view_pos + scs.view_tangent * s.x + scs.view_bitangent * s.y;
    q.xy = (0.5 - q.xy / (scs.proj_info.xy * q.z) - 0.5f * scs.proj_info.zw) * size;
    return ivec2(mirror_wrap(round(q.xy), vec2(0), vec2(size)-0.01f));
#else
    vec2 range = vec2(
        TR_RESTIR.min_spatial_radius,
        TR_RESTIR.max_spatial_radius - TR_RESTIR.min_spatial_radius
    ) * size.x;
    vec2 s = sample_ring(u, range.x, range.y);
    return ivec2(mirror_wrap(p+round(s), vec2(0), vec2(size)-0.01f));
#endif
}

ivec2 spatial_candidate_pos(
    uint key,
    ivec2 p,
    ivec2 size,
    spatial_candidate_sampler scs,
    uint camera_index,
    uint sample_index,
    uint candidate_index
){
    return spatial_candidate_pos(
        p, size, scs, camera_index, sample_index, candidate_index, (key >> (candidate_index*2u))&3
    );
}

float mis_canonical(
    float canonical_confidence,
    float other_confidence,
    float total_confidence,
    float canonical_target_function_value,
    float canonical_in_other_target_function_value,
    float jacobian_canonical_to_other
){
    if(canonical_target_function_value == 0)
        return 0.0f;

    float w = canonical_confidence * canonical_target_function_value;
    return (other_confidence / total_confidence) * w /
        (w + (total_confidence - canonical_confidence) * canonical_in_other_target_function_value * jacobian_canonical_to_other);
}

float mis_noncanonical(
    float canonical_confidence,
    float other_confidence,
    float total_confidence,
    float other_target_function_value,
    float other_in_canonical_target_function_value,
    float jacobian_other_to_canonical
){
    if(other_target_function_value == 0)
        return 0.0f;

    float w = (total_confidence - canonical_confidence) * other_target_function_value;
    return (other_confidence / total_confidence) * w /
        (w + canonical_confidence * other_in_canonical_target_function_value * jacobian_other_to_canonical);
}

// https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22699-fast-denoising-with-self-stabilizing-recurrent-blurs.pdf
float disocclusion_detect(vec3 normal1, vec3 pos1, vec3 pos2, float inv_max_plane_dist)
{
    return clamp(1.0f - abs(dot(normal1, pos2-pos1)) * inv_max_plane_dist, 0.0f, 1.0f);
}

#endif
