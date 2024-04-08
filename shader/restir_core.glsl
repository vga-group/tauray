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
layout(binding = 4) uniform sampler2D emission_tex;
layout(binding = 5) uniform sampler2D material_tex;


layout(binding = 6, rgba32ui) readonly uniform uimage2D in_reservoir_ris_data_tex;
layout(binding = 7, rgba32ui) readonly uniform uimage2D in_reservoir_reconnection_data_tex;
layout(binding = 8, rgba32f) readonly uniform image2D in_reservoir_reconnection_radiance_tex;
layout(binding = 9, rg32ui) readonly uniform uimage2D in_reservoir_rng_seeds_tex;

layout(binding = 10, rgba32ui) uniform uimage2D out_reservoir_ris_data_tex;
layout(binding = 11, rgba32ui) uniform uimage2D out_reservoir_reconnection_data_tex;
layout(binding = 12, rgba32f) uniform image2D out_reservoir_reconnection_radiance_tex;
layout(binding = 13, rg32ui) uniform uimage2D out_reservoir_rng_seeds_tex;

#include "alias_table.glsl"
#include "math.glsl"
#include "random_sampler.glsl"
#include "ggx.glsl"

#ifdef RESTIR_TEMPORAL
layout(binding = 14) uniform sampler2D prev_depth_or_position_tex;
layout(binding = 15) uniform sampler2D prev_normal_tex;
layout(binding = 16) uniform sampler2D prev_flat_normal_tex;
layout(binding = 17) uniform sampler2D prev_albedo_tex;
layout(binding = 18) uniform sampler2D prev_emission_tex;
layout(binding = 19) uniform sampler2D prev_material_tex;
layout(binding = 20) uniform sampler2D motion_tex;

//#include "temporal_tables.glsl"
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
};

sampled_material get_material(ivec2 p)
{
    sampled_material mat = sample_gbuffer_material(
        albedo_tex, material_tex, emission_tex, p
    );
    // Reconnection shift cannot deal with roughness == 0 extremities
    // correctly, so we just set the roughness to _almost_ zero but not
    // quite. This avoids having to tag whether a ray was an extremity ray
    // or not, while producing _almost_ the same results anyway.
    // This should not be an issue with hybrid shift, as it would only use
    // reconnection shift in cases where the reconnection lobes are rough.
#ifdef USE_RECONNECTION_SHIFT
    mat.roughness = max(mat.roughness, 0.001f);
#endif
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
    pos = (cam.view_inverse * vec4(unproject_position(linear_depth, vec2(uv.x, 1-uv.y), cam.projection_info), 1)).xyz;
    return depth == 1.0f;
#endif
}

bool read_domain(camera_data  cam, ivec2 p, out domain d)
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

    d.view = normalize(d.pos - origin);
    d.tview = view_to_tangent_space(d.view, d.tbn);
    return miss;
}

#ifdef RESTIR_TEMPORAL
sampled_material get_prev_material(ivec2 p)
{
    sampled_material mat = sample_gbuffer_material(
        prev_albedo_tex, prev_material_tex, prev_emission_tex, p
    );
#ifdef USE_RECONNECTION_SHIFT
    mat.roughness = max(mat.roughness, 0.001f);
#endif
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
    pos = (cam.view_inverse * vec4(unproject_position(linear_depth, vec2(uv.x, 1-uv.y), cam.projection_info), 1)).xyz;
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

    d.view = normalize(d.pos - origin);
    d.tview = view_to_tangent_space(d.view, d.tbn);
    return miss;
}
#endif

reservoir unpack_reservoir(
    uvec4 ris_data,
    uvec4 reconnection_data,
    vec4 reconnection_radiance,
    uvec2 rng_seeds
){
    reservoir r;

    r.target_function_value = uintBitsToFloat(ris_data.r);
    r.ucw = uintBitsToFloat(ris_data.g);
    r.output_sample.base_path_jacobian_part = uintBitsToFloat(ris_data.b);

    uint confidence_path_length = ris_data[3];
    r.confidence = bitfieldExtract(confidence_path_length, 0, 15);
    r.output_sample.nee_terminal= bitfieldExtract(confidence_path_length, 15, 1) == 1 ? true : false;
    r.output_sample.head_length = bitfieldExtract(confidence_path_length, 16, 8);
    r.output_sample.tail_length = bitfieldExtract(confidence_path_length, 24, 8);

    r.output_sample.vertex.hit_info = unpackUnorm2x16(reconnection_data[0]);
    r.output_sample.vertex.incident_direction = octahedral_unpack(unpackSnorm2x16(reconnection_data[1]));
    r.output_sample.vertex.instance_id = r.ucw <= 0 ? NULL_INSTANCE_ID : reconnection_data[2];
    r.output_sample.vertex.primitive_id = reconnection_data[3];

    r.output_sample.vertex.radiance_estimate = reconnection_radiance.rgb;

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
        RESTIR_HAS_SEEDS ? imageLoad(in_reservoir_rng_seeds_tex, p).rg : uvec2(0)
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
        RESTIR_HAS_SEEDS ? imageLoad(out_reservoir_rng_seeds_tex, p).rg : uvec2(0)
    );
}

void write_reservoir(reservoir r, ivec2 p, uvec2 size)
{
    r.confidence = min(r.confidence, TR_RESTIR.max_confidence);
    uint reconnection_info = 0;
    reconnection_info = bitfieldInsert(reconnection_info, int(r.confidence), 0, 15);
    reconnection_info = bitfieldInsert(reconnection_info, r.output_sample.nee_terminal ? 1 : 0, 15, 1);
    reconnection_info = bitfieldInsert(reconnection_info, r.output_sample.head_length, 16, 8);
    reconnection_info = bitfieldInsert(reconnection_info, r.output_sample.tail_length, 24, 8);
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
            packUnorm2x16(r.output_sample.vertex.hit_info),
            packSnorm2x16(octahedral_pack(r.output_sample.vertex.incident_direction)),
            r.output_sample.vertex.instance_id,
            r.output_sample.vertex.primitive_id
        );
        imageStore(out_reservoir_reconnection_data_tex, p, reconnection_data);
        imageStore(out_reservoir_reconnection_radiance_tex, p, vec4(r.output_sample.vertex.radiance_estimate, 0));
    }

    if(RESTIR_HAS_SEEDS)
    {
        uvec2 rng_seeds = uvec2(
            r.output_sample.head_rng_seed,
            r.output_sample.tail_rng_seed
        );
        imageStore(out_reservoir_rng_seeds_tex, p, uvec4(rng_seeds,0,0));
    }
}

int self_shadow(vec3 view, vec3 flat_normal, vec3 light_dir, sampled_material mat)
{
    return dot(flat_normal, light_dir) * dot(view, flat_normal) < 0 && mat.transmittance == 0.0f ? 0 : 1;
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
        primary_bsdf.rgb = bsdf * inv_scale;
        primary_bsdf.a = scale <= 0 ? 0 : reflection * inv_scale;
        bsdf = vec3(scale);
    }
#endif
    return bsdf * self_shadow(-src.view, src.flat_normal, dir, src.mat);
}

struct sum_contribution
{
    vec3 diffuse;
    vec3 reflection;
    float canonical_reflection;
};

sum_contribution init_sum_contribution(vec3 emission)
{
    sum_contribution sc;
#ifdef DEMODULATE_OUTPUT
    sc.diffuse = vec3(0);
    sc.reflection = vec3(0);
    sc.canonical_reflection = 0;
#else
    sc.diffuse = emission;
#endif
    return sc;
}

void add_contribution(inout sum_contribution sc, reservoir r, vec4 contrib, float weight)
{
#ifdef DEMODULATE_OUTPUT
    vec3 color = contrib.rgb * weight;
    sc.diffuse += color * (1.0f - contrib.a);
    sc.reflection += color * contrib.a;
#else
    sc.diffuse += contrib.rgb * weight;
#endif
}

void add_canonical_contribution(inout sum_contribution sc, reservoir r, vec4 contrib, float weight)
{
    add_contribution(sc, r, contrib, weight);
#ifdef DEMODULATE_OUTPUT
    sc.canonical_reflection = contrib.a;
#endif
}

#ifdef DEMODULATE_OUTPUT
#define finish_output_color(p, reservoir, out_value, sc, display_size) { \
    write_reservoir(reservoir, p, display_size); \
    if(pc.accumulate_color != 0) \
    { \
        float ray_length = imageLoad(out_reflection, p).r; \
        imageStore(out_diffuse, p, vec4(sc.diffuse, sc.canonical_reflection > 0.5 ? 0 : ray_length)); \
        imageStore(out_reflection, p, vec4(sc.reflection, sc.canonical_reflection > 0.5 ? ray_length : 0)); \
    } \
    if(pc.update_sample_color != 0) \
        imageStore(out_diffuse, p, out_value); \
}
#else
#define finish_output_color(p, reservoir, out_value, sc, display_size) { \
    write_reservoir(reservoir, p, display_size); \
    if(pc.update_sample_color != 0) \
        imageStore(out_diffuse, p, out_value); \
    else \
        imageStore(out_reflection, p, vec4(sc.diffuse, 1.0f)); \
}
#endif

#ifdef RAY_TRACING
#ifdef SHADE_ALL_EXPLICIT_LIGHTS
#define SHADOW_MAPPING_SCREEN_COORD vec2(gl_LaunchIDEXT.xy)
#include "shadow_mapping.glsl"
vec3 shade_explicit_lights(
    vec3 view,
    vertex_data vd,
    sampled_material mat
){
    vec3 normal = vd.mapped_normal;
    mat3 tbn = mat3(vd.tangent, vd.bitangent, vd.mapped_normal);
    vec3 tview = view_to_tangent_space(view, tbn);

    vec3 contrib = vec3(0);
#ifdef SHADE_FAKE_INDIRECT
    TODO
#endif

    for(uint i = 0; i < scene_metadata.point_light_count; ++i)
    {
        vec3 d, s;
        point_light pl = point_lights.lights[i];
        vec3 light_dir;
        float light_dist;
        vec3 light_color;
        get_point_light_info(pl, vd.pos, light_dir, light_dist, light_color);
        ggx_brdf(light_dir * tbn, tview, mat, d, s);
        d *= light_color;
        s *= light_color;
        float shadow = 1.0f;
        if(pl.shadow_map_index >= 0 && dot(vd.mapped_normal, light_dir) > 0)
            shadow = calc_point_shadow(
                pl.shadow_map_index, vd.pos, vd.hard_normal, light_dir
            );
        shadow = dot(vd.hard_normal, light_dir) < 0 ? 0 : shadow;
        contrib += (d + s) * shadow;
    }

    for(uint i = 0; i < scene_metadata.directional_light_count; ++i)
    {
        vec3 d, s;
        directional_light dl = directional_lights.lights[i];
        ggx_brdf(-dl.dir * tbn, tview, mat, d, s);
        d *= dl.color;
        s *= dl.color;
        float shadow = 1.0f;
        if(dl.shadow_map_index >= 0 && dot(vd.mapped_normal, -dl.dir) > 0)
            shadow = calc_directional_shadow(
                dl.shadow_map_index, vd.pos, vd.hard_normal, -dl.dir
            );
        shadow = dot(vd.hard_normal, dl.dir) > 0 ? 0 : shadow;
        contrib += (d + s) * shadow;
    }
    return contrib;
}
#endif

int test_visibility(uint seed, vec3 pos, vec3 dir, float dist, vec3 flat_normal)
{
    payload.random_seed = seed;
    float s = sign(dot(dir, flat_normal));

    vec3 origin = pos + s * flat_normal * TR_RESTIR.min_ray_dist;
    vec3 target = pos + dir * dist;

    vec3 new_dir = dir * dist - s * flat_normal * TR_RESTIR.min_ray_dist;
    float new_dist = length(new_dir);
    new_dir /= new_dist;

    traceRayEXT(
        tlas,
#ifdef STOCHASTIC_ALPHA_BLENDING
        gl_RayFlagsTerminateOnFirstHitEXT,
#else
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
#endif
        VISIBILITY_RAY_MASK,
        0, 0, 0,
        origin,
        TR_RESTIR.min_ray_dist,
        new_dir,
        new_dist-TR_RESTIR.min_ray_dist*2,
        0
    );
    return payload.instance_id < 0 ? 1 : 0;
}

#ifdef RESTIR_TEMPORAL
int test_prev_visibility(uint seed, vec3 pos, vec3 dir, float dist, vec3 flat_normal)
{
    payload.random_seed = seed;
    float s = sign(dot(dir, flat_normal));

    vec3 origin = pos + s * flat_normal * TR_RESTIR.min_ray_dist;
    vec3 target = pos + dir * dist;

    vec3 new_dir = dir * dist - s * flat_normal * TR_RESTIR.min_ray_dist;
    float new_dist = length(new_dir);
    new_dir /= new_dist;

#ifdef ASSUME_UNCHANGED_ACCELERATION_STRUCTURES
    traceRayEXT(
        tlas,
        gl_RayFlagsCullNoOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        VISIBILITY_RAY_MASK,
        0, 0, 0,
        origin,
        TR_RESTIR.min_ray_dist,
        new_dir,
        new_dist-TR_RESTIR.min_ray_dist*2,
        0
    );
#else
    traceRayEXT(
        scene_prev_tlas,
        gl_RayFlagsCullNoOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        VISIBILITY_RAY_MASK,
        0, 0, 0,
        origin,
        TR_RESTIR.min_ray_dist,
        new_dir,
        new_dist-TR_RESTIR.min_ray_dist*2,
        0
    );
#endif
    return payload.instance_id < 0 ? 1 : 0;
}
#endif

struct resolved_vertex
{
    vec4 pos; // Homogeneous coordinate, 0 in w indicates direction.
    vec3 normal; // 0 for zero-radius point light sources
    // If the vertex is emissive, this holds the emission. If not, treat the value as undefined.
    vec3 emission;
};

resolved_vertex resolve_point_light_vertex(reconnection_vertex rv, bool prev, vec3 domain_pos)
{
    resolved_vertex v;
#ifdef RESTIR_TEMPORAL
    point_light pl;
    /*if(prev) pl = prev_point_lights.lights[rv.primitive_id];
    else*/ pl = point_lights.lights[rv.primitive_id];
#else
    point_light pl = point_lights.lights[rv.primitive_id];
#endif
    v.pos = vec4(pl.pos, 1);

    vec3 center_dir = v.pos.xyz - domain_pos;
    float dist2 = dot(center_dir, center_dir);

    float radius = pl.radius;

    if(radius != 0)
    {
        v.normal = octahedral_unpack(rv.hit_info * 2.0f - 1.0f);
        v.pos.xyz += radius * v.normal;
    }
    else v.normal = vec3(0);

    v.emission = pl.color;
    if(radius != 0.0f) v.emission /= M_PI * radius * radius;
    v.emission *= get_spotlight_intensity(pl, normalize(center_dir));
    if(radius == 0) v.emission /= dist2;

    return v;
}

resolved_vertex resolve_directional_light_vertex(reconnection_vertex rv)
{
    resolved_vertex v;
    vec3 dir = octahedral_unpack(rv.hit_info * 2.0f - 1.0f);
    v.pos = vec4(dir, 0);
    v.normal = -dir;

#ifdef RESTIR_TEMPORAL
    v.emission = scene_metadata.environment_factor.rgb;
    if(scene_metadata.environment_proj >= 0)
    {
        vec2 uv = vec2(0);
        uv.y = asin(-dir.y)/M_PI+0.5f;
        uv.x = atan(dir.z, dir.x)/(2*M_PI)+0.5f;
        v.emission *= texture(environment_map_tex, uv).rgb;
    }

#ifndef SHADE_ALL_EXPLICIT_LIGHTS
    // Directional lights don't generate vertices when they're rasterized.
    for(uint i = 0; i < scene_metadata.directional_light_count; ++i)
    {
        directional_light dl = directional_lights.lights[i];
        float visible = step(dl.dir_cutoff, dot(dir, -dl.dir));
        v.emission += visible * dl.color / (2.0f * M_PI * (1.0f - dl.dir_cutoff));
    }
#endif

#else
    v.emission = rv.radiance_estimate;
#endif
    return v;
}

void resolve_mesh_triangle_vertex(
    reconnection_vertex rv,
    bool rv_is_in_past,
    vec3 from_pos,
    out resolved_vertex to
){
#if !defined(SHADE_ALL_EXPLICIT_LIGHTS) || defined(ASSUME_UNCHANGED_RECONNECTION_RADIANCE)
    // Normal operation: just find the emissiveness of the vertex.

    vertex_data vd;
    float pdf;
    vd = get_interpolated_vertex(
        vec3(0),
        rv.hit_info,
        int(rv.instance_id),
        int(rv.primitive_id)
#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
        , from_pos, pdf
#endif
    );
    to.pos.xyz = vd.pos;
#ifdef RESTIR_TEMPORAL
    if(!rv_is_in_past)
        to.pos.xyz = vd.prev_pos;
#endif
    to.normal = vd.hard_normal;
    to.pos.w = 1;
    // TODO: Allow mesh triangle material changes?
    to.emission = rv.radiance_estimate;

#else
    // If shading all explicit lights too, we need some extra info.

    instance i = instances.array[nonuniformEXT(rv.instance_id)];
    vertex_data vd = get_vertex_data(rv.instance_id, rv.primitive_id, rv.hit_info, false);
#ifdef RESTIR_TEMPORAL
    if(!rv_is_in_past)
        vd.pos = vd.prev_pos;
#endif
    vec3 ray_direction = normalize(vd.pos - from_pos);

    bool front_facing = dot(ray_direction, vd.flat_normal) < 0;
    material mat = sample_material(i.material, front_facing, vd.uv.xy, vec2(0.0), vec2(0.0f));

    vd.tangent_space = apply_normal_map(vd.tangent_space, mat);
    to.emission = mat.emission + shade_explicit_lights(i.environment, ray_direction, vd, mat);

    to.normal = vd.flat_normal;
    to.pos.w = 1;
    to.pos.xyz = vd.pos;
#endif
}

bool resolve_reconnection_vertex(
    inout reconnection_vertex rv,
    bool rv_is_in_past,
    vec3 pos,
    out resolved_vertex to
){
    // if rv_is_in_past, to is cur. Otherwise, to is prev.
    if(rv.instance_id == NULL_INSTANCE_ID)
    {
        to.pos = vec4(0);
        to.normal = vec3(0);
        to.emission = vec3(0);
        return false;
    }
    else if(rv.instance_id == POINT_LIGHT_INSTANCE_ID)
    {
#ifdef RESTIR_TEMPORAL
        //uint primitive_id;
        //if(rv_is_in_past) primitive_id = point_light_map_forward.array[rv.primitive_id];
        //else primitive_id = point_light_map_backward.array[rv.primitive_id];

        if(rv.primitive_id >= scene_metadata.point_light_count)
            return false;

        to = resolve_point_light_vertex(rv, !rv_is_in_past, pos);
#else
        to = resolve_point_light_vertex(rv, false, pos);
#endif
        return true;
    }
    else if(
        rv.instance_id == DIRECTIONAL_LIGHT_INSTANCE_ID ||
        rv.instance_id == ENVMAP_INSTANCE_ID
    ){
        to = resolve_directional_light_vertex(rv);
        return true;
    }
    else
    { // Regular mesh triangle
#ifdef RESTIR_TEMPORAL
        //if(rv_is_in_past)
        //    rv.instance_id = instance_map_forward.array[rv.instance_id];
        //if(rv.instance_id >= scene_metadata.instance_count)
        //    return false;
#endif
        resolve_mesh_triangle_vertex(rv, rv_is_in_past, pos, to);
        return true;
    }
}
#endif

void homogeneous_to_ray(vec3 from, vec4 to, out vec3 dir, out float len)
{
    if(to.w == 0)
    {
        dir = to.xyz;
        len = TR_RESTIR.max_ray_dist;
    }
    else
    {
        dir = to.xyz - from;
        len = length(dir);
        dir /= len;
    }
}

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
    as_head = !bounces ? true : mat.roughness > 0.05f;
    return head && as_head && (dist > TR_RESTIR.reconnection_scale || !bounces);
#endif
}

#ifdef RAY_TRACING
bool get_intersection_info(
    vec3 ray_origin,
    vec3 ray_direction,
    hit_payload payload,
    bool in_past,
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
        /*if(in_past)
        {
            // Translate previous mesh to current frame
            uint new_id = instance_map_forward.array[payload.instance_id];
            if(new_id == 0xFFFFFFFFu)
            {
                candidate.instance_id = NULL_INSTANCE_ID;
                candidate.primitive_id = 0;
                candidate.hit_info = vec2(0);
                candidate.radiance_estimate = vec3(0);
                return false;
            }
            payload.instance_id = int(new_id);
        }*/
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
#ifdef RESTIR_TEMPORAL
        if(in_past) info.vd.pos = info.vd.prev_pos;
#endif

        bool front_facing = dot(ray_direction, info.vd.hard_normal) < 0;
        info.mat = sample_material(payload.instance_id, info.vd);
#ifdef USE_RECONNECTION_SHIFT
        // See read_domain() for context.
        info.mat.roughness = max(info.mat.roughness, 0.001f);
#endif

        info.local_pdf = any(greaterThan(info.mat.emission, vec3(0))) ? pdf : 0;

        candidate.instance_id = payload.instance_id;
        candidate.primitive_id = payload.primitive_id;
        candidate.hit_info = payload.barycentrics;
        candidate.radiance_estimate = info.mat.emission;

#ifdef SHADE_ALL_EXPLICIT_LIGHTS
        candidate.radiance_estimate += shade_explicit_lights(
            ray_direction,
            info.vd,
            info.mat
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
        /*
        if(in_past)
        {
            pl = prev_point_lights.lights[payload.primitive_id];
            uint new_id = point_light_map_forward.array[payload.primitive_id];
            if(new_id == 0xFFFFFFFFu)
            {
                candidate.instance_id = NULL_INSTANCE_ID;
                candidate.primitive_id = 0;
                candidate.hit_info = vec2(0);
                candidate.radiance_estimate = vec3(0);
                return false;
            }
            payload.primitive_index = int(new_id);
        }
        else
        */
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

        // TODO: Not sure what to do here... This also receives directional
        // lights, which can and do overlap with the envmap.
        candidate.instance_id = ENVMAP_INSTANCE_ID;
        candidate.primitive_id = 0;
        candidate.hit_info = octahedral_pack(ray_direction) * 0.5f + 0.5f;
        candidate.radiance_estimate = info.light;
        info.vd.pos = ray_origin + TR_RESTIR.max_ray_dist * ray_direction;
        return false;
    }
}

bool trace_ray(
    uint seed,
    vec3 origin,
    vec3 dir,
    out intersection_info info,
    out reconnection_vertex candidate
){
    payload.random_seed = seed;
    traceRayEXT(
        tlas,
#ifdef STOCHASTIC_ALPHA_BLENDING
        gl_RayFlagsNoneEXT,
#else
        gl_RayFlagsOpaqueEXT,
#endif
        RAY_MASK,
        0,
        0,
        0,
        origin,
        TR_RESTIR.min_ray_dist,
        dir,
        TR_RESTIR.max_ray_dist,
        0
    );

    return get_intersection_info(origin, dir, payload, false, info, candidate);
}

#ifdef RESTIR_TEMPORAL
bool trace_prev_ray(
    uint seed,
    vec3 origin,
    vec3 dir,
    out intersection_info info,
    out reconnection_vertex candidate
){
    payload.random_seed = seed;
#ifdef ASSUME_UNCHANGED_ACCELERATION_STRUCTURES
    traceRayEXT(
        tlas,
#ifdef STOCHASTIC_ALPHA_BLENDING
        gl_RayFlagsNoneEXT,
#else
        gl_RayFlagsOpaqueEXT,
#endif
        RAY_MASK,
        0,
        0,
        0,
        origin,
        TR_RESTIR.min_ray_dist,
        dir,
        TR_RESTIR.max_ray_dist,
        0
    );
    return get_intersection_info(origin, dir, payload, false, info, candidate);
#else
    traceRayEXT(
        prev_tlas,
        gl_RayFlagsOpaqueEXT,
        RAY_MASK,
        0,
        0,
        0,
        origin,
        TR_RESTIR.min_ray_dist,
        dir,
        TR_RESTIR.max_ray_dist,
        0
    );
    return get_intersection_info(origin, dir, payload, true, info, candidate);
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
    vec3 pos,
    float min_dist,
    float max_dist
){
    float point_prob, triangle_prob, dir_prob, envmap_prob;
    get_nee_sampling_probabilities(point_prob, triangle_prob, dir_prob, envmap_prob);

    light_sample ls;
    ls.color = vec3(0);

    vec4 u = ldexp(vec4(rand32), ivec4(-32));

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
        if(local_pdf == 0.0f) ls.normal = vec3(0);
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

        ls.color = r9g9b9e5_to_rgb(tl.emission_factor);
        if(tl.emission_tex_id >= 0)
        { // Textured emissive triangle, so read texture.
            vec2 uv =
                bary.x * unpackHalf2x16(tl.uv[0]) +
                bary.y * unpackHalf2x16(tl.uv[1]) +
                bary.z * unpackHalf2x16(tl.uv[2]);
            ls.color *= texture(textures[nonuniformEXT(tl.emission_tex_id)], uv).rgb;
        }

        ls.normal = normalize(cross(
            tl.pos[2] - tl.pos[0],
            tl.pos[1] - tl.pos[0]
        ));

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

    ls.infinitesimal = local_pdf == 0;
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

    if(
        instance_id == DIRECTIONAL_LIGHT_INSTANCE_ID ||
        instance_id == ENVMAP_INSTANCE_ID
    ){
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
    out domain next_domain, // Only valid if true is returned.
    out reconnection_vertex vertex,
    out float nee_pdf
){
    intersection_info info;
    bool bounces =
#ifdef RESTIR_TEMPORAL
        in_past ? trace_prev_ray(rand32.w, cur_domain.pos, dir, info, vertex) :
#endif
        trace_ray(rand32.w, cur_domain.pos, dir, info, vertex);
    nee_pdf = calculate_light_pdf(
        vertex.instance_id, vertex.primitive_id, info.local_pdf, info.envmap_pdf
    );

    next_domain.mat = info.mat;
    next_domain.mat.emission = vertex.radiance_estimate;
    next_domain.pos = info.vd.pos;
    next_domain.tbn = create_tangent_space(info.vd.mapped_normal);
    next_domain.flat_normal = info.vd.hard_normal;
    next_domain.view = dir;
    next_domain.tview = view_to_tangent_space(next_domain.view, next_domain.tbn);

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
    vec3 candidate_dir = vec3(0);
    vec3 candidate_normal = vec3(0);
    float candidate_dist = 0.0f;
    float nee_pdf = 0.0f;
    uvec4 rand32 = pcg1to4(path_seed);
    reconnection_vertex vertex;
    bool extremity = generate_nee_vertex(rand32, src, vertex, candidate_normal, candidate_dir, candidate_dist, nee_pdf);

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
    inout domain src,
    bool in_past,
    inout bool head_allows_reconnection,
    inout bool reconnected
){
    uvec4 rand32 = pcg1to4(path_seed);

    vec4 u = ldexp(vec4(rand32), ivec4(-32));
    vec3 tdir = vec3(0,0,1);
    bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
    float bsdf_pdf = 0;
    ggx_bsdf_sample(u, src.tview, src.mat, tdir, lobes, bsdf_pdf);
    if(bsdf_pdf == 0) bsdf_pdf = 1;
    vec3 dir = src.tbn * tdir;

    path_throughput *= get_bounce_throughput(
        bounce_index, src, dir, lobes, primary_bsdf
    ) / bsdf_pdf;

    float nee_pdf;
    domain dst;
    reconnection_vertex vertex;
    bool bounces = generate_bsdf_vertex(rand32, dir, in_past, src, dst, vertex, nee_pdf);
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
        bool bounces = replay_path_bsdf_bounce(bounce, seed, throughput, primary_bsdf, src, in_past, head_allows_reconnection, reconnected);

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

void update_tail_radiance(domain tail_domain, bool end_nee, uint tail_length, uint tail_rng_seed, inout vec3 radiance, inout vec3 tail_dir)
{
    vec3 path_throughput = vec3(1.0f);

    for(int bounce = 0; bounce < min(end_nee ? tail_length-1 : tail_length, MAX_BOUNCES); ++bounce)
    {
        // Eat NEE sample
        pcg1to4(tail_rng_seed);

        uvec4 rand32 = pcg1to4(tail_rng_seed);

        vec4 u = ldexp(vec4(rand32), ivec4(-32));
        vec3 tdir = vec3(0,0,1);
        bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
        float bsdf_pdf = 0.0f;
        ggx_bsdf_sample(u, tail_domain.tview, tail_domain.mat, tdir, lobes, bsdf_pdf);
        vec3 dir = tail_domain.tbn * tdir;

        if(bsdf_pdf == 0) bsdf_pdf = 1;

        if(bounce != 0)
        {
            path_throughput *= modulate_bsdf(tail_domain.mat, lobes);
#ifdef USE_PRIMARY_SAMPLE_SPACE
            path_throughput /= bsdf_pdf;
#endif
        }

        domain dst;
        reconnection_vertex vertex;
        float nee_pdf;
        bool bounces = generate_bsdf_vertex(rand32, dir, false, tail_domain, dst, vertex, nee_pdf);

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
#ifdef USE_PRIMARY_SAMPLE_SPACE
        if(tail_length > 1)
            path_throughput /= nee_pdf;
#endif
        radiance = path_throughput * vertex.radiance_estimate;

        if(tail_length <= 1) tail_dir = candidate_dir;
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

    resolved_vertex to;
    if(!resolve_reconnection_vertex(rs.vertex, !cur_to_prev, to_domain.pos, to))
    {
        // Failed to reconnect, vertex may have ceased to exist.
        return false;
    }

    vec3 reconnect_ray_dir;
    float reconnect_ray_dist;
    homogeneous_to_ray(to_domain.pos, to.pos, reconnect_ray_dir, reconnect_ray_dist);

    vec3 emission = to.emission;
    if(rs.tail_length >= 1)
    {
        // The reconnection vertex wasn't the last path vertex, so we need the
        // full material data too now.
        float nee_pdf;
        vertex_data vd = get_interpolated_vertex(
            reconnect_ray_dir,
            rs.vertex.hit_info,
            int(rs.vertex.instance_id),
            int(rs.vertex.primitive_id)
#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
            , to_domain.pos, nee_pdf
#endif
        );

        sampled_material mat = sample_material(int(rs.vertex.instance_id), vd);
        mat.roughness = max(mat.roughness, 0.001f);

        mat3 tbn = mat3(vd.tangent, vd.bitangent, vd.mapped_normal);
        vec3 tview = -reconnect_ray_dir * tbn;

        // Optionally, update radiance based on tail path in the timeframe of
        // to_domain. This is only needed for temporal reuse.
#if defined(RESTIR_TEMPORAL) && !defined(ASSUME_UNCHANGED_RECONNECTION_RADIANCE)
        if(!cur_to_prev)
        {
            domain tail_domain;
            tail_domain.mat = mat;
            tail_domain.pos = vd.pos;
            tail_domain.tbn = mat3(
                vd.tangent,
                vd.bitangent,
                vd.mapped_normal
            );
            tail_domain.flat_normal = vd.hard_normal;
            tail_domain.view = reconnect_ray_dir;
            tail_domain.tview = tview;
            update_tail_radiance(tail_domain, rs.nee_terminal, rs.tail_length, rs.tail_rng_seed, rs.vertex.radiance_estimate, rs.vertex.incident_direction);
        }
#endif

        vec3 incident_dir = rs.vertex.incident_direction * tbn;

        // Turn radiance into emission
        bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
        ggx_bsdf(incident_dir, tview, mat, lobes);
        vec3 bsdf = modulate_bsdf(mat, lobes);

        emission = bsdf * rs.vertex.radiance_estimate;
    }

    vec3 tdir = reconnect_ray_dir * to_domain.tbn;
    bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
    ggx_bsdf(tdir, to_domain.tview, to_domain.mat, lobes);

    contribution =
        get_bounce_throughput(0, to_domain, reconnect_ray_dir, lobes, primary_bsdf) * emission;

    if(do_visibility_test)
    {
#ifdef RESTIR_TEMPORAL
        contribution *= cur_to_prev ?
            test_prev_visibility(seed, to_domain.pos, reconnect_ray_dir, reconnect_ray_dist, to_domain.flat_normal) :
            test_visibility(seed, to_domain.pos, reconnect_ray_dir, reconnect_ray_dist, to_domain.flat_normal);
#else
        contribution *= test_visibility(seed, to_domain.pos, reconnect_ray_dir, reconnect_ray_dist, to_domain.flat_normal);
#endif
    }

    float half_jacobian = reconnection_shift_half_jacobian(to_domain.pos, to.pos, to.normal);

    jacobian = half_jacobian / rs.base_path_jacobian_part;
    if(isinf(jacobian) || isnan(jacobian)) jacobian = 0;

    if(update_sample) rs.base_path_jacobian_part = half_jacobian;
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
    int replay_status = replay_path(
        to_domain,
        rs.head_rng_seed,
        int(rs.head_length + 1 + rs.tail_length),
        rs.nee_terminal,
        false,
        false,
        cur_to_prev,
        throughput,
        primary_bsdf
    );
    if(replay_status == 2)
    {
        // Failed to replay path here, so contribution is zero.
        return false;
    }
    contribution = throughput * to_domain.mat.emission;
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
            throughput,
            primary_bsdf
        );

        if(!has_reconnection)
        { // This path has no reconnection so it's just a random replay path.
            if(replay_status == 2) // Failed to replay path
                return false;
            contribution = throughput * to_domain.mat.emission;
            return true;
        }
        else if(replay_status != 0) // Failed to replay path
            return false;
    }

    if(!allow_initial_reconnection(to_domain.mat))
        return false;

    resolved_vertex to;
    if(!resolve_reconnection_vertex(rs.vertex, !cur_to_prev, to_domain.pos, to))
    {
        // Failed to reconnect, vertex may have ceased to exist.
        return false;
    }

    vec3 reconnect_ray_dir;
    float reconnect_ray_dist;
    homogeneous_to_ray(to_domain.pos, to.pos, reconnect_ray_dir, reconnect_ray_dist);

    float v0_pdf = 1.0f;
    float v1_pdf = 1.0f;

    vec3 emission = to.emission;
    if(rs.tail_length >= 1)
    {
        // The reconnection vertex wasn't the last path vertex, so we need the
        // full material data too now.
        float nee_pdf;
        vertex_data vd = get_interpolated_vertex(
            reconnect_ray_dir,
            rs.vertex.hit_info,
            int(rs.vertex.instance_id),
            int(rs.vertex.primitive_id)
#ifdef NEE_SAMPLE_EMISSIVE_TRIANGLES
            , to_domain.pos, nee_pdf
#endif
        );

        sampled_material mat = sample_material(int(rs.vertex.instance_id), vd);
        mat3 tbn = mat3(vd.tangent, vd.bitangent, vd.mapped_normal);

        bool allowed = true;
        if(!allow_reconnection(reconnect_ray_dist, mat, true, allowed))
            return false;

        vec3 tview = -reconnect_ray_dir * tbn;

        // Optionally, update radiance based on tail path in the timeframe of
        // to_domain. This is only needed for temporal reuse.
#if defined(RESTIR_TEMPORAL) && !defined(ASSUME_UNCHANGED_RECONNECTION_RADIANCE)
        if(!cur_to_prev)
        {
            domain tail_domain;
            tail_domain.mat = mat;
            tail_domain.pos = vd.pos;
            tail_domain.tbn = mat3(
                vd.tangent,
                vd.bitangent,
                vd.mapped_normal
            );
            tail_domain.flat_normal = vd.hard_normal;
            tail_domain.view = reconnect_ray_dir;
            tail_domain.tview = tview;
            update_tail_radiance(tail_domain, rs.nee_terminal, rs.tail_length, rs.tail_rng_seed, rs.vertex.radiance_estimate, rs.vertex.incident_direction);
        }
#endif

        vec3 incident_dir = rs.vertex.incident_direction * tbn;

        // Turn radiance into to.emission
        bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
        v1_pdf = ggx_bsdf_pdf(incident_dir, tview, mat, lobes);
        vec3 bsdf = modulate_bsdf(mat, lobes);

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
        if(!allow_reconnection(reconnect_ray_dist, to_domain.mat, false, allowed))
            return false;
    }

    bsdf_lobes lobes = bsdf_lobes(0,0,0,0);
    vec3 tdir = reconnect_ray_dir * to_domain.tbn;
    v0_pdf = ggx_bsdf_pdf(tdir, to_domain.tview, to_domain.mat, lobes);
    contribution = throughput * get_bounce_throughput(rs.head_length, to_domain, reconnect_ray_dir, lobes, primary_bsdf) * emission;

    if(isnan(v0_pdf) || isinf(v0_pdf) || v0_pdf == 0)
        contribution = vec3(0);
    else contribution /= v0_pdf;

    if(do_visibility_test)
    {
#ifdef RESTIR_TEMPORAL
        contribution *= cur_to_prev && rs.head_length == 0 ?
            test_prev_visibility(seed, to_domain.pos, reconnect_ray_dir, reconnect_ray_dist, to_domain.flat_normal) :
            test_visibility(seed, to_domain.pos, reconnect_ray_dir, reconnect_ray_dist, to_domain.flat_normal);
#else
        contribution *= test_visibility(seed, to_domain.pos, reconnect_ray_dir, reconnect_ray_dist, to_domain.flat_normal);
#endif
    }

    float half_jacobian = hybrid_shift_half_jacobian(v0_pdf, v1_pdf, to_domain.pos, to.pos, to.normal);

    jacobian = half_jacobian / rs.base_path_jacobian_part;
    if(isinf(jacobian) || isnan(jacobian))
        jacobian = 0;

    if(update_sample)
        rs.base_path_jacobian_part = half_jacobian;
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
    vec2 proj_info;
};

spatial_candidate_sampler init_spatial_candidate_sampler(
    camera_data cam, vec3 pos, mat3 tbn, float max_plane_dist
){
    spatial_candidate_sampler scs;
    scs.view_pos = (cam.view * vec4(pos, 1)).xyz;
    scs.view_tangent = (mat3(cam.view) * tbn[0]) * max_plane_dist * 2.0f;
    scs.view_bitangent = (mat3(cam.view) * tbn[1]) * max_plane_dist * 2.0f;
    scs.proj_info = cam.projection_info.zw;
    scs.proj_info.y = -scs.proj_info.y;
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
    q.xy = (0.5 - q.xy / (scs.proj_info * q.z)) * size;
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
