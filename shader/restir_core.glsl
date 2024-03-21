#ifndef RESTIR_CORE_GLSL
#define RESTIR_CORE_GLSL

#ifndef TR_RESTIR
#define TR_RESTIR pc.config
#endif

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

#ifdef RESTIR_TEMPORAL
layout(binding = 14) uniform sampler2D prev_depth_or_position_tex;
layout(binding = 15) uniform sampler2D prev_normal_tex;
layout(binding = 16) uniform sampler2D prev_flat_normal_tex;
layout(binding = 17) uniform sampler2D prev_albedo_tex;
layout(binding = 18) uniform sampler2D prev_emission_tex;
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
    mat.roughness = max(mat.roughness, ZERO_ROUGHNESS_LIMIT);
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
    float linear_depth = linearize_depth(depth, cam.projection_info);

    vec2 uv = (vec2(p)+0.5f)/vec2(size);
    pos = unproject_position(depth, vec2(uv.x, 1-uv.y), cam.projection_info);
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
    mat.roughness = max(mat.roughness, ZERO_ROUGHNESS_LIMIT);
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
    float depth = texelFetch(prev_depth_or_position_tex, p, 0);
    float linear_depth = linearize_depth(depth, cam.projection_info);

    vec2 uv = (vec2(p)+0.5f)/vec2(size);
    pos = unproject_position(depth, vec2(uv.x, 1-uv.y), cam.projection_info);
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

    d.tbn = get_prev_tbn(p, d.flat_normal);
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
    r.output_sample.vertex.instance_id = reconnection_data[2];
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
    vec3 direct_diffuse;
    vec3 indirect_diffuse;
    vec3 reflection;
    float canonical_reflection;
};

sum_contribution init_sum_contribution(vec3 emission)
{
    sum_contribution sc;
#ifdef DEMODULATE_OUTPUT
    sc.direct_diffuse = vec3(0);
    sc.indirect_diffuse = vec3(0);
    sc.reflection = vec3(0);
    sc.canonical_reflection = 0;
#else
    sc.direct_diffuse = emission;
#endif
    return sc;
}

void add_contribution(inout sum_contribution sc, reservoir r, vec4 contrib, float weight)
{
#ifdef DEMODULATE_OUTPUT
    vec3 color = contrib.rgb * weight;
    vec3 diffuse_transmission = color * (1.0f - contrib.a);
    bool indirect = r.output_sample.head_length+r.output_sample.tail_length != 0;
    sc.direct_diffuse += indirect ? vec3(0) : diffuse_transmission;
    sc.indirect_diffuse += indirect ? diffuse_transmission : vec3(0);
    sc.reflection += color * contrib.a;
#else
    sc.direct_diffuse += contrib.rgb * weight;
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
        imageStore(out_direct_diffuse, p, vec4(sc.direct_diffuse, 0)); \
        imageStore(out_indirect_diffuse, p, vec4(sc.indirect_diffuse, sc.canonical_reflection > 0.5 ? 0 : ray_length)); \
        imageStore(out_reflection, p, vec4(sc.reflection, sc.canonical_reflection > 0.5 ? ray_length : 0)); \
    } \
    if(pc.update_sample_color != 0) \
        imageStore(out_indirect_diffuse, p, out_value); \
}
#else
#define finish_output_color(p, reservoir, out_value, sc, display_size) { \
    write_reservoir(reservoir, p, display_size); \
    if(pc.accumulate_color != 0) \
    { \
        vec4 prev_color = imageLoad(out_reflection, p); \
        prev_color.rgb *= prev_color.a; \
        if(pc.accumulated_samples == 0 && pc.initialize_output != 0) \
            prev_color = vec4(0); \
        float confidence_ratio = min(reservoir.confidence/TR_RESTIR.max_confidence, 1.0f); \
        vec4 ccolor = prev_color + vec4(sc.direct_diffuse, 1.0f) * confidence_ratio; \
        ccolor.rgb /= ccolor.a; \
        if(ccolor.a == 0) ccolor = vec4(0); \
        imageStore(out_reflection, p, ccolor); \
    } \
    if(pc.update_sample_color != 0) \
        imageStore(out_indirect_diffuse, p, out_value); \
}
#endif

#ifdef RAYBASE_RAY_TRACING
vec3 shade_explicit_lights(
    ivec4 environment,
    vec3 view,
    vertex_data vd,
    material mat
){
    vec3 normal = vd.tangent_space[2];
    vec3 tview = view_to_tangent_space(view, vd.tangent_space);

    vec3 contrib = vec3(0);
#ifdef SHADE_FAKE_INDIRECT
    contrib += get_indirect_light(
        vd.pos, environment, vd.tangent_space, vd.smooth_normal,
        tview, mat, vd.uv.zw
    );
#endif

    FOR_CULLED_POINT_LIGHTS(light, vd.pos)
        vec3 dir;
        vec3 color;
        if(get_point_light_info(light, vd.pos, dir, color))
        {
            vec3 tdir = dir * vd.tangent_space;
            bsdf_lobes lobes;
            full_bsdf(mat, tdir, tview, lobes);
            float shadow = calc_point_shadow(
                0, 0, 0, 0, 0, light, vd.pos, vd.smooth_normal, vec2(0), 0
            );
            contrib += shadow * color * modulate_bsdf(mat, lobes);
        }
    END_CULLED_POINT_LIGHTS

    for(uint i = 0; i < scene_params.directional_light_count; ++i)
    {
        directional_light light = directional_lights.array[i];
        vec3 dir;
        vec3 color;
        get_directional_light_info(light, dir, color);
        color *= light.solid_angle == 0 ? 1.0f : 2.0f * M_PI * light.solid_angle;
        float shadow = calc_directional_shadow(
            0, 0, 0, 0, 0, light, vd.pos, vd.smooth_normal, vec2(0), 0
        );
        vec3 tdir = dir * vd.tangent_space;
        bsdf_lobes lobes;
        full_bsdf(mat, tdir, tview, lobes);
        contrib += shadow * color * modulate_bsdf(mat, lobes);
    }
    return contrib;
}

int test_visibility(uint seed, vec3 pos, vec3 dir, float dist, vec3 flat_normal)
{
    payload.seed = seed;
    float s = sign(dot(dir, flat_normal));

    vec3 origin = pos + s * flat_normal * TR_RESTIR.min_ray_dist;
    vec3 target = pos + dir * dist;

    vec3 new_dir = dir * dist - s * flat_normal * TR_RESTIR.min_ray_dist;
    float new_dist = length(new_dir);
    new_dir /= new_dist;

    traceRayEXT(
        scene_tlas,
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
    return payload.instance_index < 0 ? 1 : 0;
}

#ifdef RESTIR_TEMPORAL
int test_prev_visibility(uint seed, vec3 pos, vec3 dir, float dist, vec3 flat_normal)
{
    payload.seed = seed;
    float s = sign(dot(dir, flat_normal));

    vec3 origin = pos + s * flat_normal * TR_RESTIR.min_ray_dist;
    vec3 target = pos + dir * dist;

    vec3 new_dir = dir * dist - s * flat_normal * TR_RESTIR.min_ray_dist;
    float new_dist = length(new_dir);
    new_dir /= new_dist;

#ifdef ASSUME_UNCHANGED_ACCELERATION_STRUCTURES
    traceRayEXT(
        scene_tlas,
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
    return payload.instance_index < 0 ? 1 : 0;
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
    if(prev) pl = prev_point_lights.array[rv.primitive_id];
    else pl = point_lights.array[rv.primitive_id];
#else
    point_light pl = point_lights.array[rv.primitive_id];
#endif
    v.pos = vec4(pl.pos_x, pl.pos_y, pl.pos_z, 1);

    vec3 center_dir = v.pos.xyz - domain_pos;
    float dist2 = dot(center_dir, center_dir);

    vec2 rc = unpackHalf2x16(pl.radius_and_cutoff_radius).xy;
    float radius = rc.r;

    if(radius != 0)
    {
        v.normal = octahedral_decode(rv.hit_info * 2.0f - 1.0f);
        v.pos.xyz += radius * v.normal;
    }
    else v.normal = vec3(0);

    v.emission = rgbe_to_rgb(pl.color);
    if(radius != 0.0f) v.emission /= M_PI * radius * radius;
    v.emission *= get_spotlight_cutoff(pl, normalize(center_dir));
    if(radius == 0) v.emission /= dist2;

    return v;
}

resolved_vertex resolve_directional_light_vertex(reconnection_vertex rv)
{
    resolved_vertex v;
    vec3 dir = octahedral_decode(rv.hit_info * 2.0f - 1.0f); 
    v.pos = vec4(dir, 0);
    v.normal = -dir;

#ifdef RESTIR_TEMPORAL
    vec3 env_dir = quat_rotate(scene_params.envmap_orientation, dir);
    v.emission = scene_params.envmap_index >= 0 ?
        textureLod(cube_textures[scene_params.envmap_index], env_dir, 0.0f).rgb :
        vec3(0);

#ifndef SHADE_ALL_EXPLICIT_LIGHTS
    // Directional lights don't generate vertices when they're rasterized.
    for(uint i = 0; i < scene_params.directional_light_count; ++i)
    {
        directional_light light = directional_lights.array[i];
        vec3 ddir;
        vec3 color;
        get_directional_light_info(light, ddir, color);
        float visible = step(1-dot(dir, ddir), max(light.solid_angle, 1e-6f));
        float pdf = sample_directional_light_pdf(light);
        v.emission += visible * color * (pdf == 0.0f ? 1.0f : pdf);
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
#ifdef RESTIR_TEMPORAL
    if(!rv_is_in_past)
    {
        get_vertex_prev_position(
            rv.instance_id,
            rv.primitive_id,
            rv.hit_info,
            to.pos.xyz,
            to.normal
        );
    }
    else
#endif
    {
        get_vertex_position(
            rv.instance_id,
            rv.primitive_id,
            rv.hit_info,
            to.pos.xyz,
            to.normal
        );
    }
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
        uint primitive_id;
        if(rv_is_in_past) primitive_id = point_light_map_forward.array[rv.primitive_id];
        else primitive_id = point_light_map_backward.array[rv.primitive_id];

        if(primitive_id == 0xFFFFFFFFu)
            return false;

        rv.primitive_id = primitive_id;
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
        if(rv_is_in_past)
            rv.instance_id = instance_map_forward.array[rv.instance_id];
        if(rv.instance_id == 0xFFFFFFFFu)
            return false;
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
    return mat.roughness > 0.05f;
#elif defined(USE_ADAPTIVE_HYBRID_SHIFT)
    // TODO: Better reconnection heuristics, but I'm not sure what's legal to
    // take into account...
    return mat.roughness > 0.05f;
#endif
}

bool allow_reconnection(
    float dist_scale,
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
#elif defined(USE_ADAPTIVE_HYBRID_SHIFT)
    bool head = as_head;
    as_head = !bounces ? true : mat.roughness > 0.05f;
    return head && as_head && (dist > dist_scale * TR_RESTIR.reconnection_scale || !bounces);
#endif
}

#ifdef RAYBASE_RAY_TRACING
bool get_intersection_info(
    vec3 ray_origin,
    vec3 ray_direction,
    bounce_payload payload,
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

    if(payload.instance_index >= 0)
    { // Mesh
#ifdef RESTIR_TEMPORAL
        if(in_past)
        {
            // Translate previous mesh to current frame
            uint new_id = instance_map_forward.array[payload.instance_index];
            if(new_id == 0xFFFFFFFFu)
            {
                candidate.instance_id = NULL_INSTANCE_ID;
                candidate.primitive_id = 0;
                candidate.hit_info = vec2(0);
                candidate.radiance_estimate = vec3(0);
                return false;
            }
            payload.instance_index = int(new_id);
        }
#endif

        instance i = instances.array[nonuniformEXT(payload.instance_index)];
        bool has_light = (i.attribute_mask_flags & INSTANCE_FLAG_HAS_TRI_LIGHTS) != 0;
        info.vd = get_vertex_data(
            payload.instance_index,
            payload.primitive_index,
            payload.hit_attribs,
            has_light
        );
#ifdef RESTIR_TEMPORAL
        if(in_past) info.vd.pos = info.vd.prev_pos;
#endif

        bool front_facing = dot(ray_direction, info.vd.flat_normal) < 0;
        info.mat = sample_material(i.material, front_facing, info.vd.uv.xy, vec2(0.0), vec2(0.0f));
#ifdef USE_RECONNECTION_SHIFT
        // See read_domain() for context.
        info.mat.roughness = max(info.mat.roughness, ZERO_ROUGHNESS_LIMIT);
#endif

        info.vd.tangent_space = apply_normal_map(info.vd.tangent_space, info.mat);
        info.local_pdf = has_light ?
            1.0f / triangle_solid_angle(
                ray_origin,
                info.vd.triangle_pos[0],
                info.vd.triangle_pos[1],
                info.vd.triangle_pos[2]
            ) : 0.0f;

        candidate.instance_id = payload.instance_index;
        candidate.primitive_id = payload.primitive_index;
        candidate.hit_info = payload.hit_attribs;
        candidate.radiance_estimate = info.mat.emission;

#ifdef SHADE_ALL_EXPLICIT_LIGHTS
        candidate.radiance_estimate += shade_explicit_lights(
            i.environment,
            ray_direction,
            info.vd,
            info.mat
        );
#endif

        return true;
    }
#ifndef SHADE_ALL_EXPLICIT_LIGHTS
    else if(payload.primitive_index >= 0)
    { // Point light
        // Point lights are unhittable with SHADE_ALL_EXPLICIT_LIGHTS, as ray
        // flags cull them.
        point_light pl;
#ifdef RESTIR_TEMPORAL
        if(in_past)
        {
            pl = prev_point_lights.array[payload.primitive_index];
            uint new_id = point_light_map_forward.array[payload.primitive_index];
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
#endif
        pl = point_lights.array[payload.primitive_index];

        vec2 radius_and_cutoff_radius = unpackHalf2x16(pl.radius_and_cutoff_radius);
        // No worries, radius cannot be zero -- we couldn't hit the light here
        // if it was!
        float radius = radius_and_cutoff_radius.x;

        vec3 pos = vec3(pl.pos_x, pl.pos_y, pl.pos_z);
        vec3 hit_pos = ray_origin + ray_direction * payload.hit_attribs.x;

        info.light = rgbe_to_rgb(pl.color) / (M_PI * radius * radius);
        info.local_pdf = sample_point_light_pdf(pl, ray_origin);
        info.vd.flat_normal = normalize(hit_pos-pos);
        info.vd.pos = hit_pos;

        candidate.instance_id = POINT_LIGHT_INSTANCE_ID;
        candidate.primitive_id = payload.primitive_index;
        candidate.hit_info = octahedral_encode(info.vd.flat_normal) * 0.5f + 0.5f;
        candidate.radiance_estimate = info.light;
        return false;
    }
#endif
    else
    { // Miss
        vec3 sample_dir = quat_rotate(scene_params.envmap_orientation, ray_direction);
        info.light = scene_params.envmap_index >= 0 ?
            textureLod(cube_textures[scene_params.envmap_index], sample_dir, 0.0f).rgb :
            vec3(0);

        info.envmap_pdf = scene_params.envmap_index >= 0 ? sample_environment_map_pdf(sample_dir) : 0.0f;

        info.local_pdf = 0;
#ifndef SHADE_ALL_EXPLICIT_LIGHTS
        // Directional lights don't generate vertices when they're rasterized.
        for(uint i = 0; i < scene_params.directional_light_count; ++i)
        {
            directional_light light = directional_lights.array[i];
            vec3 dir;
            vec3 color;
            get_directional_light_info(light, dir, color);
            float visible = step(1-dot(ray_direction, dir), light.solid_angle);
            float pdf = sample_directional_light_pdf(light);
            info.light += visible * color * (pdf == 0.0f ? 1.0f : pdf);
            info.local_pdf += visible * pdf;
        }
#endif

        // TODO: Not sure what to do here... This also receives directional
        // lights, which can and do overlap with the envmap.
        candidate.instance_id = ENVMAP_INSTANCE_ID;
        candidate.primitive_id = 0;
        candidate.hit_info = octahedral_encode(ray_direction) * 0.5f + 0.5f;
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
    payload.seed = seed;
    traceRayEXT(
        scene_tlas,
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
    payload.seed = seed;
#ifdef ASSUME_UNCHANGED_ACCELERATION_STRUCTURES
    traceRayEXT(
        scene_tlas,
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
        scene_prev_tlas,
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
        d.tbn[2],
        d.mat.transmission,
        TR_RESTIR.min_ray_dist,
        TR_RESTIR.max_ray_dist
    );

    if(ls.link.instance_id == POINT_LIGHT_INSTANCE_ID && ls.infinitesimal)
        ls.color /= ls.dist * ls.dist;

    vertex.instance_id = ls.link.instance_id;
    vertex.primitive_id = ls.link.primitive_id;
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
    light_link link;
    link.instance_id = vertex.instance_id;
    link.primitive_id = vertex.primitive_id;
    nee_pdf = calculate_light_pdf(
        link, info.local_pdf, info.envmap_pdf,
        cur_domain.pos, cur_domain.tbn[2],
        cur_domain.mat.transmission
    );

    next_domain.mat = info.mat;
    next_domain.mat.emission = vertex.radiance_estimate;
    next_domain.pos = info.vd.pos;
    next_domain.tbn = info.vd.tangent_space;
    next_domain.flat_normal = info.vd.flat_normal;
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
    float reconnect_scale,
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

    reconnected = allow_reconnection(reconnect_scale, candidate_dist, src.mat, false, head_allows_reconnection);
    
    rand32.w += 7u;
    vec3 tdir = candidate_dir * src.tbn;
    bsdf_lobes lobes;
    float bsdf_pdf = full_bsdf_pdf(src.mat, tdir, src.tview, lobes);

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
    float reconnect_scale,
    bool in_past,
    inout bool head_allows_reconnection,
    inout bool reconnected
){
    uvec4 rand32 = pcg1to4(path_seed);

    vec3 u = ldexp(vec3(rand32), ivec3(-32));
    vec3 tdir = vec3(0,0,1);
    int lobe_index;
    bsdf_lobes lobes;
    float bsdf_pdf = full_bsdf_sample(u, src.mat, src.tview, tdir, lobes, lobe_index);
    if(bsdf_pdf == 0) bsdf_pdf = 1;
    vec3 dir = src.tbn * tdir;

    path_throughput *= get_bounce_throughput(
        bounce_index, src, dir, lobes, primary_bsdf
    ) / bsdf_pdf;

    float nee_pdf;
    domain dst;
    reconnection_vertex vertex;
    bool bounces = generate_bsdf_vertex(rand32, dir, in_past, src, dst, vertex, nee_pdf);
    reconnected = allow_reconnection(reconnect_scale, distance(src.pos, dst.pos), dst.mat, bounces, head_allows_reconnection);
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
    float reconnect_scale,
    bool in_past,
    out vec3 throughput,
    out vec4 primary_bsdf
){
    throughput = vec3(1.0f);

    int end_state = 0;

    bool head_allows_reconnection = allow_initial_reconnection(src.mat);
    primary_bsdf = vec4(0);
    for(int bounce = 0; bounce < min(end_nee ? path_length-1 : path_length, TR_MAX_BOUNCES); ++bounce)
    {
        // Eat NEE sample
        if(!TR_RESTIR_DI)
            pcg1to4(seed);

        bool reconnected = false;
        bool bounces = replay_path_bsdf_bounce(bounce, seed, throughput, primary_bsdf, src, reconnect_scale, in_past, head_allows_reconnection, reconnected);
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
        bool success = replay_path_nee_leaf(path_length-1, seed, throughput, primary_bsdf, src, reconnect_scale, in_past, head_allows_reconnection, reconnected);
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

    for(int bounce = 0; bounce < min(end_nee ? tail_length-1 : tail_length, TR_MAX_BOUNCES); ++bounce)
    {
        // Eat NEE sample
        pcg1to4(tail_rng_seed);

        vec3 throughput;
        uvec4 rand32 = pcg1to4(tail_rng_seed);

        vec3 u = ldexp(vec3(rand32), ivec3(-32));
        vec3 tdir = vec3(0,0,1);
        int lobe_index;
        bsdf_lobes lobes;
        float bsdf_pdf = full_bsdf_sample(u, tail_domain.mat, tail_domain.tview, tdir, lobes, lobe_index);
        vec3 dir = tail_domain.tbn * tdir;

        if(bsdf_pdf == 0) bsdf_pdf = 1;

        if(bounce == 0) tail_dir = dir;
        else path_throughput *= modulate_bsdf(tail_domain.mat, lobes) / bsdf_pdf;

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

        path_throughput *= visibility / nee_pdf;
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
    float reconnect_scale,
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
        instance i = instances.array[nonuniformEXT(rs.vertex.instance_id)];
        vertex_data vd = get_vertex_data(
            rs.vertex.instance_id,
            rs.vertex.primitive_id,
            rs.vertex.hit_info,
            false
        );

        bool front_facing = dot(reconnect_ray_dir, vd.flat_normal) < 0;
        material mat = sample_material(i.material, front_facing, vd.uv.xy, vec2(0.0), vec2(0.0f));
        mat.roughness = max(mat.roughness, ZERO_ROUGHNESS_LIMIT);
        vd.tangent_space = apply_normal_map(vd.tangent_space, mat);

        vec3 tview = -reconnect_ray_dir * vd.tangent_space;

        // Optionally, update radiance based on tail path in the timeframe of
        // to_domain. This is only needed for temporal reuse.
#ifdef RESTIR_TEMPORAL
        if(!cur_to_prev && (TR_RESTIR_FLAGS & ASSUME_UNCHANGED_RECONNECTION_RADIANCE) == 0)
        {
            domain tail_domain;
            tail_domain.mat = mat;
            tail_domain.pos = vd.pos;
            tail_domain.tbn = vd.tangent_space;
            tail_domain.flat_normal = vd.flat_normal;
            tail_domain.view = reconnect_ray_dir;
            tail_domain.tview = tview;
            update_tail_radiance(tail_domain, rs.nee_terminal, rs.tail_length, rs.tail_rng_seed, rs.vertex.radiance_estimate, rs.vertex.incident_direction);
        }
#endif

        vec3 incident_dir = rs.vertex.incident_direction * vd.tangent_space;

        // Turn radiance into emission
        bsdf_lobes lobes;
        full_bsdf(mat, incident_dir, tview, lobes);
        vec3 bsdf = modulate_bsdf(mat, lobes);

        emission = bsdf * rs.vertex.radiance_estimate;
    }

    vec3 tdir = reconnect_ray_dir * to_domain.tbn;
    bsdf_lobes lobes;
    full_bsdf(to_domain.mat, tdir, to_domain.tview, lobes);

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
    float reconnect_scale,
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
        reconnect_scale,
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
    float reconnect_scale,
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
            reconnect_scale,
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
        instance i = instances.array[nonuniformEXT(rs.vertex.instance_id)];
        vertex_data vd = get_vertex_data(
            rs.vertex.instance_id,
            rs.vertex.primitive_id,
            rs.vertex.hit_info,
            false
        );

        bool front_facing = dot(reconnect_ray_dir, vd.flat_normal) < 0;
        material mat = sample_material(i.material, front_facing, vd.uv.xy, vec2(0.0), vec2(0.0f));
        vd.tangent_space = apply_normal_map(vd.tangent_space, mat);

        bool allowed = true;
        if(!allow_reconnection(reconnect_scale, reconnect_ray_dist, mat, true, allowed))
            return false;

        vec3 tview = -reconnect_ray_dir * vd.tangent_space;

        // Optionally, update radiance based on tail path in the timeframe of
        // to_domain. This is only needed for temporal reuse.
#if defined(RESTIR_TEMPORAL) && !defined(ASSUME_UNCHANGED_RECONNECTION_RADIANCE)
        if(!cur_to_prev)
        {
            domain tail_domain;
            tail_domain.mat = mat;
            tail_domain.pos = vd.pos;
            tail_domain.tbn = vd.tangent_space;
            tail_domain.flat_normal = vd.flat_normal;
            tail_domain.view = reconnect_ray_dir;
            tail_domain.tview = tview;
            update_tail_radiance(tail_domain, rs.nee_terminal, rs.tail_length, rs.tail_rng_seed, rs.vertex.radiance_estimate, rs.vertex.incident_direction);
        }
#endif

        vec3 incident_dir = rs.vertex.incident_direction * vd.tangent_space;

        // Turn radiance into to.emission
        bsdf_lobes lobes;
        v1_pdf = full_bsdf_pdf(mat, incident_dir, tview, lobes);
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
        if(!allow_reconnection(reconnect_scale, reconnect_ray_dist, to_domain.mat, false, allowed))
            return false;
    }

    bsdf_lobes lobes;
    vec3 tdir = reconnect_ray_dir * to_domain.tbn;
    v0_pdf = full_bsdf_pdf(to_domain.mat, tdir, to_domain.tview, lobes);
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
    float reconnect_scale,
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
        to_domain, do_visibility_test, update_sample, reconnect_scale,
        jacobian, contribution, primary_bsdf
    );
#elif defined(USE_RANDOM_REPLAY_SHIFT)
    return random_replay_shift_map(
        seed, rs,
#ifdef RESTIR_TEMPORAL
        cur_to_prev,
#endif
        to_domain, update_sample, reconnect_scale,
        jacobian, contribution, primary_bsdf
    );
#else
    return hybrid_shift_map(
        seed, rs,
#ifdef RESTIR_TEMPORAL
        cur_to_prev,
#endif
        to_domain, do_visibility_test, update_sample, reconnect_scale,
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

#endif
