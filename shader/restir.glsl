#ifndef RESTIR_GLSL
#define RESTIR_GLSL

#define CALC_PREV_VERTEX_POS

#include "rt.glsl"
#include "sampling.glsl"

struct light_candidate
{
    uint light_type;
    uint light_index;

    float sourcePDF_weight;
    float light_distance_square;

    vec3 evaluation;
    vec3 light_direction;
    vec2 light_data;
};

struct light_sample
{
    uint light_type;
    uint light_index;
    vec2 light_data;
};

struct reservoir
{
    light_sample ls;
    vec3 contribution;
    float target_function;
    float weight_sum;
    float uc_weight;
    float confidence;
};

struct domain
{
    sampled_material mat;
    vec3 pos;
    vec3 view;
    mat3 tbn;
    vec3 flat_normal;
};

layout(binding = 29, set = 0) uniform parity_data_buffer
{
    int parity;
} parity_data;

layout(binding = 30, set = 0, rgba32f) uniform image2DArray reservoir_data;
layout(binding = 31, set = 0, rg16) uniform image2DArray light_data_uni;
layout(binding = 32, set = 0, rg16_snorm) uniform image2DArray previous_normal_data;
layout(binding = 33, set = 0, rgba32f) uniform image2DArray previous_pos_data;

struct hit_payload
{
    uint random_seed;
    int instance_id;
    int primitive_id;
    vec2 barycentrics;
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
        gl_RayFlagsTerminateOnFirstHitEXT,
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

const uint NULL_SAMPLE_INDEX = ((1u<<30)-1);
bool is_null_sample(light_sample ls)
{
    return ls.light_type == 0 && ls.light_index == NULL_SAMPLE_INDEX;
}

domain get_domain(ivec3 p)
{
    domain dom;
    dom.mat = unpack_gbuffer_material(
        read_gbuffer_material(p),
        read_gbuffer_albedo(p),
        read_gbuffer_emission(p)
    );
    dom.pos = read_gbuffer_pos(p);
    dom.view = normalize(dom.pos - get_camera().origin.xyz);
    dom.flat_normal = read_gbuffer_flat_normal(p);
    dom.tbn = create_tangent_space(read_gbuffer_normal(p));
    return dom;
}

reservoir create_reservoir()
{
    reservoir r;

    r.ls.light_type = 0;
    r.ls.light_index = NULL_SAMPLE_INDEX;
    r.ls.light_data = vec2(0);

    r.contribution = vec3(0.0f);
    r.target_function = 0.0f;
    r.weight_sum = 0.0f;
    r.uc_weight = 0.0f;
    r.confidence = 0.0f;

    return r;
}

bool update_reservoir(
    inout reservoir r,
    light_sample ls,
    float weight,
    float u
){
    r.weight_sum += weight;
    if (u * r.weight_sum < weight)
    {
        r.ls = ls;
        return true;
    }
    return false;
}

float calculate_uc_weight(reservoir r)
{
    return r.weight_sum > 0.0f ? r.weight_sum / r.target_function : 0.0f;
}

ivec2 get_reprojected_pixel(domain dom, inout local_sampler lsampler)
{
    vec2 motion = read_gbuffer_screen_motion(ivec3(gl_LaunchIDEXT.xyz));

    motion.y = 1.f - motion.y;
    motion = motion * vec2(gl_LaunchSizeEXT.xy) - vec2(0.5);

    ivec2 imotion = ivec2(motion);
    vec2 off = motion - vec2(imotion);
    vec4 rs = generate_uniform_random(lsampler.rs);

    return imotion + ivec2(
        off.x < rs.x ? 0 : 1,
        off.y < rs.y ? 0 : 1
    );
}

vec3 light_contribution(
    light_sample ls,
    domain dom,
    out vec3 dir,
    out float dist2,
    out vec3 normal
){
    vec3 light_color = vec3(0.0f);

    if(is_null_sample(ls)) // null light
    {
        dir = vec3(0);
        dist2 = 0.0f;
        normal = vec3(0);
        return vec3(0.0f);
    }
    else if(ls.light_type == 0) // point light
    {
        point_light pl = point_lights.lights[ls.light_index];
        vec3 to_light = pl.pos - dom.pos;
        dir = normalize(to_light);
        normal = vec3(0);
        dist2 = dot(to_light, to_light);
        light_color = pl.color * get_spotlight_intensity(pl, dir);
        light_color /= (max(dist2, 1e-7));

    }
    else if(ls.light_type == 1) // triangle light
    {
        tri_light tl = tri_lights.lights[ls.light_index];

        vec3 A = tl.pos[0]-dom.pos;
        vec3 B = tl.pos[1]-dom.pos;
        vec3 C = tl.pos[2]-dom.pos;

        vec3 b = vec3(ls.light_data, 1.0f - ls.light_data.x - ls.light_data.y);
        vec3 pos = b.x * A + b.y * B + b.z * C;

        dir = normalize(pos);
        dist2 = dot(pos, pos);
        normal = normalize(cross(tl.pos[2]-tl.pos[0], tl.pos[1]-tl.pos[0]));

        vec3 color = tl.emission_factor;

        if(abs(dot(dir, normal)) < 0.001)
            color = vec3(0);

        if(tl.emission_tex_id >= 0)
        { // Textured emissive triangle, so read texture.
            vec2 uv =
                b.x * unpackHalf2x16(tl.uv[0]) +
                b.y * unpackHalf2x16(tl.uv[1]) +
                b.z * unpackHalf2x16(tl.uv[2]);
            color *= texture(textures[nonuniformEXT(tl.emission_tex_id)], uv).rgb;
        }

        light_color = color;
    }
    else if(ls.light_type == 2) // directional light
    {
        directional_light dl = directional_lights.lights[ls.light_index];
        dist2 = RAY_MAX_DIST * RAY_MAX_DIST;
        dir = sample_cone(ls.light_data, -dl.dir, dl.dir_cutoff);
        float pdf = dl.dir_cutoff >= 1.0f ? -1.0f : 1.0f / (2.0f * M_PI * (1.0f - dl.dir_cutoff));
        light_color = pdf > 0 ? dl.color * pdf : dl.color;
    }
    else if(ls.light_type == 3) // environment map
    {
        vec2 uv = ls.light_data;
        dir = uv_to_latlong_direction(uv);
        dist2 = RAY_MAX_DIST*RAY_MAX_DIST;
        light_color = scene_metadata.environment_factor.rgb;

        light_color *= texture(environment_map_tex, vec2(uv.x, uv.y)).rgb;
    }

    if(dot(dir, dom.flat_normal) < 1e-6f)
        return vec3(0.0f);

    vec3 shading_view = -dom.view * dom.tbn;
    vec3 shading_light = dir * dom.tbn;

    if(shading_view.z < 0.00001f)
        shading_view = vec3(shading_view.xy, max(shading_view.z, 0.00001f));
    shading_view = normalize(shading_view);

    vec3 d, s = vec3(0.0f);
    ggx_brdf(shading_light, shading_view, dom.mat, d, s);

    vec3 value = (d * dom.mat.albedo.rgb + s) * light_color;
    if(any(isnan(ls.light_data)))
        value = vec3(0);

    return value;
}

void sample_canonical(
    uvec4 random,
    domain dom,
    out uint light_type,
    out uint light_index,
    out vec2 light_data,
    out float light_pdf
){
    vec4 data = vec4(0.0f);
    vec4 u = ldexp(vec4(random), ivec4(-32));

    float point_prob, triangle_prob, dir_prob, envmap_prob;
    get_nee_sampling_probabilities(point_prob, triangle_prob, dir_prob, envmap_prob);

    if((u.w -= point_prob) < 0) // point light
    {
        light_type = 0;
        const int item_count = int(scene_metadata.point_light_count);
        light_index = clamp(int(u.x * item_count), 0, item_count-1);
        float selection_pmf = 1.0f / item_count;

        light_data = vec2(0,0);

        if(selection_pmf == 0)
            light_index = NULL_SAMPLE_INDEX;
        light_pdf = selection_pmf * point_prob;
    }
    else if((u.w -= triangle_prob) < 0) // triangle light
    {
        light_type = 1;
        int light_count = int(scene_metadata.tri_light_count);

        light_index = clamp(int(u.z*light_count), 0, light_count-1);
        float selection_pmf = 1.0f / light_count;

        if(selection_pmf == 0)
        {
            light_type = 0;
            light_index = NULL_SAMPLE_INDEX;
        }
        else
        {
            tri_light tl = tri_lights.lights[light_index];
            vec3 A = tl.pos[0]-dom.pos;
            vec3 B = tl.pos[1]-dom.pos;
            vec3 C = tl.pos[2]-dom.pos;

            float tri_pdf = 0.0f;
            vec3 out_dir = sample_triangle_light(u.xy, A, B, C, tri_pdf);
            float out_length = ray_plane_intersection_dist(out_dir, A, B, C);

            if(isinf(tri_pdf) || tri_pdf <= 0 || out_length <= control.min_ray_dist || any(isnan(out_dir)))
            { // Same triangle, trying to intersect itself... Or zero-area degenerate triangle.
                tri_pdf = 1e9;
            }

            light_data = get_barycentric_coords(out_dir*out_length, A, B, C).xy;
            light_pdf = tri_pdf * selection_pmf;
            light_pdf *= triangle_prob;
        }
  }
  else if((u.w -= dir_prob) < 0) // directional light
  {
      light_type = 2;
      const int light_count = int(scene_metadata.directional_light_count);
      light_index = clamp(int(u.z*light_count), 0, light_count-1);

      directional_light dl = directional_lights.lights[light_index];
      light_pdf = dl.dir_cutoff >= 1.0f ? -1.0f : 1.0f / (2.0f * M_PI * (1.0f - dl.dir_cutoff));
      light_pdf /= light_count;
      light_data = u.xy;

      light_pdf *= dir_prob;
    }
    else if((u.w -= envmap_prob) < 0) // environment map
    {
        light_type = 3;
        if(scene_metadata.environment_proj >= 0)
        {
            uvec2 size = textureSize(environment_map_tex, 0).xy;
            const uint pixel_count = size.x * size.y;
            uvec2 ip = clamp(random.xy / (0xFFFFFFFFu / size), uvec2(0), size-1u);
            int i = int(ip.x + ip.y * size.x);
            alias_table_entry at = environment_map_alias_table.entries[i];
            light_pdf = at.pdf;
            if(random.z > at.probability)
            {
                i = int(at.alias_id);
                light_pdf = at.alias_pdf;
            }

            ivec2 p = ivec2(i % size.x, i / size.x);
            vec2 off = ldexp(vec2(uvec2(random.xy*pixel_count)), ivec2(-32));
            light_data = (vec2(p) + off)/vec2(size);
        }
        else
        {
            light_data = vec2(0.0f);
            light_pdf = 1.0f;
        }

        light_pdf *= envmap_prob;
    }
}

vec3 get_previous_normal(ivec2 pixpos, int id)
{
    vec3 data = vec3(0.0f);

    if(parity_data.parity == -1)
    {
        return data;
    }

    data = unpack_gbuffer_normal(imageLoad(previous_normal_data, ivec3(pixpos, id)).xy);
    return data;
}

vec3 get_previous_position(ivec2 pixpos, int id)
{
    return parity_data.parity < 0 ? vec3(0.0f) :
        imageLoad(previous_pos_data, ivec3(pixpos, id)).xyz;
}

bool temporal_edge_detection(domain dom, ivec2 pixpos)
{
    vec3 current_normal = dom.tbn[2];
    vec3 prev_normal = get_previous_normal(pixpos, int(parity_data.parity == 1)).xyz;
    vec3 previous_pos = get_previous_position(pixpos, int(parity_data.parity == 1));

    vec3 camera_origin = get_camera().origin.xyz;

    float delta_pos = distance(dom.pos, previous_pos);
    float max_range = 0.01 * distance(camera_origin, dom.pos) /
        (abs(dot(normalize(dom.pos - camera_origin), current_normal)) + 0.001f);

    return dot(prev_normal, current_normal) < 0.95 || delta_pos > max_range;
}

// https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22699-fast-denoising-with-self-stabilizing-recurrent-blurs.pdf
float edge_detect(vec3 normal1, vec3 pos1, vec3 pos2, float inv_max_plane_dist)
{
    return clamp(1.0f - abs(dot(normal1, pos2-pos1)) * inv_max_plane_dist, 0.0f, 1.0f);
}

// Used for edge detection algorithms. Inverse size of frustum at the depth of
// a given point.
float get_inv_max_plane_dist(camera_data cam, vec3 pos)
{
    float frustum_size = min(abs(cam.proj_inverse[0][0] * 2.0f), abs(cam.proj_inverse[1][1] * 2.0f));
    frustum_size *= abs(dot(cam.view_inverse[2].xyz, cam.view_inverse[3].xyz - pos));
    return 1.0f / frustum_size;
}

bool in_screen(ivec2 pos)
{
    return all(lessThan(pos.xy, ivec2(get_screen_size().xy))) &&
           all(greaterThanEqual(pos.xy, ivec2(0.0f)));
}

void write_reservoir_buffer(reservoir r, ivec2 pos, int id)
{
    vec4 r_data = vec4(0.0f);

    r_data.x = r.uc_weight;
    r_data.y = r.target_function;
    r_data.z = uintBitsToFloat(r.ls.light_type << 30 | r.ls.light_index);
    r_data.w = r.confidence;

    //store reservoir data
    imageStore(reservoir_data, ivec3(pos, id), r_data);

    //store light data
    imageStore(light_data_uni, ivec3(pos, id), vec4(r.ls.light_data, 0, 0));
}

reservoir read_reservoir_buffer(ivec2 pos, int id)
{
    reservoir r = create_reservoir();
    if(pos.x >= 0 && pos.y >= 0)
    {
        vec4 r_data = imageLoad(reservoir_data, ivec3(pos, id));
        vec4 l_data = imageLoad(light_data_uni, ivec3(pos, id));

        r.uc_weight       = r_data.x;
        r.target_function = r_data.y;
        r.ls.light_type   = floatBitsToUint(r_data.z) >> 30;
        r.ls.light_index  = floatBitsToUint(r_data.z) & 0x3FFFFFFF;
        r.confidence      = r_data.w;

        r.ls.light_data = l_data.xy;
    }
    else
    {
        r.uc_weight       = 0;
        r.target_function = 0;
        r.ls.light_type   = 0;
        r.ls.light_index  = NULL_SAMPLE_INDEX;
        r.ls.light_data   = vec2(0);
        r.confidence      = 0;
    }

    return r;
}

bool is_first_frame()
{
    return parity_data.parity < 0;
}

void write_previous_normal_buffer(vec3 normal, ivec2 pixpos)
{
    vec4 data = vec4(pack_gbuffer_normal(normalize(normal)), 0, 0);

    imageStore(previous_normal_data, ivec3(pixpos,
        (is_first_frame() ? false : parity_data.parity < 1)), data);
}

void write_previous_position_buffer(vec3 pos, ivec2 pixpos)
{
    vec4 data = vec4(pos, 0);

    imageStore(previous_pos_data, ivec3(pixpos,
        (is_first_frame() ? false : parity_data.parity < 1)), data);
}

float reconnection_jacobian(
    vec3 prev_src,
    vec3 prev_dst,
    vec3 prev_dst_normal,
    vec3 cur_src,
    vec3 cur_dst,
    vec3 cur_dst_normal
){
    vec3 qdelta = prev_src-prev_dst;
    vec3 rdelta = cur_src-cur_dst;

    float qdelta_dist = distance(prev_src, prev_dst);
    float rdelta_dist = distance(cur_src, cur_dst);

    float theta_q = abs(dot(prev_dst_normal, qdelta));
    float theta_r = abs(dot(cur_dst_normal, rdelta));
    if(all(equal(prev_dst_normal, vec3(0))))
        return 1;

    float v = (theta_r * qdelta_dist * qdelta_dist * qdelta_dist) / (theta_q * rdelta_dist * rdelta_dist * rdelta_dist);

    if(isinf(v) || isnan(v)) v = 0.0f;

    return v;
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

#endif
#endif
