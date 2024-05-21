#ifndef SHADOW_MAPPING_GLSL
#define SHADOW_MAPPING_GLSL
#include "scene_raster.glsl"
#include "projection.glsl"
#include "poisson_samples_2d.glsl"

#ifndef SHADOW_MAPPING_PARAMS
#define SHADOW_MAPPING_PARAMS control.sm_params
#endif

#ifndef SHADOW_MAPPING_SCREEN_COORD
#define SHADOW_MAPPING_SCREEN_COORD gl_FragCoord.xy
#endif

// Does bilinear interpolation, handling the atlas and clamping edges properly.
float shadow_map_bilinear_sample(vec4 rect, vec2 uv, float depth)
{
    vec2 min_uv = rect.xy + SHADOW_MAPPING_PARAMS.shadow_map_atlas_pixel_margin;
    vec2 max_uv = rect.xy + rect.zw - SHADOW_MAPPING_PARAMS.shadow_map_atlas_pixel_margin;
    vec2 read_uv = clamp(uv * rect.zw + rect.xy, min_uv, max_uv);
    return texture(shadow_map_atlas_test, vec3(read_uv.x, 1.0f-read_uv.y, depth));
}

void calc_omni_shadow_map_pos(
    in shadow_map sm,
    vec3 p,
    out vec2 uv,
    out float depth,
    out vec4 rect
){
    vec3 ap = abs(p);
    float max_axis = max(max(ap.x, ap.y), ap.z);

    vec2 face_offset = vec2(0);
    vec3 face_pos = p;

    if(max_axis == ap.x) // +-X is depth
    {
        face_offset.x = 0;
        face_pos = face_pos.zyx;
    }
    else if(max_axis == ap.y) // +-Y is depth
    {
        face_offset.x = sm.rect.z;
        face_pos = vec3(face_pos.x, -face_pos.z, -face_pos.y);
    }
    else // +-Z is depth
    {
        face_offset.x = 2*sm.rect.z;
        face_pos = vec3(-face_pos.x, face_pos.yz);
    }

    // Check sign
    if(face_pos.z < 0)
    {
        face_pos.xz = -face_pos.xz;
        face_offset.y = sm.rect.w;
    }

    // Projection is simple due to forced 90 degree FOV and 1:1 aspect ratio
    uv = 0.5f + 0.5f * face_pos.xy / face_pos.z;

    depth = hyperbolic_depth(min(-face_pos.z, 0.0f), sm.projection_info) * 0.5f + 0.5f;
    rect = vec4(sm.rect.xy+face_offset, sm.rect.zw);
}

bool find_cascade(
    in shadow_map sm,
    inout vec3 p,
    out vec4 rect,
    out vec2 radius,
    float bias
){
    rect = sm.rect;
    radius = SHADOW_MAPPING_PARAMS.pcf_samples <= 0 ? vec2(0) : sm.range_radius.zw;

    vec3 q = p.xyz;
    q.z -= bias;

    if(abs(q.z) > 1.0f) return false;

    // If out of range, try cascades.
    if(any(greaterThanEqual(abs(q.xy), vec2(1.0f)-2.0f*radius)))
    {
        int begin_cascade = sm.cascade_index;
        int end_cascade = begin_cascade + sm.type;
        for(; begin_cascade < end_cascade; ++begin_cascade)
        {
            shadow_map_cascade c = shadow_map_cascades.cascades[begin_cascade];
            vec2 nq = q.xy * c.offset_scale.z + c.offset_scale.xy;
            vec2 new_radius = radius * c.offset_scale.z;
            if(all(lessThan(abs(nq.xy), vec2(1.0f)-2.0f*new_radius)))
            {
                p.xy = nq;
                rect = c.rect;
                p.z -= bias * c.offset_scale.w;
                radius = new_radius;
                return true;
            }
        }
        // Didn't find any suitable cascade.
        return false;
    }
    else
    {
        p = q;
        return true;
    }
}

void calc_perspective_shadow_map_pos(
    in shadow_map sm,
    vec3 p,
    float ndotl,
    out vec2 uv,
    out float depth
){
    // We do the projection manually so that bias can easily be applied to
    // linear depth.
    float bias = max(sm.max_bias * (1.0f - ndotl), sm.min_bias);
    uv = project_position(p, sm.projection_info, vec2(0));
    depth = hyperbolic_depth((1-bias)*p.z, sm.projection_info) * 0.5f + 0.5f;
}

bool calc_perspective_pcss_radius(
    shadow_map sm,
    vec2 uv,
    float depth,
    mat2 rotation,
    inout vec2 radius
){
    float near = -sm.range_radius.x;
    float linear_depth = linearize_depth(depth, sm.projection_info);
    float occluders = 0.0f;
    float avg_z = 0.0f;
    vec2 search_radius = -radius/linear_depth;

    vec2 min_uv = sm.rect.xy+SHADOW_MAPPING_PARAMS.shadow_map_atlas_pixel_margin;
    vec2 max_uv = sm.rect.xy+sm.rect.zw-SHADOW_MAPPING_PARAMS.shadow_map_atlas_pixel_margin;

    for(int i = 0; i < SHADOW_MAPPING_PARAMS.pcss_samples; ++i)
    {
        vec2 o = rotation * poisson_disk_samples[i] * search_radius;
        vec2 read_uv = clamp(
            (uv + o) * sm.rect.zw + sm.rect.xy, min_uv, max_uv
        );
        vec4 v = linearize_depth(
            textureGather(shadow_map_atlas, vec2(read_uv.x, 1.0f-read_uv.y)),
            sm.projection_info
        );
        vec4 mask = vec4(greaterThan(v, vec4(linear_depth)));

        occluders += mask.x+mask.y+mask.z+mask.w;
        avg_z += dot(v, mask);
    }

    float penumbra = (linear_depth * max(occluders, 1.0f) - avg_z)/avg_z;
    radius = search_radius * (penumbra + SHADOW_MAPPING_PARAMS.pcss_minimum_radius);

    // Signal caller that we can give up if there were no blockers.
    return occluders >= 1.0f;
}

bool calc_directional_pcss_radius(
    shadow_map sm,
    vec4 rect,
    vec2 uv,
    float depth,
    mat2 rotation,
    inout vec2 radius
){
    float occluders = 0.0f;
    float avg_z = 0.0f;
    vec2 min_uv = rect.xy+SHADOW_MAPPING_PARAMS.shadow_map_atlas_pixel_margin;
    vec2 max_uv = rect.xy+rect.zw-SHADOW_MAPPING_PARAMS.shadow_map_atlas_pixel_margin;
    float z_range = abs(sm.range_radius.y - sm.range_radius.x);
    vec2 search_radius = radius * z_range * (1.0f/20.0f);

    for(int i = 0; i < SHADOW_MAPPING_PARAMS.pcss_samples; ++i)
    {
        vec2 o = rotation * poisson_disk_samples[i] * search_radius;
        vec2 read_uv = clamp(
            (uv + o) * rect.zw + rect.xy, min_uv, max_uv
        );
        vec4 v = textureGather(
            shadow_map_atlas, vec2(read_uv.x, 1.0f-read_uv.y)
        );
        vec4 mask = vec4(lessThan(v, vec4(depth)));

        occluders += mask.x+mask.y+mask.z+mask.w;
        avg_z += dot(v, mask);
    }
    // Instead of dividing avg_z like we should, we can just multiply depth to
    // achieve the same effect.
    float penumbra = (depth * max(occluders, 1.0f) - avg_z)/avg_z;
    radius = 5.0f * search_radius * penumbra + SHADOW_MAPPING_PARAMS.pcss_minimum_radius;
    // Signal caller that we can give up if there were no blockers.
    return occluders >= 1.0f;
}

float pcf_2d_perspective(shadow_map sm, vec2 uv, vec2 radius, float depth)
{
    ivec2 noise_pos = ivec2(mod(
        SHADOW_MAPPING_SCREEN_COORD * SHADOW_MAPPING_PARAMS.noise_scale,
        vec2(textureSize(pcf_noise_vector_2d, 0))
    ));
    vec2 cs = texelFetch(pcf_noise_vector_2d, noise_pos, 0).xy;
    mat2 rotation = mat2(cs.x, cs.y, -cs.y, cs.x);

    if(SHADOW_MAPPING_PARAMS.pcss_samples > 0)
    {
        if(!calc_perspective_pcss_radius(sm, uv, depth, rotation, radius))
            return 1.0f;
    }

    float shadow = 0.0f;

    for(int i = 0; i < SHADOW_MAPPING_PARAMS.pcf_samples; ++i)
    {
        vec2 o = rotation * poisson_disk_samples[i] * radius;
        shadow += shadow_map_bilinear_sample(sm.rect, uv + o, depth);
    }
    return shadow/SHADOW_MAPPING_PARAMS.pcf_samples;
}

float pcf_2d_directional(
    shadow_map sm, vec4 rect, vec2 uv, vec2 radius, float depth
){
    ivec2 noise_pos = ivec2(mod(
        SHADOW_MAPPING_SCREEN_COORD * SHADOW_MAPPING_PARAMS.noise_scale,
        vec2(textureSize(pcf_noise_vector_2d, 0))
    ));
    vec2 cs = texelFetch(pcf_noise_vector_2d, noise_pos, 0).xy;
    mat2 rotation = mat2(cs.x, cs.y, -cs.y, cs.x);

    if(SHADOW_MAPPING_PARAMS.pcss_samples > 0)
    {
        if(!calc_directional_pcss_radius(sm, rect, uv, depth, rotation, radius))
            return 1.0f;
    }

    float shadow = 0.0f;

    for(int i = 0; i < SHADOW_MAPPING_PARAMS.pcf_samples; ++i)
    {
        vec2 o = rotation * poisson_disk_samples[i] * radius;
        shadow += shadow_map_bilinear_sample(rect, uv + o, depth);
    }
    return shadow/SHADOW_MAPPING_PARAMS.pcf_samples;
}

float pcf_3d(shadow_map sm, vec3 p, float radius, float bias)
{
    ivec2 noise_pos = ivec2(mod(
        SHADOW_MAPPING_SCREEN_COORD * SHADOW_MAPPING_PARAMS.noise_scale,
        vec2(textureSize(pcf_noise_vector_3d, 0))
    ));
    vec3 random_vec = texelFetch(pcf_noise_vector_3d, noise_pos, 0).xyz;

    vec3 ndir = normalize(p);
    vec3 tangent = normalize(random_vec - ndir * dot(random_vec, ndir));
    mat2x3 tangent_space = mat2x3(tangent, cross(ndir, tangent));
    float len = max(length(p)*(1.0f-bias), sm.range_radius.x);

    float shadow = 0.0f;

    for(int i = 0; i < SHADOW_MAPPING_PARAMS.omni_pcf_samples; ++i)
    {
        vec3 sample_offset = tangent_space * poisson_disk_samples[i] * radius;
        vec2 uv; float depth; vec4 rect;
        calc_omni_shadow_map_pos(
            sm, normalize(ndir+sample_offset)*len, uv, depth, rect
        );
        shadow += shadow_map_bilinear_sample(rect, uv, depth);
    }
    return shadow/SHADOW_MAPPING_PARAMS.omni_pcf_samples;
}

float calc_directional_shadow(
    int shadow_map_index,
    vec3 pos,
    vec3 normal,
    vec3 dir
){
    shadow_map sm = shadow_maps.maps[shadow_map_index];

    vec4 p = sm.world_to_shadow * vec4(pos, 1.0f);
    p.z = p.z * 2.0f - 1.0f;
    float ndotl = abs(dot(normal, dir));
    float bias = max(sm.max_bias * (1.0f - ndotl), sm.min_bias);

    vec4 rect;
    vec2 radius;
    if(find_cascade(sm, p.xyz, rect, radius, bias))
    {
        p.xyz = p.xyz * 0.5f + 0.5f;

        if(SHADOW_MAPPING_PARAMS.pcf_samples <= 0)
            return shadow_map_bilinear_sample(rect, p.xy, p.z);
        else return pcf_2d_directional(sm, rect, p.xy, radius, p.z);
    }
    else return 1.0f;
}

float calc_point_shadow(
    int shadow_map_index,
    vec3 pos,
    vec3 normal,
    vec3 dir
){
    shadow_map sm = shadow_maps.maps[shadow_map_index];

    vec4 p = sm.world_to_shadow * vec4(pos, 1.0f);
    float ndotl = abs(dot(normal, dir));
    float bias = max(sm.max_bias * (1.0f - ndotl), sm.min_bias);

    // Perspective shadow map.
    if(sm.type == 0)
    {
        vec2 uv;
        float depth;
        calc_perspective_shadow_map_pos(sm, p.xyz, bias, uv, depth);
        if(SHADOW_MAPPING_PARAMS.pcf_samples <= 0)
            return shadow_map_bilinear_sample(sm.rect, uv, depth);
        else return pcf_2d_perspective(
            sm, uv, sm.range_radius.zw, depth
        );
    }
    // Omni shadow map.
    else
    {
        if(SHADOW_MAPPING_PARAMS.omni_pcf_samples <= 0)
        {
            vec2 uv; float depth; vec4 rect;
            calc_omni_shadow_map_pos(
                sm, p.xyz-normalize(p.xyz)*bias, uv, depth, rect
            );
            return shadow_map_bilinear_sample(rect, uv, depth);
        }
        // Not physically based in any way, that's why there are so many
        // Hervantafactors.
        else
        {
            float soft_radius = sm.range_radius.z*0.2f;
            float bias = max(
                sm.max_bias * (1.0f + 2e3 * soft_radius * soft_radius) * (1.0f - ndotl),
                sm.min_bias
            );
            return pcf_3d(sm, p.xyz, soft_radius, bias);
        }
    }
}

#endif
