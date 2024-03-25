#include "shadow_map.hh"
#include "camera.hh"
#include "scene_stage.hh"

namespace
{
using namespace tr;

bool find_optimal_cascade_placement(
    ray* frustum_rays, // Should be 4 long. Order doesn't matter.
    vec2 view_dir, // Used to maximize useful area.
    vec2 cascade_size,
    vec2& cascade_offset, // Offset from center of the cascade.
    bool centered
){
    // Stupid binary search at the moment. Since this should be done once per
    // frame, performance probably won't be an issue. But an analytical
    // algorithm would be nice to have still.
    float min_t = 0.0f;
    float max_t = 0.0f;
    bool found_valid = false;

    vec4 bb = vec4(0);
    bvec4 bb_edge_is_origin;

    for(int iterations = 0; iterations < 64; ++iterations)
    {
        float try_t;
        if(max_t == 0.0f)
        {
            if(min_t == 0.0f) try_t = 1.0f;
            else try_t = min_t * 16.0f;
        }
        else try_t = (min_t + max_t)*0.5f;

        // Find bounding box
        vec2 mi = frustum_rays[0].o;
        vec2 ma = frustum_rays[0].o;
        bvec2 min_is_origin(true);
        bvec2 max_is_origin(true);
        for(int i = 0; i < 4; ++i)
        {
            vec2 o = frustum_rays[i].o;
            vec2 e = o + vec2(frustum_rays[i].dir) * try_t;
            if(o.x < mi.x) { mi.x = o.x; min_is_origin.x = true; }
            if(o.y < mi.y) { mi.y = o.y; min_is_origin.y = true; }
            if(o.x > ma.x) { ma.x = o.x; max_is_origin.x = true; }
            if(o.y > ma.y) { ma.y = o.y; max_is_origin.y = true; }
            if(e.x < mi.x) { mi.x = e.x; min_is_origin.x = false; }
            if(e.y < mi.y) { mi.y = e.y; min_is_origin.y = false; }
            if(e.x > ma.x) { ma.x = e.x; max_is_origin.x = false; }
            if(e.y > ma.y) { ma.y = e.y; max_is_origin.y = false; }
        }

        // If can fit, use it.
        vec2 size = ma-mi;
        if(all(lessThanEqual(size, cascade_size)))
        {
            found_valid = true;
            min_t = try_t;
            bb = vec4(mi, ma);
            bb_edge_is_origin = bvec4(min_is_origin, max_is_origin);
        }
        else max_t = try_t;
    }

    // Calculate "return" values.
    if(found_valid)
    {
        if(centered)
        {
            cascade_offset = vec2(bb.x+bb.z, bb.y+bb.w)*0.5f;
        }
        else
        {
            vec2 cascade_min_offset = vec2(bb.x, bb.y) + cascade_size * 0.5f;
            vec2 cascade_max_offset = vec2(bb.z, bb.w) - cascade_size * 0.5f;

            if(abs(view_dir.y) > abs(view_dir.x)) view_dir /= abs(view_dir.y);
            else view_dir /= abs(view_dir.x);

            cascade_offset = mix(
                cascade_max_offset, cascade_min_offset, view_dir*0.5f+0.5f
            );
        }

        // Update origins.
        for(int i = 0; i < 4; ++i)
            frustum_rays[i].o += frustum_rays[i].dir * min_t;
    }

    return found_valid;
}

}

namespace tr
{

void directional_shadow_map::track_cameras(
    const mat4& light_transform,
    const std::vector<camera*>& cameras,
    const std::vector<transformable*>& camera_transforms,
    bool conservative
){
    if(cascades.size() == 0)
        cascades.push_back(vec2(0));

    if(cameras.size() == 1)
    {
        const camera& cam = *cameras[0];
        const transformable& t = *camera_transforms[0];
        mat4 cam_transform = t.get_global_transform();
        mat4 inv_light_transform = affineInverse(light_transform);
        mat4 cam_to_light_transform = inv_light_transform * cam_transform;

        vec2 view_dir = normalize(vec2(cam_to_light_transform * vec4(0,0,-1, 0)));
        ray frustum_rays[4] = {
            cam_to_light_transform * cam.get_view_ray(vec2(0, 0)),
            cam_to_light_transform * cam.get_view_ray(vec2(1, 0)),
            cam_to_light_transform * cam.get_view_ray(vec2(0, 1)),
            cam_to_light_transform * cam.get_view_ray(vec2(1, 1))
        };

        float scale = 1.0f;
        for(size_t i = 0; i < cascades.size(); ++i)
        {
            vec2 cascade_offset = vec2(0);
            vec2 cascade_size = abs(vec2(
                x_range.y-x_range.x,
                y_range.y-y_range.x
            )) * scale;

            find_optimal_cascade_placement(
                frustum_rays, view_dir, cascade_size, cascade_offset,
                i+1 == cascades.size() ? false : conservative
            );
            vec2 geom_center = cascade_size*0.5f;
            vec2 real_center = vec2(-x_range.x, -y_range.x) * scale;
            cascades[i] = cascade_offset - geom_center + real_center;

            scale *= 2.0f;
        }
    }
    else
    {
        vec2 cam_light_pos = vec2(0);

        for(const transformable* t: camera_transforms)
        {
            mat4 inv_light_transform = affineInverse(light_transform);
            vec2 pos = inv_light_transform * vec4(t->get_global_position(), 1.0f);
            cam_light_pos += pos;
        }
        cam_light_pos /= cameras.size();

        float scale = 1.0f;
        for(size_t i = 0; i < cascades.size(); ++i)
        {
            vec2 cascade_offset = cam_light_pos;
            vec2 cascade_size = abs(vec2(
                x_range.y-x_range.x,
                y_range.y-y_range.x
            )) * scale;

            vec2 geom_center = cascade_size*0.5f;
            vec2 real_center = vec2(-x_range.x, -y_range.x) * scale;
            cascades[i] = cascade_offset - geom_center + real_center;

            scale *= 2.0f;
        }
    }
}

gpu_shadow_mapping_parameters create_shadow_mapping_parameters(shadow_map_filter filter, scene_stage& ss)
{
    gpu_shadow_mapping_parameters params;
    params.shadow_map_atlas_pixel_margin = ss.get_shadow_map_atlas_pixel_margin();
    params.noise_scale = 1.0f;
    params.pcss_minimum_radius = filter.pcss_minimum_radius;
    params.pcf_samples = filter.pcf_samples;
    params.omni_pcf_samples = filter.omni_pcf_samples;
    params.pcss_samples = filter.pcss_samples;
    return params;
}

}
