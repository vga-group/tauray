#include "shadow_map_renderer.hh"
#include "context.hh"
#include "scene.hh"

namespace
{

using namespace tr;

const quat face_orientations[6] = {
    glm::quatLookAt(vec3(-1,0,0), vec3(0,1,0)),
    glm::quatLookAt(vec3(1,0,0), vec3(0,1,0)),
    glm::quatLookAt(vec3(0,-1,0), vec3(0,0,1)),
    glm::quatLookAt(vec3(0,1,0), vec3(0,0,1)),
    glm::quatLookAt(vec3(0,0,-1), vec3(0,1,0)),
    glm::quatLookAt(vec3(0,0,1), vec3(0,1,0))
};

const uvec2 face_offset_mul[6] = {
    uvec2(0,0), uvec2(0,1),
    uvec2(1,0), uvec2(1,1),
    uvec2(2,0), uvec2(2,1)
};

vec2 align_cascade(vec2 offset, vec2 area, float scale, uvec2 resolution)
{
    vec2 cascade_step_size = (area*scale)/vec2(resolution);
    return round(offset / cascade_step_size) * cascade_step_size;
}

}


namespace tr
{

shadow_map_renderer::shadow_map_renderer(context& ctx)
: ctx(&ctx)
{
    init_resources();
}

shadow_map_renderer::~shadow_map_renderer()
{
}

void shadow_map_renderer::set_scene(scene* s)
{
    cur_scene = s;
    init_scene_resources();
}

dependencies shadow_map_renderer::render(dependencies deps)
{
    dependencies out_deps;

    for(auto& p: smp) out_deps.concat(p->run(deps));
    return out_deps;
}

const atlas* shadow_map_renderer::get_shadow_map_atlas() const
{
    return shadow_atlas.get();
}

int shadow_map_renderer::get_shadow_map_index(const light* l) const
{
    auto it = shadow_map_indices.find(l);
    if(it == shadow_map_indices.end())
        return -1;
    return shadow_maps[it->second].map_index;
}

void shadow_map_renderer::update_shadow_map_params()
{
    if(!cur_scene) return;

    size_t map_index = 0;

    for(const directional_light* dl: cur_scene->get_directional_lights())
    {
        const auto* spec = cur_scene->get_shadow_map(dl);
        if(!spec) continue;

        mat4 transform = dl->get_global_transform();
        shadow_map& sm = shadow_maps[map_index++];

        // Bias is adjusted here so that it's independent of depth range. The
        // constant is simply so that the values are in similar ranges to other
        // shadow types.
        float bias_scale = 20.0f/abs(spec->depth_range.x - spec->depth_range.y);
        vec2 area_size = abs(vec2(
            spec->x_range.y-spec->x_range.x, spec->y_range.y-spec->y_range.x
        ));
        sm.min_bias = spec->min_bias*bias_scale;
        sm.max_bias = spec->max_bias*bias_scale;
        sm.radius = tan(radians(dl->get_angle()))/area_size;
        vec2 top_offset = spec->cascades.empty() ? vec2(0) : align_cascade(
            spec->cascades[0], area_size, 1.0f, spec->resolution
        );
        camera face_cam;
        face_cam.ortho(
            spec->x_range.x+top_offset.x, spec->x_range.y+top_offset.x,
            spec->y_range.x+top_offset.y, spec->y_range.y+top_offset.y,
            spec->depth_range.x, spec->depth_range.y
        );
        face_cam.set_transform(transform);
        sm.faces = {face_cam};

        float cascade_scale = 2.0f;
        for(size_t i = 1; i < spec->cascades.size(); ++i)
        {
            shadow_map::cascade& c = sm.cascades[i-1];

            vec2 offset = align_cascade(
                spec->cascades[i], area_size, cascade_scale, spec->resolution
            );
            vec4 area = vec4(
                spec->x_range * cascade_scale + offset.x,
                spec->y_range * cascade_scale + offset.y
            );

            c.offset = (top_offset-offset)/
                abs(0.5f*vec2(area.x-area.y, area.z-area.w));
            c.scale = cascade_scale;
            c.bias_scale = sqrt(cascade_scale);
            c.cam = face_cam;
            c.cam.ortho(
                area.x, area.y, area.z, area.w,
                spec->depth_range.x, spec->depth_range.y
            );

            cascade_scale *= 2.0f;
        }
    }

    for(const point_light* pl: cur_scene->get_point_lights())
    {
        const auto* spec = cur_scene->get_shadow_map(pl);
        if(!spec) continue;

        mat4 transform = pl->get_global_transform();
        shadow_map& sm = shadow_maps[map_index++];

        sm.min_bias = spec->min_bias;
        sm.max_bias = spec->max_bias;
        sm.radius = vec2(pl->get_radius()); // TODO: Radius scaling for PCF?

        // Omnidirectional
        sm.faces.clear();
        for(int i = 0; i < 6; ++i)
        {
            camera face_cam;
            face_cam.set_position(get_matrix_translation(transform));
            face_cam.set_orientation(face_orientations[i]);
            face_cam.perspective(
                90.0f, 1.0f, spec->near, pl->get_cutoff_radius()
            );
            sm.faces.push_back(face_cam);
        }
    }

    for(const spotlight* sl: cur_scene->get_spotlights())
    {
        const auto* spec = cur_scene->get_shadow_map(sl);
        if(!spec) continue;

        mat4 transform = sl->get_global_transform();
        shadow_map& sm = shadow_maps[map_index++];

        // Perspective shadow map, if cutoff angle is small enough.
        if(sl->get_cutoff_angle() < 60)
        {
            camera face_cam;
            face_cam.set_transform(transform);
            face_cam.perspective(
                sl->get_cutoff_angle()*2, 1.0f,
                spec->near, sl->get_cutoff_radius()
            );
            sm.faces = {face_cam};
        }
        // Otherwise, just use omnidirectional shadow map like other point
        // lights
        else
        {
            sm.faces.clear();
            for(int i = 0; i < 6; ++i)
            {
                camera face_cam;
                face_cam.set_position(get_matrix_translation(transform));
                face_cam.set_orientation(face_orientations[i]);
                face_cam.perspective(
                    90.0f, 1.0f, spec->near, sl->get_cutoff_radius()
                );
                sm.faces.push_back(face_cam);
            }
        }

        sm.min_bias = spec->min_bias;
        sm.max_bias = spec->max_bias;
        // TODO: Radius scaling for PCF?
        sm.radius = vec2(sl->get_radius());
    }

    size_t j = 0;
    for(size_t i = 0; i < shadow_maps.size(); ++i)
    {
        shadow_map& sm = shadow_maps[i];
        for(const camera& cam: sm.faces) smp[j++]->set_camera(cam);
        for(const auto& c: sm.cascades) smp[j++]->set_camera(c.cam);
    }
}

const std::vector<shadow_map_renderer::shadow_map>&
shadow_map_renderer::get_shadow_map_info() const
{
    return shadow_maps;
}

size_t shadow_map_renderer::get_total_shadow_map_count() const
{
    return total_shadow_map_count;
}

size_t shadow_map_renderer::get_total_cascade_count() const
{
    return total_cascade_count;
}

void shadow_map_renderer::init_resources()
{
    device_data& d = ctx->get_display_device();

    shadow_atlas.reset(new atlas(
        d, {}, 1, vk::Format::eD32Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled |
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::ImageLayout::eShaderReadOnlyOptimal
    ));
}

void shadow_map_renderer::init_scene_resources()
{
    std::vector<uvec2> shadow_map_sizes;

    // Cascades don't count towards this, but do count towards the above.
    total_shadow_map_count = 0;
    total_cascade_count = 0;
    size_t total_viewport_count = 0;

    shadow_maps.clear();
    shadow_map_indices.clear();
    if(!cur_scene) return;

    for(const directional_light* dl: cur_scene->get_directional_lights())
    {
        const auto* spec = cur_scene->get_shadow_map(dl);
        if(!spec) continue;

        total_shadow_map_count++;

        shadow_map_indices[dl] = shadow_maps.size();
        shadow_map* sm = &shadow_maps.emplace_back();

        sm->atlas_index = shadow_map_sizes.size();
        sm->map_index = shadow_maps.size()-1;
        uvec2 face_size = spec->resolution;
        sm->face_size = face_size;

        sm->faces.resize(1);
        total_viewport_count++;

        shadow_map_sizes.push_back(spec->resolution);

        for(size_t i = 1; i < spec->cascades.size(); ++i)
        {
            shadow_map::cascade c;
            c.atlas_index = (unsigned)shadow_map_sizes.size();
            sm->cascades.push_back(c);
            shadow_map_sizes.push_back(spec->resolution);
            total_cascade_count++;
            total_viewport_count++;
        }
    }

    for(const point_light* pl: cur_scene->get_point_lights())
    {
        const auto* spec = cur_scene->get_shadow_map(pl);
        if(!spec) continue;

        total_shadow_map_count++;

        shadow_map_indices[pl] = shadow_maps.size();
        shadow_map* sm = &shadow_maps.emplace_back();

        sm->atlas_index = shadow_map_sizes.size();
        sm->map_index = shadow_maps.size()-1;
        sm->face_size = spec->resolution;
        shadow_map_sizes.push_back(spec->resolution*uvec2(3,2));

        sm->faces.resize(6);
        total_viewport_count += sm->faces.size();
    }

    for(const spotlight* sl: cur_scene->get_spotlights())
    {
        const auto* spec = cur_scene->get_shadow_map(sl);
        if(!spec) continue;

        shadow_map_indices[sl] = shadow_maps.size();
        shadow_map* sm = &shadow_maps.emplace_back();

        // Perspective shadow map, if cutoff angle is small enough.
        if(sl->get_cutoff_angle() < 60)
        {
            shadow_map_sizes.push_back(spec->resolution);
            sm->faces.resize(1);
        }
        // Otherwise, just use omnidirectional shadow map like other point
        // lights
        else
        {
            shadow_map_sizes.push_back(spec->resolution * uvec2(3,2));
            sm->faces.resize(6);
        }
        total_viewport_count += sm->faces.size();

        total_shadow_map_count++;
        sm->atlas_index = shadow_map_sizes.size()-1;
        sm->map_index = shadow_maps.size()-1;
        sm->face_size = spec->resolution;
    }

    smp.clear();
    shadow_atlas->set_sub_textures(shadow_map_sizes, 0);

    size_t atlas_index = 0;
    for(size_t i = 0; i < shadow_maps.size(); ++i)
    {
        shadow_map& sm = shadow_maps[i];
        for(size_t k = 0; k < sm.faces.size(); ++k)
            add_stage(atlas_index, k, sm.faces.size());
        atlas_index++;
        for(size_t k = 0; k < sm.cascades.size(); ++k)
            add_stage(atlas_index++, 0, 1);
    }

    update_shadow_map_params();
}

void shadow_map_renderer::add_stage(
    unsigned atlas_index, unsigned face_index, unsigned face_count
){
    // Assume all atlases have the same layout
    uvec4 rect = shadow_atlas->get_rect_px(atlas_index);

    if(face_count == 6)
    {
        rect.z /= 3;
        rect.w /= 2;
        uvec2 offset = face_offset_mul[face_index];
        rect.x += offset.x * rect.z;
        rect.y += offset.y * rect.w;
    }

    render_target target = shadow_atlas->get_layer_render_target(
        0, ctx->get_display_device().index
    );
    smp.emplace_back(new shadow_map_stage(
        ctx->get_display_device(), rect, target,
        {cur_scene->get_sampler_count()}
    ));
    smp.back()->set_scene(cur_scene);
}

}
