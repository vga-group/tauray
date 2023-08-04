#include "scene_stage.hh"
#include "shadow_map_renderer.hh"
#include "sh_grid.hh"
#include "environment_map.hh"
#include "placeholders.hh"
#include "misc.hh"

namespace
{
using namespace tr;

struct material_buffer
{
    pvec4 albedo_factor;
    pvec4 metallic_roughness_factor;
    pvec4 emission_factor_double_sided;
    float transmittance;
    float ior;
    float normal_factor;
    float pad[1];
    int albedo_tex_id;
    int metallic_roughness_tex_id;
    int normal_tex_id;
    int emission_tex_id;
};

struct instance_buffer
{
    // -1 if not an area light source, otherwise base index to triangle light
    // array.
    int32_t light_base_id;
    int32_t sh_grid_index;
    uint32_t pad;
    float shadow_terminator_mul;
    pmat4 model;
    pmat4 model_normal;
    pmat4 model_prev;
    material_buffer mat;
};

struct directional_light_entry
{
    directional_light_entry() = default;
    directional_light_entry(
        const directional_light& dl,
        const shadow_map_renderer* smr
    ):  color(dl.get_color()), shadow_map_index(-1),
        dir(dl.get_global_direction()), dir_cutoff(cos(radians(dl.get_angle())))
    {
        if(smr) shadow_map_index = smr->get_shadow_map_index(&dl);
    }

    pvec3 color;
    int shadow_map_index;
    pvec3 dir;
    float dir_cutoff;
};

struct point_light_entry
{
    point_light_entry() = default;
    point_light_entry(const point_light& pl, const shadow_map_renderer* smr)
    :   color(pl.get_color()), dir(vec3(0)), pos(pl.get_global_position()),
        radius(pl.get_radius()), dir_cutoff(0.0f), dir_falloff(0.0f),
        cutoff_radius(pl.get_cutoff_radius()), spot_radius(-1.0f),
        shadow_map_index(-1)
    {
        if(smr) shadow_map_index = smr->get_shadow_map_index(&pl);
    }

    point_light_entry(const spotlight& sl, const shadow_map_renderer* smr)
    :   color(sl.get_color()), dir(sl.get_global_direction()),
        pos(sl.get_global_position()), radius(sl.get_radius()),
        dir_cutoff(cos(radians(sl.get_cutoff_angle()))),
        dir_falloff(sl.get_falloff_exponent()),
        cutoff_radius(sl.get_cutoff_radius()),
        spot_radius(
            sl.get_cutoff_radius() * tan(radians(sl.get_cutoff_angle()))
        ),
        shadow_map_index(-1)
    {
        if(smr) shadow_map_index = smr->get_shadow_map_index(&sl);
    }

    pvec3 color;
    pvec3 dir;
    pvec3 pos;
    float radius;
    float dir_cutoff;
    float dir_falloff;
    float cutoff_radius;
    float spot_radius;
    int shadow_map_index;
    int padding;
};

// These aren't built on the CPU, so this definition is only used for sizeof.
// They're also not supported with rasterization, so they don't carry any shadow
// mapping info.
struct tri_light_entry
{
    pvec3 pos[3];
    pvec3 emission_factor;

    pvec2 uv[3];
    int emission_tex_id;

    // TODO: Put pre-calculated data here, since we want to pad to a multiple of
    // 32 anyway. There's 5 ints left.
    //int padding[5];
};

struct sh_grid_buffer
{
    alignas(16) pmat4 pos_from_world;
    alignas(16) pmat4 normal_from_world;
    pvec3 grid_clamp;
    float pad0;
    pvec3 grid_resolution;
    float pad1;
};

struct shadow_map_entry
{
    // If directional shadow map, number of additional cascades. Otherwise, 0 if
    // perspective, 1 if omni.
    int type;
    float min_bias;
    float max_bias;
    // Index to the cascade buffer, if directional shadow map and type > 0.
    int cascade_index;
    // The actual shadow map data resides in the atlas. This 'rect' defines the
    // portion of the atlas which contains the actual shadow. For omni shadows,
    // this is the +X face, and the faces are arranged as follows:
    //
    // +X +Y +Z
    // -X -Y -Z
    //
    // So -Z would be found at rect.xy + ivec2(rect.z*2, rect.w).
    // xy = origin, zw = width and height
    pvec4 rect;
    // Used for linearizing depth for omni shadows.
    // w = near plane.
    pvec4 clip_info;
    // xy = projection info, used to do projection for perspective shadows.
    // zw = PCF radius, adjusted for aspect ratio.
    pvec4 projection_info_radius;
    // Takes a world space point to the light's space.
    pmat4 world_to_shadow;
};

struct shadow_map_cascade_entry
{
    // xy = offset
    // z = scale
    // w = bias_scale
    pvec4 offset_scale;
    // Same as shadow_map_entry.rect
    pvec4 rect;
};

struct light_order_push_constants
{
    uint32_t point_light_count;
    uint32_t morton_shift;
    uint32_t morton_bits;
};

struct scene_metadata_buffer
{
    uint32_t point_light_count;
    uint32_t directional_light_count;
    uint32_t tri_light_count;
};

struct skinning_push_constants
{
    uint32_t vertex_count;
};

struct extract_tri_light_push_constants
{
    uint32_t triangle_count;
    uint32_t instance_id;
};

struct pre_tranform_push_constants
{
    uint32_t vertex_count;
    uint32_t instance_id;
};

}

namespace tr
{

scene_stage::scene_stage(device_mask dev, const options& opt)
:   multi_device_stage(dev),
    as_rebuild(true),
    envmap_change_counter(1),
    geometry_change_counter(1),
    light_change_counter(1),
    command_buffers_outdated(true),
    force_instance_refresh_frames(0),
    cur_scene(nullptr),
    envmap(nullptr),
    ambient(0),
    skinning(dev, compute_pipeline::params{{"shader/skinning.comp"}, {}, 1, true}),
    extract_tri_lights(dev, compute_pipeline::params{
        {"shader/extract_tri_lights.comp", opt.pre_transform_vertices ?
            std::map<std::string, std::string>{{"PRE_TRANSFORMED_VERTICES", ""}} :
            std::map<std::string, std::string>()
        },
        {
            {"vertices", (uint32_t)opt.max_instances},
            {"indices", (uint32_t)opt.max_instances}
        },
        1
    }),
    pre_transform(dev, compute_pipeline::params{
        {"shader/pre_transform.comp"}, {}, 1, true
    }),
    opt(opt), stage_timer(dev, "scene update")
{
}

void scene_stage::set_scene(scene* target)
{
    cur_scene = target;

    cur_scene->refresh_instance_cache(true);

    size_t point_light_count =
        cur_scene->get_point_lights().size() +
        cur_scene->get_spotlights().size();

    size_t point_light_mem = sizeof(point_light_entry) * point_light_count;
    size_t directional_light_mem =
        sizeof(directional_light_entry) * cur_scene->get_directional_lights().size();
    size_t tri_light_count = 0;

    for(const mesh_scene::instance& i: cur_scene->get_instances())
    {
        if(i.mat->emission_factor != vec3(0))
            tri_light_count += i.m->get_indices().size() / 3;
    }

    cur_scene->point_light_data.resize(point_light_mem);
    cur_scene->directional_light_data.resize(directional_light_mem);
    if(opt.gather_emissive_triangles)
        cur_scene->tri_light_data.resize(tri_light_count * sizeof(tri_light_entry));
    else
        cur_scene->tri_light_data.resize(0);

    cur_scene->scene_metadata.resize(sizeof(scene_metadata_buffer));

    cur_scene->s_table.update_scene(cur_scene);

    force_instance_refresh_frames = MAX_FRAMES_IN_FLIGHT;
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        update(i);

    as_rebuild = get_context()->is_ray_tracing_supported();
    command_buffers_outdated = true;

    // TODO: Do this per-category in update().
    geometry_change_counter++;
}

scene* scene_stage::get_scene() const
{
    return cur_scene;
}

bool scene_stage::check_update(uint32_t categories, uint32_t& prev_counter) const
{
    uint32_t new_counter = 0;
    if(categories&ENVMAP) new_counter += envmap_change_counter;
    if(categories&GEOMETRY) new_counter += geometry_change_counter;
    if(categories&LIGHT) new_counter += light_change_counter;
    if(prev_counter != new_counter)
    {
        prev_counter = new_counter;
        return true;
    }
    return false;
}

environment_map* scene_stage::get_environment_map() const
{
    return envmap;
}

vec3 scene_stage::get_ambient() const
{
    return ambient;
}

const std::vector<scene_stage::instance>& scene_stage::get_instances() const
{
    return cur_scene->get_instances();
}

vk::AccelerationStructureKHR scene_stage::get_acceleration_structure(
    device_id id
) const {
    device_mask dev = get_device_mask();
    if(!dev.get_context()->is_ray_tracing_supported())
        throw std::runtime_error(
            "Trying to use TLAS, but ray tracing is not available!"
        );
    return *cur_scene->tlas->get_tlas_handle(id);
}

void scene_stage::set_shadow_map_renderer(shadow_map_renderer* smr)
{
    cur_scene->smr = smr;
}

void scene_stage::set_sh_grid_textures(
    std::unordered_map<sh_grid*, texture>* sh_grid_textures
){
    cur_scene->sh_grid_textures = sh_grid_textures;
}

vec2 scene_stage::get_shadow_map_atlas_pixel_margin() const
{
    if(cur_scene->smr)
        return vec2(0.5)/vec2(cur_scene->smr->get_shadow_map_atlas()->get_size());
    else
        return vec2(0);
}

std::vector<descriptor_state> scene_stage::get_descriptor_info(device_id id, int32_t camera_index) const
{
    const std::vector<sh_grid*>& sh_grids = cur_scene->get_sh_grids();

    std::vector<vk::DescriptorImageInfo> dii_3d;
    if(cur_scene->sh_grid_textures)
    {
        for(sh_grid* sg: sh_grids)
        {
            texture& tex = cur_scene->sh_grid_textures->at(sg);
            dii_3d.push_back({
                cur_scene->sh_grid_sampler.get_sampler(id),
                tex.get_image_view(id),
                vk::ImageLayout::eShaderReadOnlyOptimal
            });
        }
    }

    vk::AccelerationStructureKHR tlas = {};
    device_mask dev = get_device_mask();
    if(dev.get_context()->is_ray_tracing_supported())
        tlas = get_acceleration_structure(id);

    std::vector<vk::DescriptorBufferInfo> dbi_vertex = cur_scene->get_vertex_buffer_bindings(id);
    std::vector<vk::DescriptorBufferInfo> dbi_index = cur_scene->get_index_buffer_bindings(id);
    std::vector<vk::DescriptorImageInfo> dii = cur_scene->s_table.get_image_infos(id);

    std::vector<descriptor_state> descriptors = {
        {"scene", {cur_scene->scene_data[id], 0, VK_WHOLE_SIZE}},
        {"scene_metadata", {cur_scene->scene_metadata[id], 0, VK_WHOLE_SIZE}},
        {"vertices", dbi_vertex},
        {"indices", dbi_index},
        {"textures", dii},
        {"directional_lights", {
            cur_scene->directional_light_data[id], 0, VK_WHOLE_SIZE
        }},
        {"point_lights", {cur_scene->point_light_data[id], 0, VK_WHOLE_SIZE}},
        {"tri_lights", {cur_scene->tri_light_data[id], 0, VK_WHOLE_SIZE}},
        {"environment_map_tex", {
            cur_scene->envmap_sampler.get_sampler(id),
            envmap ? envmap->get_image_view(id) : vk::ImageView{},
            vk::ImageLayout::eShaderReadOnlyOptimal
        }},
        {"environment_map_alias_table", {
            envmap ? envmap->get_alias_table(id) : vk::Buffer{}, 0, VK_WHOLE_SIZE
        }},
        {"textures3d", dii_3d},
        {"sh_grids", {cur_scene->sh_grid_data[id], 0, VK_WHOLE_SIZE}}
    };

    if(camera_index >= 0)
    {
        std::pair<size_t, size_t> camera_offset = cur_scene->camera_data_offsets[camera_index];
        descriptors.push_back(
            {"camera", {cur_scene->camera_data[id], camera_offset.first, VK_WHOLE_SIZE}}
        );
    }

    if(dev.get_context()->is_ray_tracing_supported())
    {
        descriptors.push_back({"tlas", {1, cur_scene->tlas->get_tlas_handle(id)}});
    }

    if(cur_scene->smr)
    {
        placeholders& pl = dev.get_context()->get_placeholders();

        const atlas* shadow_map_atlas = cur_scene->smr->get_shadow_map_atlas();

        descriptors.push_back(
            {"shadow_maps", {cur_scene->shadow_map_data[id], 0, cur_scene->shadow_map_range}}
        );
        descriptors.push_back(
            {"shadow_map_cascades", {
                cur_scene->shadow_map_data[id], cur_scene->shadow_map_range,
                cur_scene->shadow_map_cascade_range
            }}
        );
        descriptors.push_back(
            {"shadow_map_atlas", {
                pl.default_sampler.get_sampler(id),
                shadow_map_atlas->get_image_view(id),
                vk::ImageLayout::eShaderReadOnlyOptimal
            }}
        );
        descriptors.push_back(
            {"shadow_map_atlas_test", {
                cur_scene->shadow_sampler.get_sampler(id),
                shadow_map_atlas->get_image_view(id),
                vk::ImageLayout::eShaderReadOnlyOptimal
            }}
        );
    }
    return descriptors;
}

void scene_stage::bind(basic_pipeline& pipeline, uint32_t frame_index, int32_t camera_index)
{
    device* dev = pipeline.get_device();
    std::vector<descriptor_state> descriptors = get_descriptor_info(dev->index, camera_index);
    pipeline.update_descriptor_set(descriptors, frame_index);
}

void scene_stage::push(basic_pipeline& pipeline, vk::CommandBuffer cmd, int32_t camera_index)
{
    device* dev = pipeline.get_device();
    std::vector<descriptor_state> descriptors = get_descriptor_info(dev->index, camera_index);
    pipeline.push_descriptors(cmd, descriptors);
}

void scene_stage::bind_placeholders(
    basic_pipeline& pipeline,
    size_t max_samplers,
    size_t max_3d_samplers
){
    device* dev = pipeline.get_device();
    placeholders& pl = dev->ctx->get_placeholders();

    pipeline.update_descriptor_set({
        {"textures", max_samplers},
        {"shadow_maps"},
        {"shadow_map_cascades"},
        {"shadow_map_atlas"},
        {"shadow_map_atlas_test", {
            pl.default_sampler.get_sampler(dev->index),
            pl.depth_test_sample.get_image_view(dev->index),
            vk::ImageLayout::eShaderReadOnlyOptimal
        }},
        {"textures3d", {
            pl.default_sampler.get_sampler(dev->index),
            pl.sample3d.get_image_view(dev->index),
            vk::ImageLayout::eShaderReadOnlyOptimal
        }, max_3d_samplers}
    });
}

void scene_stage::update(uint32_t frame_index)
{
    if(!cur_scene) return;

    bool lights_outdated = false;
    bool geometry_outdated = false;

    environment_map* new_envmap = cur_scene->get_environment_map();
    if(new_envmap != envmap)
    {
        envmap = new_envmap;
        envmap_change_counter++;
    }

    if(ambient != cur_scene->get_ambient())
    {
        ambient = cur_scene->get_ambient();
        lights_outdated = true;
    }

    cur_scene->refresh_instance_cache();
    if(cur_scene->cameras.size() != 0)
        cur_scene->track_shadow_maps(cur_scene->cameras);

    uint64_t frame_counter = get_context()->get_frame_counter();
    for(mesh_object* obj: cur_scene->get_mesh_objects())
    {
        model* m = const_cast<model*>(obj->get_model());
        if(!m) continue;
        if(m->has_joints_buffer())
            m->update_joints(frame_index);
    }

    size_t tri_light_count = 0;
    size_t vertex_count = 0;

    const std::vector<scene::instance>& instances = cur_scene->get_instances();
    cur_scene->scene_data.resize(sizeof(instance_buffer) * instances.size());
    cur_scene->scene_data.foreach<instance_buffer>(
        frame_index, instances.size(),
        [&](instance_buffer& inst, size_t i){
            if(instances[i].mat->emission_factor != vec3(0))
            {
                inst.light_base_id = tri_light_count;
                tri_light_count += instances[i].m->get_indices().size() / 3;
            }
            else inst.light_base_id = -1;

            vertex_count += instances[i].m->get_indices().size();

            // Skip unchanged instances.
            if(
                force_instance_refresh_frames == 0 &&
                instances[i].last_refresh_frame+MAX_FRAMES_IN_FLIGHT < frame_counter
            ) return;

            pmat4 model = instances[i].transform;
            inst.model = model;
            inst.model_normal = instances[i].normal_transform;
            inst.model_prev = instances[i].prev_transform;
            int index = -1;
            if(
                cur_scene->sh_grid_textures &&
                !cur_scene->get_sh_grid(model[3], &index)
            ) cur_scene->get_largest_sh_grid(&index);
            inst.sh_grid_index = index;
            inst.pad = 0;
            inst.shadow_terminator_mul = 1.0f/(
                1.0f-0.5f * instances[i].o->get_shadow_terminator_offset()
            );

            const material& mat = *instances[i].mat;
            inst.mat.albedo_factor = mat.albedo_factor;
            inst.mat.metallic_roughness_factor =
                vec4(mat.metallic_factor, mat.roughness_factor, 0, 0);
            inst.mat.emission_factor_double_sided = vec4(
                mat.emission_factor, mat.double_sided ? 1.0f : 0.0f
            );
            inst.mat.transmittance = mat.transmittance;
            inst.mat.ior = mat.ior;
            inst.mat.normal_factor = mat.normal_factor;

            inst.mat.albedo_tex_id = cur_scene->s_table.find_tex_id(mat.albedo_tex);
            inst.mat.metallic_roughness_tex_id =
                cur_scene->s_table.find_tex_id(mat.metallic_roughness_tex);
            inst.mat.normal_tex_id = cur_scene->s_table.find_tex_id(mat.normal_tex);
            inst.mat.emission_tex_id = cur_scene->s_table.find_tex_id(mat.emission_tex);
        }
    );
    if(force_instance_refresh_frames > 0) force_instance_refresh_frames--;

    const std::vector<point_light*>& point_lights = cur_scene->get_point_lights();
    const std::vector<spotlight*>& spotlights = cur_scene->get_spotlights();
    const std::vector<directional_light*>& directional_lights =
        cur_scene->get_directional_lights();

    cur_scene->point_light_data.map<uint8_t>(frame_index, [&](uint8_t* light_data){
        point_light_entry* point_light_data =
            reinterpret_cast<point_light_entry*>(light_data);

        size_t i = 0;
        for(size_t j = 0; j < point_lights.size(); ++j)
        {
            const point_light& pl = *point_lights[j];
            point_light_entry pe = point_light_entry(pl, cur_scene->smr);
            point_light_data[i] = pe;
            ++i;
        }
        for(size_t j = 0; j < spotlights.size(); ++j)
        {
            const spotlight& sl = *spotlights[j];
            point_light_entry pe = point_light_entry(sl, cur_scene->smr);
            point_light_data[i] = pe;
            ++i;
        }
    });

    cur_scene->directional_light_data.map<uint8_t>(frame_index, [&](uint8_t* light_data){
        directional_light_entry* directional_light_data =
            reinterpret_cast<directional_light_entry*>(light_data);

        // Punctual directional lights
        for(size_t i = 0; i < directional_lights.size(); ++i)
        {
            const directional_light& dl = *directional_lights[i];
            directional_light_data[i] = directional_light_entry(dl, cur_scene->smr);
        }
    });

    const std::vector<sh_grid*>& sh_grids = cur_scene->get_sh_grids();
    cur_scene->sh_grid_data.resize(sizeof(sh_grid_buffer) * sh_grids.size());
    cur_scene->sh_grid_data.foreach<sh_grid_buffer>(
        frame_index, sh_grids.size(),
        [&](sh_grid_buffer& sh_data , size_t i){
            sh_data.grid_clamp = 0.5f/vec3(sh_grids[i]->get_resolution());
            sh_data.grid_resolution = sh_grids[i]->get_resolution();
            mat4 transform = sh_grids[i]->get_global_transform();
            quat orientation = get_matrix_orientation(transform);
            sh_data.pos_from_world = glm::affineInverse(transform);
            sh_data.normal_from_world = mat4(inverse(orientation));
        }
    );

    //auto& si = cur_scene->scene_infos[dev->index];

    cur_scene->camera_data_offsets.clear();
    size_t start_offset = 0;
    for(camera* cam: cur_scene->cameras)
    {
        size_t buf_size = camera::get_projection_type_uniform_buffer_size(cam->get_projection_type()) * 2;
        cur_scene->camera_data_offsets.push_back({start_offset, buf_size});
        start_offset += buf_size;
    }
    cur_scene->camera_data.resize(start_offset);
    old_camera_data.resize(start_offset);
    cur_scene->camera_data.map<uint8_t>(
        frame_index, [&](uint8_t* data){
            uint8_t* old_data = old_camera_data.data();
            for(size_t i = 0; i < cur_scene->cameras.size(); ++i)
            {
                camera* cam = cur_scene->cameras[i];
                uint8_t* cur_data = data + cur_scene->camera_data_offsets[i].first;
                size_t buf_size = camera::get_projection_type_uniform_buffer_size(cam->get_projection_type());
                cur_scene->cameras[i]->write_uniform_buffer(cur_data);
                memcpy(cur_data + buf_size, old_data, buf_size);
                memcpy(old_data, cur_data, buf_size);
                old_data += buf_size;
            }
        }
    );

    if(cur_scene->smr)
    {
        cur_scene->smr->update_shadow_map_params();
        const atlas* shadow_map_atlas = cur_scene->smr->get_shadow_map_atlas();
        const std::vector<shadow_map_renderer::shadow_map>& shadow_maps =
            cur_scene->smr->get_shadow_map_info();

        // TODO: Shadow maps should be stored here, in scene_stage, so that
        // this weird interplay with shadow_map_renderer owning some scene
        // assets can be removed.
        cur_scene->shadow_map_range =
            sizeof(shadow_map_entry) * cur_scene->smr->get_total_shadow_map_count();

        cur_scene->shadow_map_cascade_range =
            sizeof(shadow_map_cascade_entry) * cur_scene->smr->get_total_cascade_count();

        cur_scene->shadow_map_data.resize(cur_scene->shadow_map_range + cur_scene->shadow_map_cascade_range);
        cur_scene->shadow_map_data.map<uint8_t>(
            frame_index, [&](uint8_t* sm_data){
            shadow_map_entry* shadow_map_data =
                reinterpret_cast<shadow_map_entry*>(sm_data);
            shadow_map_cascade_entry* shadow_map_cascade_data =
                reinterpret_cast<shadow_map_cascade_entry*>(
                    sm_data + cur_scene->shadow_map_range
                );

            int cascade_index = 0;
            for(const auto& sm: shadow_maps)
            {
                shadow_map_entry& map = shadow_map_data[sm.map_index];
                const camera& first_cam = sm.faces[0];

                map.clip_info = vec4(
                    first_cam.get_clip_info(),
                    first_cam.get_near()
                );
                map.projection_info_radius = vec4(
                    first_cam.get_projection_info(), sm.radius
                );

                // Determine shadow map type from projection
                switch(first_cam.get_projection_type())
                {
                case camera::PERSPECTIVE:
                    { // Cubemap / perspective shadow map
                        if(sm.faces.size() == 6)
                        {
                            map.type = 1;
                            map.world_to_shadow = glm::inverse(
                                sm.faces[5].get_global_transform());
                        }
                        else
                        {
                            map.type = 0;
                            map.world_to_shadow = glm::inverse(
                                first_cam.get_global_transform());
                        }
                    }
                    break;
                case camera::ORTHOGRAPHIC:
                    { // Directional
                        map.clip_info.z = first_cam.get_far();
                        map.type = sm.cascades.size();
                        map.cascade_index = cascade_index;
                        map.world_to_shadow = first_cam.get_view_projection();
                    }
                    break;
                default:
                    throw std::runtime_error(
                        "Only perspective & ortho projections are supported in "
                        "shadow maps!"
                    );
                }

                map.min_bias = sm.min_bias;
                map.max_bias = sm.max_bias;
                map.rect = vec4(
                    ivec2(shadow_map_atlas->get_rect_px(sm.atlas_index)),
                    sm.face_size
                )/vec4(
                    shadow_map_atlas->get_size(),
                    shadow_map_atlas->get_size()
                );

                for(auto& c: sm.cascades)
                {
                    shadow_map_cascade_entry& cascade =
                        shadow_map_cascade_data[cascade_index];
                    cascade.offset_scale = vec4(
                        c.offset,
                        1.0f/c.scale,
                        c.bias_scale
                    );
                    cascade.rect = vec4(
                        ivec2(shadow_map_atlas->get_rect_px(c.atlas_index)),
                        sm.face_size
                    )/vec4(
                        shadow_map_atlas->get_size(),
                        shadow_map_atlas->get_size()
                    );

                    cascade_index++;
                }
            }
        });
    }
    cur_scene->scene_metadata.map<scene_metadata_buffer>(
        frame_index, [&](scene_metadata_buffer* data){
            data->point_light_count = point_lights.size() + spotlights.size();
            data->directional_light_count = directional_lights.size();
            data->tri_light_count = tri_light_count;
        }
    );

    if(lights_outdated) light_change_counter++;
    if(geometry_outdated) geometry_change_counter++;

    if(get_context()->is_ray_tracing_supported())
    {
        bool need_scene_reset = false;
        for(device& dev: get_device_mask())
        {
            cur_scene->light_scene::update_acceleration_structures(
                dev.index,
                frame_index,
                need_scene_reset,
                command_buffers_outdated
            );
            cur_scene->mesh_scene::update_acceleration_structures(
                dev.index,
                frame_index,
                need_scene_reset,
                command_buffers_outdated
            );

            auto& instance_buffer = cur_scene->tlas->get_instances_buffer();

            as_instance_count = 0;
            uint32_t total_max_capacity =
                cur_scene->mesh_scene::get_max_capacity() +
                cur_scene->light_scene::get_max_capacity();
            instance_buffer.map_one<vk::AccelerationStructureInstanceKHR>(
                dev.index,
                frame_index,
                [&](vk::AccelerationStructureInstanceKHR* as_instances){
                    cur_scene->mesh_scene::add_acceleration_structure_instances(
                        as_instances, dev.index, frame_index, as_instance_count, total_max_capacity
                    );
                    cur_scene->light_scene::add_acceleration_structure_instances(
                        as_instances, dev.index, frame_index, as_instance_count, total_max_capacity
                    );
                }
            );
        }

        if(as_rebuild == false)
            as_rebuild = need_scene_reset;

        if(opt.pre_transform_vertices)
            need_scene_reset |= cur_scene->reserve_pre_transformed_vertices(vertex_count);
        else
            cur_scene->clear_pre_transformed_vertices();

        command_buffers_outdated |= need_scene_reset;
    }

    if(command_buffers_outdated)
    {
        record_command_buffers();
        if(as_rebuild == false)
            command_buffers_outdated = false;
        else as_rebuild = false;
    }
}

void scene_stage::record_command_buffers()
{
    clear_commands();

    for(device& dev: get_device_mask())
    {
        if(opt.gather_emissive_triangles)
        {
            extract_tri_lights[dev.index].reset_descriptor_sets();
            bind(extract_tri_lights[dev.index], 0, 0);
        }

        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vk::CommandBuffer cb = begin_graphics(dev.index);
            stage_timer.begin(cb, dev.index, i);
            cur_scene->scene_data.upload(dev.index, i, cb);
            cur_scene->directional_light_data.upload(dev.index, i, cb);
            cur_scene->point_light_data.upload(dev.index, i, cb);
            cur_scene->sh_grid_data.upload(dev.index, i, cb);
            cur_scene->shadow_map_data.upload(dev.index, i, cb);
            cur_scene->camera_data.upload(dev.index, i, cb);
            cur_scene->scene_metadata.upload(dev.index, i, cb);

            bulk_upload_barrier(cb, vk::PipelineStageFlagBits::eComputeShader);

            record_skinning(dev.index, i, cb);
            if(dev.ctx->is_ray_tracing_supported())
            {
                record_as_build(dev.index, i, cb);
                if(opt.pre_transform_vertices)
                    record_pre_transform(dev.index, cb);
                if(cur_scene->tri_light_data.get_size() != 0)
                    record_tri_light_extraction(dev.index, cb);
            }

            stage_timer.end(cb, dev.index, i);
            end_graphics(cb, dev.index, i);
        }
    }
}

void scene_stage::record_skinning(device_id id, uint32_t frame_index, vk::CommandBuffer cb)
{
    skinning[id].bind(cb);

    // Update vertex buffers
    for(mesh_object* obj: cur_scene->get_mesh_objects())
    {
        model* m = const_cast<model*>(obj->get_model());
        if(!m || !m->has_joints_buffer()) continue;

        m->upload_joints(cb, id, frame_index);
        for(auto& vg: *m)
        {
            mesh* dst = vg.m;
            mesh* src = dst->get_animation_source();
            uint32_t vertex_count = vg.m->get_vertices().size();

            skinning[id].push_constants(cb, skinning_push_constants{vertex_count});
            skinning[id].push_descriptors(cb, {
                {"source_data", {src->get_vertex_buffer(id), 0, VK_WHOLE_SIZE}},
                {"destination_data", {dst->get_vertex_buffer(id), 0, VK_WHOLE_SIZE}},
                {"skin_data", {src->get_skin_buffer(id), 0, VK_WHOLE_SIZE}},
                {"joint_data", {m->get_joint_buffer()[id], 0, VK_WHOLE_SIZE}}
            });
            cb.dispatch((vertex_count+31u)/32u, 1, 1);
        }
    }

    // Update acceleration structures
    if(get_context()->is_ray_tracing_supported())
    {
        // Barrier to ensure vertex buffers are updated by the time we try
        // to do BLAS updates.
        vk::MemoryBarrier barrier(
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eAccelerationStructureWriteKHR
        );

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
            {}, barrier, {}, {}
        );

        cur_scene->refresh_dynamic_acceleration_structures(id, frame_index, cb);
    }
}

void scene_stage::record_as_build(
    device_id id,
    uint32_t frame_index,
    vk::CommandBuffer cb
){
    auto& instance_buffer = cur_scene->tlas->get_instances_buffer();
    bool as_update = !as_rebuild;
    cur_scene->mesh_scene::record_acceleration_structure_build(
        cb, id, frame_index, as_update
    );

    cur_scene->light_scene::record_acceleration_structure_build(
        cb, id, frame_index, as_update
    );

    if(as_instance_count > 0)
    {
        instance_buffer.upload(id, frame_index, cb);

        vk::MemoryBarrier barrier(
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eAccelerationStructureWriteKHR
        );

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
            {},
            barrier, {}, {}
        );
    }

    cur_scene->tlas->rebuild(id, cb, as_instance_count, as_update);
}

void scene_stage::record_tri_light_extraction(
    device_id id,
    vk::CommandBuffer cb
){
    const std::vector<scene::instance>& instances = cur_scene->get_instances();
    extract_tri_lights[id].bind(cb, 0);
    for(size_t i = 0; i < instances.size(); ++i)
    {
        const mesh_scene::instance& inst = instances[i];
        if(inst.mat->emission_factor == vec3(0))
            continue;

        extract_tri_light_push_constants pc;
        pc.triangle_count = inst.m->get_indices().size() / 3;
        pc.instance_id = i;

        extract_tri_lights[id].push_constants(cb, pc);
        cb.dispatch((pc.triangle_count+255u)/256u, 1, 1);
    }
}

void scene_stage::record_pre_transform(
    device_id id,
    vk::CommandBuffer cb
){
    const std::vector<scene::instance>& instances = cur_scene->get_instances();
    vk::Buffer pre_transformed_vertices = cur_scene->get_pre_transformed_vertices(id);
    pre_transform[id].bind(cb);
    size_t offset = 0;
    for(size_t i = 0; i < instances.size(); ++i)
    {
        const mesh_scene::instance& inst = instances[i];

        pre_tranform_push_constants pc;
        pc.vertex_count = inst.m->get_vertices().size();
        pc.instance_id = i;

        size_t bytes = pc.vertex_count * sizeof(mesh::vertex);

        pre_transform[id].push_descriptors(cb, {
            {"input_verts", {inst.m->get_vertex_buffer(id), 0, bytes}},
            {"output_verts", {pre_transformed_vertices, offset, bytes}},
            {"scene", {cur_scene->scene_data[id], 0, VK_WHOLE_SIZE}}
        });

        pre_transform[id].push_constants(cb, pc);
        cb.dispatch((pc.vertex_count+255u)/256u, 1, 1);

        offset += bytes;
    }
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, {}, {
            {
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead,
                {}, {}, pre_transformed_vertices, 0, VK_WHOLE_SIZE
            },
        }, {}
    );
}

}
