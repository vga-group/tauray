#include "scene_stage.hh"
#include "sh_grid.hh"
#include "environment_map.hh"
#include "shadow_map.hh"
#include "placeholders.hh"
#include "light.hh"
#include "misc.hh"
#include "log.hh"

namespace
{
using namespace tr;

constexpr uint32_t MATERIAL_FLAG_DOUBLE_SIDED = 1<<0;
constexpr uint32_t MATERIAL_FLAG_TRANSIENT = 1<<1;

struct material_buffer
{
    pvec4 albedo_factor;
    pvec4 metallic_roughness_factor;
    pvec4 emission_factor;
    float transmittance;
    float ior;
    float normal_factor;
    uint32_t flags;
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
        const transformable& t,
        const directional_light& dl,
        int shadow_map_index
    ):  color(dl.get_color()), shadow_map_index(shadow_map_index),
        dir(t.get_global_direction()), dir_cutoff(cos(radians(dl.get_angle())))
    {
    }

    pvec3 color;
    int shadow_map_index;
    pvec3 dir;
    float dir_cutoff;
};

struct point_light_entry
{
    point_light_entry() = default;
    point_light_entry(const transformable& t, const point_light& pl, int shadow_map_index)
    :   color(pl.get_color()), dir(vec3(0)), pos(t.get_global_position()),
        radius(pl.get_radius()), dir_cutoff(0.0f), dir_falloff(0.0f),
        cutoff_radius(pl.get_cutoff_radius()), spot_radius(-1.0f),
        shadow_map_index(shadow_map_index)
    {
    }

    point_light_entry(const transformable& t, const spotlight& sl, int shadow_map_index)
    :   color(sl.get_color()), dir(t.get_global_direction()),
        pos(t.get_global_position()), radius(sl.get_radius()),
        dir_cutoff(cos(radians(sl.get_cutoff_angle()))),
        dir_falloff(sl.get_falloff_exponent()),
        cutoff_radius(sl.get_cutoff_radius()),
        spot_radius(
            sl.get_cutoff_radius() * tan(radians(sl.get_cutoff_angle()))
        ),
        shadow_map_index(shadow_map_index)
    {
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
    uint32_t emission_factor; // R9G9B9E5
    uint32_t instance_id;
    uint32_t primitive_id;
    uint32_t uv[3];
    int emission_tex_id;
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
    pvec4 projection_info;
    // x = near plane
    // z = far plane
    // zw = PCF radius, adjusted for aspect ratio.
    pvec4 range_radius;
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
    uint32_t instance_count;
    uint32_t point_light_count;
    uint32_t directional_light_count;
    uint32_t tri_light_count;
    pvec4 environment_factor;
    int32_t environment_proj;
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

struct temporal_instance_data
{
    uint64_t frame_stamp;
    uint32_t static_base_index;
    uint32_t dynamic_base_index;
    uint32_t group_count;
    mat4 cur_transform;
    mat4 cur_normal_transform;
    mat4 prev_transform;
    mat4 prev_normal_transform;
};

struct temporal_light_data
{
    uint32_t prev_index;
};

const quat face_orientations[6] = {
    glm::quatLookAt(vec3(-1,0,0), vec3(0,1,0)),
    glm::quatLookAt(vec3(1,0,0), vec3(0,1,0)),
    glm::quatLookAt(vec3(0,-1,0), vec3(0,0,1)),
    glm::quatLookAt(vec3(0,1,0), vec3(0,0,1)),
    glm::quatLookAt(vec3(0,0,-1), vec3(0,1,0)),
    glm::quatLookAt(vec3(0,0,1), vec3(0,1,0))
};

vec2 align_cascade(vec2 offset, vec2 area, float scale, uvec2 resolution)
{
    vec2 cascade_step_size = (area*scale)/vec2(resolution);
    return round(offset / cascade_step_size) * cascade_step_size;
}

}

namespace tr
{

scene_stage::scene_stage(device_mask dev, const options& opt)
:   multi_device_stage(dev),
    prev_was_rebuild(false),
    envmap_change_counter(1),
    geometry_change_counter(1),
    light_change_counter(1),
    geometry_outdated(true),
    lights_outdated(true),
    force_instance_refresh_frames(0),
    cur_scene(nullptr),
    envmap(nullptr),
    ambient(0),
    pre_transformed_vertices(dev),
    group_strategy(opt.group_strategy),
    total_shadow_map_count(0),
    total_cascade_count(0),
    shadow_map_range(0),
    shadow_map_cascade_range(0),
    s_table(dev, true),
    instance_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    scene_metadata(dev, sizeof(scene_metadata_buffer), vk::BufferUsageFlagBits::eUniformBuffer),
    directional_light_data(dev, 4, vk::BufferUsageFlagBits::eStorageBuffer),
    point_light_data(dev, max(opt.max_lights * sizeof(point_light_entry), (size_t)4), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc),
    tri_light_data(dev, 4, vk::BufferUsageFlagBits::eStorageBuffer),
    sh_grid_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    shadow_map_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    camera_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    envmap_sampler(
        dev, vk::Filter::eLinear, vk::Filter::eLinear,
        vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest,
        0, true, false
    ),
    shadow_sampler(
        dev,
        vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest,
        0,
        true,
        false,
        true
    ),
    sh_grid_sampler(
        dev,
        vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest,
        0,
        true,
        false
    ),
    brdf_integration_sampler(
        dev,
        vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest,
        0,
        true,
        false
    ),
    brdf_integration(dev, get_resource_path("data/brdf_integration.exr")),
    noise_vector_2d(dev, get_resource_path("data/noise_vector_2d.exr")),
    noise_vector_3d(dev, get_resource_path("data/noise_vector_3d.exr")),
    prev_instance_count(0),
    prev_point_light_count(0),
    temporal_tables(dev, max(2 * (opt.max_instances + opt.max_lights) * sizeof(uint32_t), (size_t)4), vk::BufferUsageFlagBits::eStorageBuffer),
    prev_point_light_data(dev, max(opt.max_lights * sizeof(point_light_entry), (size_t)4), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst),
    scene_desc(dev),
    scene_raster_desc(dev),
    temporal_tables_desc(dev),
    skinning_desc(dev),
    pre_transform_desc(dev),
    opt(opt), stage_timer(dev, "scene update")
{
    skinning.emplace(dev);
    extract_tri_lights.emplace(dev);
    pre_transform.emplace(dev);

    init_descriptor_set_layout();
    {
        shader_source src("shader/skinning.comp");
        skinning_desc.add(src);
        for(const auto&[dev, p]: skinning)
            p.init(src, {&skinning_desc});
    }

    for(const auto&[dev, p]: extract_tri_lights)
    {
        p.init(
            {"shader/extract_tri_lights.comp", opt.pre_transform_vertices ?
                std::map<std::string, std::string>{{"PRE_TRANSFORMED_VERTICES", ""}} :
                std::map<std::string, std::string>()
            }, {&scene_desc}
        );
    }

    {
        shader_source src("shader/pre_transform.comp");
        pre_transform_desc.add(src);
        for(const auto&[dev, p]: pre_transform)
            p.init(src, {&pre_transform_desc, &scene_desc});
    }

    if(opt.shadow_mapping)
    {
        shadow_atlas.reset(new atlas(
            dev, {}, 1, vk::Format::eD32Sfloat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::ImageLayout::eShaderReadOnlyOptimal
        ));
    }

    if(dev.get_context()->is_ray_tracing_supported())
    {
        tlas.emplace(dev, opt.max_instances);
        if(opt.track_prev_tlas)
            prev_tlas.emplace(dev, opt.max_instances);

        if(opt.max_lights > 0)
        {
            light_aabb_buffer = gpu_buffer(
                dev, opt.max_lights * sizeof(vk::AabbPositionsKHR),
                vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eShaderDeviceAddress|
                vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
            );

            light_blas.emplace(
                dev,
                std::vector<bottom_level_acceleration_structure::entry>{
                    {nullptr, opt.max_lights, &light_aabb_buffer, mat4(1.0f), true}
                },
                false, true, false
            );
        }
    }
}

void scene_stage::set_scene(scene* target)
{
    cur_scene = target;

    events[0].emplace(target->subscribe([this](scene&, const add_component<model>&){ geometry_outdated = true; }));
    events[1].emplace(target->subscribe([this](scene&, const remove_component<model>&){ geometry_outdated = true; }));
    events[2].emplace(target->subscribe([this](scene&, const add_component<point_light>&){ lights_outdated = true; }));
    events[3].emplace(target->subscribe([this](scene&, const remove_component<point_light>&){ lights_outdated = true; }));
    events[4].emplace(target->subscribe([this](scene&, const add_component<directional_light>&){ lights_outdated = true; }));
    events[5].emplace(target->subscribe([this](scene&, const remove_component<directional_light>&){ lights_outdated = true; }));
    events[6].emplace(target->subscribe([this](scene&, const add_component<spotlight>&){ lights_outdated = true; }));
    events[7].emplace(target->subscribe([this](scene&, const remove_component<spotlight>&){ lights_outdated = true; }));
    events[8].emplace(target->subscribe([this](scene&, const add_component<sh_grid>&){ lights_outdated = true; }));
    events[9].emplace(target->subscribe([this](scene&, const remove_component<sh_grid>&){ lights_outdated = true; }));

    prev_was_rebuild = false;
    force_instance_refresh_frames = MAX_FRAMES_IN_FLIGHT;

    envmap_change_counter++;
    geometry_change_counter++;
    light_change_counter++;
    geometry_outdated = true;
    lights_outdated = true;
}

scene* scene_stage::get_scene() const
{
    return cur_scene;
}

bool scene_stage::check_update(uint32_t categories, uint32_t& prev_counter) const
{
    uint32_t new_counter = 0;
    // HACK: Descriptor sets are currently updated from changes that we aren't tracking here
    // which causes all command buffers to require updating
    /*if (categories & ENVMAP)*/ new_counter += envmap_change_counter;
    /*if (categories & GEOMETRY)*/ new_counter += geometry_change_counter;
    /*if (categories & LIGHT)*/ new_counter += light_change_counter;
    if(prev_counter != new_counter)
    {
        prev_counter = new_counter;
        return true;
    }
    return false;
}

bool scene_stage::has_prev_tlas() const
{
    return opt.track_prev_tlas;
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
    return instances;
}

const std::unordered_map<sh_grid*, texture>& scene_stage::get_sh_grid_textures() const
{
    return sh_grid_textures;
}

descriptor_set& scene_stage::get_descriptors()
{
    return scene_desc;
}

descriptor_set& scene_stage::get_raster_descriptors()
{
    return scene_raster_desc;
}

descriptor_set& scene_stage::get_temporal_tables()
{
    return temporal_tables_desc;
}

void scene_stage::get_defines(std::map<std::string, std::string>& defines)
{
    if(opt.pre_transform_vertices)
        defines["PRE_TRANSFORMED_VERTICES"];
}

vec2 scene_stage::get_shadow_map_atlas_pixel_margin() const
{
    if(shadow_atlas)
        return vec2(0.5)/vec2(shadow_atlas->get_size());
    else
        return vec2(0);
}

const std::vector<scene_stage::shadow_map_instance>&
scene_stage::get_shadow_maps() const
{
    return shadow_maps;
}

atlas* scene_stage::get_shadow_map_atlas() const
{
    return shadow_atlas.get();
}

bool scene_stage::update_shadow_map_params()
{
    std::vector<uvec2> shadow_map_sizes;

    // Cascades don't count towards this, but do count towards the above.
    total_shadow_map_count = 0;
    total_cascade_count = 0;
    size_t total_viewport_count = 0;

    shadow_maps.clear();
    shadow_map_indices.clear();

    cur_scene->foreach([&](transformable& t, directional_light& dl, directional_shadow_map& spec){
        total_shadow_map_count++;
        total_viewport_count++;

        shadow_map_indices[&dl] = shadow_maps.size();
        shadow_map_instance* sm = &shadow_maps.emplace_back();
        mat4 transform = t.get_global_transform();

        sm->atlas_index = shadow_map_sizes.size();
        sm->map_index = shadow_maps.size()-1;
        sm->face_size = spec.resolution;

        shadow_map_sizes.push_back(spec.resolution);

        // Bias is adjusted here so that it's independent of depth range. The
        // constant is simply so that the values are in similar ranges to other
        // shadow types.
        float bias_scale = 20.0f/abs(spec.depth_range.x - spec.depth_range.y);
        vec2 area_size = abs(vec2(
            spec.x_range.y-spec.x_range.x, spec.y_range.y-spec.y_range.x
        ));
        sm->min_bias = spec.min_bias*bias_scale;
        sm->max_bias = spec.max_bias*bias_scale;
        sm->radius = tan(radians(dl.get_angle()))/area_size;
        vec2 top_offset = spec.cascades.empty() ? vec2(0) : align_cascade(
            spec.cascades[0], area_size, 1.0f, spec.resolution
        );
        camera face_cam;
        face_cam.ortho(
            spec.x_range.x+top_offset.x, spec.x_range.y+top_offset.x,
            spec.y_range.x+top_offset.y, spec.y_range.y+top_offset.y,
            spec.depth_range.x, spec.depth_range.y
        );
        sm->faces = {{face_cam, transform}};

        float cascade_scale = 2.0f;
        for(size_t i = 1; i < spec.cascades.size(); ++i)
        {
            shadow_map_instance::cascade c;
            c.atlas_index = (unsigned)shadow_map_sizes.size();
            shadow_map_sizes.push_back(spec.resolution);
            total_cascade_count++;
            total_viewport_count++;

            vec2 offset = align_cascade(
                spec.cascades[i], area_size, cascade_scale, spec.resolution
            );
            vec4 area = vec4(
                spec.x_range * cascade_scale + offset.x,
                spec.y_range * cascade_scale + offset.y
            );

            c.offset = (top_offset-offset)/
                abs(0.5f*vec2(area.x-area.y, area.z-area.w));
            c.scale = cascade_scale;
            c.bias_scale = sqrt(cascade_scale);
            c.cam = sm->faces[0];
            c.cam.cam.ortho(
                area.x, area.y, area.z, area.w,
                spec.depth_range.x, spec.depth_range.y
            );

            cascade_scale *= 2.0f;
            sm->cascades.push_back(c);
        }
    });

    cur_scene->foreach([&](transformable& t, point_light& pl, point_shadow_map& spec){
        total_shadow_map_count++;

        shadow_map_indices[&pl] = shadow_maps.size();
        shadow_map_instance* sm = &shadow_maps.emplace_back();

        sm->atlas_index = shadow_map_sizes.size();
        sm->map_index = shadow_maps.size()-1;
        sm->face_size = spec.resolution;

        shadow_map_sizes.push_back(spec.resolution*uvec2(3,2));

        mat4 transform = t.get_global_transform();

        sm->min_bias = spec.min_bias;
        sm->max_bias = spec.max_bias;
        sm->radius = vec2(pl.get_radius()); // TODO: Radius scaling for PCF?

        // Omnidirectional
        sm->faces.clear();
        transformable temp;

        temp.set_position(get_matrix_translation(transform));

        for(int i = 0; i < 6; ++i)
        {
            camera face_cam;
            face_cam.perspective(
                90.0f, 1.0f, spec.near, pl.get_cutoff_radius()
            );

            temp.set_orientation(face_orientations[i]);
            sm->faces.push_back({face_cam, temp.get_global_transform()});
            total_viewport_count++;
        }
    });

    cur_scene->foreach([&](transformable& t, spotlight& sl, point_shadow_map& spec){
        mat4 transform = t.get_global_transform();
        shadow_map_indices[&sl] = shadow_maps.size();
        shadow_map_instance* sm = &shadow_maps.emplace_back();

        // Perspective shadow map, if cutoff angle is small enough.
        if(sl.get_cutoff_angle() < 60)
        {
            shadow_map_sizes.push_back(spec.resolution);
            camera face_cam;
            face_cam.perspective(
                sl.get_cutoff_angle()*2, 1.0f,
                spec.near, sl.get_cutoff_radius()
            );
            sm->faces = {{face_cam, transform}};
        }
        // Otherwise, just use omnidirectional shadow map like other point
        // lights
        else
        {
            shadow_map_sizes.push_back(spec.resolution * uvec2(3,2));
            sm->faces.clear();
            transformable temp;
            temp.set_position(get_matrix_translation(transform));
            for(int i = 0; i < 6; ++i)
            {
                camera face_cam;
                temp.set_orientation(face_orientations[i]);
                face_cam.perspective(
                    90.0f, 1.0f, spec.near, sl.get_cutoff_radius()
                );
                sm->faces.push_back({face_cam, temp.get_global_transform()});
            }
        }
        total_viewport_count += sm->faces.size();
        total_shadow_map_count++;

        sm->atlas_index = shadow_map_sizes.size()-1;
        sm->map_index = shadow_maps.size()-1;
        sm->face_size = spec.resolution;
        sm->min_bias = spec.min_bias;
        sm->max_bias = spec.max_bias;
        sm->radius = vec2(sl.get_radius());
    });

    return shadow_atlas->set_sub_textures(shadow_map_sizes, 0);
}

int scene_stage::get_shadow_map_index(const light* l)
{
    auto it = shadow_map_indices.find(l);
    if(it == shadow_map_indices.end())
        return -1;
    return shadow_maps[it->second].map_index;
}

bool scene_stage::refresh_instance_cache()
{
    uint64_t frame_counter = get_context()->get_frame_counter();
    uint32_t i = 0;
    entity last_object_id = INVALID_ENTITY;
    group_cache.clear();

    prev_instance_count = backward_instance_ids.size();
    backward_instance_ids.clear();

    bool scene_changed = false;

    auto add_instances = [&](bool static_mesh, bool static_transformable){
        cur_scene->foreach([&](entity id, transformable& t, model& mod, temporal_instance_data* td){
            // If requesting dynamic meshes, we don't care about the
            // transformable staticness any more.
            if(static_mesh && static_transformable != t.is_static())
                return;

            mat4 transform;
            mat4 normal_transform;
            mat4 prev_transform;
            uint32_t prev_i = 0xFFFFFFFFu;

            if(td)
            {
                if(td->group_count == mod.group_count())
                    prev_i = static_mesh ? td->static_base_index : td->dynamic_base_index;
                if(static_mesh) td->static_base_index = i;
                else td->dynamic_base_index = i;

                td->group_count = mod.group_count();

                if(td->frame_stamp == frame_counter || t.is_static())
                {
                    td->frame_stamp = frame_counter;
                    transform = td->cur_transform;
                    normal_transform = td->cur_normal_transform;
                    prev_transform = td->prev_transform;
                }
                else
                {
                    transform = t.get_global_transform();
                    normal_transform = t.get_global_inverse_transpose_transform();
                    prev_transform = td->cur_transform;
                    td->frame_stamp = frame_counter;
                    td->prev_transform = td->cur_transform;
                    td->prev_normal_transform = td->cur_normal_transform;
                    td->cur_transform = transform;
                    td->cur_normal_transform = normal_transform;
                }
            }
            else
            {
                transform = t.get_global_transform();
                normal_transform = t.get_global_inverse_transpose_transform();
                prev_transform = transform;
                cur_scene->attach(id, temporal_instance_data{
                    frame_counter,
                    static_mesh ? i : 0xFFFFFFFFu,
                    static_mesh ? 0xFFFFFFFFu : i,
                    (uint32_t)mod.group_count(), transform, normal_transform, transform, normal_transform
                });
            }

            for(const auto& vg: mod)
            {
                bool is_static = !vg.m->is_skinned() && !vg.m->get_animation_source();
                if(static_mesh != is_static)
                    continue;

                if(i == instances.size())
                {
                    instances.push_back({
                        mat4(0),
                        mat4(0),
                        mat4(0),
                        nullptr,
                        nullptr,
                        nullptr,
                        frame_counter
                    });
                    scene_changed = true;
                }
                instance& inst = instances[i];

                assign_group_cache(
                    vg.m->get_id(),
                    static_mesh,
                    static_transformable,
                    id,
                    last_object_id
                );

                if(inst.mat != &vg.mat)
                {
                    inst.mat = &vg.mat;
                    inst.prev_transform = mat4(0);
                    inst.last_refresh_frame = frame_counter;
                    scene_changed = true;
                }
                if(inst.m != vg.m)
                {
                    inst.m = vg.m;
                    inst.prev_transform = mat4(0);
                    inst.last_refresh_frame = frame_counter;
                    scene_changed = true;
                }
                if(inst.mod != &mod)
                {
                    inst.mod = &mod;
                    inst.prev_transform = mat4(0);
                    inst.last_refresh_frame = frame_counter;
                    scene_changed = true;
                }

                if(inst.last_refresh_frame+1 >= frame_counter || !t.is_static())
                {
                    if(inst.prev_transform != prev_transform)
                    {
                        inst.prev_transform = prev_transform;
                        inst.last_refresh_frame = frame_counter;
                    }
                    if(inst.transform != transform)
                    {
                        inst.transform = transform;
                        inst.normal_transform = normal_transform;
                        inst.last_refresh_frame = frame_counter;
                    }
                }
                backward_instance_ids.push_back(prev_i);
                if(prev_i != 0xFFFFFFFFu)
                    prev_i++;
                ++i;
            }
        });
    };
    add_instances(true, true);
    add_instances(true, false);
    add_instances(false, false);
    if(instances.size() > i)
    {
        instances.resize(i);
        scene_changed = true;
    }

    if(scene_changed)
        ensure_blas();

    if(instances.size() > opt.max_instances)
        throw std::runtime_error("The scene has more meshes than max_instances allows!");

    return scene_changed;
}

void scene_stage::ensure_blas()
{
    if(!get_context()->is_ray_tracing_supported())
        return;
    bool built_one = false;
    // Goes through all groups and ensures they have valid BLASes.
    size_t offset = 0;
    std::vector<bottom_level_acceleration_structure::entry> entries;
    for(const instance_group& group: group_cache)
    {
        auto it = blas_cache.find(group.id);
        if(it != blas_cache.end())
        {
            offset += group.size;
            continue;
        }

        if(!built_one)
            TR_LOG("Building acceleration structures");

        built_one = true;

        entries.clear();
        bool double_sided = false;
        for(size_t i = 0; i < group.size; ++i, ++offset)
        {
            const instance& inst = instances[offset];
            if(inst.mat->double_sided) double_sided = true;
            entries.push_back({
                inst.m,
                0, nullptr,
                group.static_transformable ? inst.transform : mat4(1),
                !inst.mat->potentially_transparent()
            });
        }
        blas_cache.emplace(
            group.id,
            bottom_level_acceleration_structure(
                get_device_mask(),
                entries,
                !double_sided,
                group_strategy == blas_strategy::ALL_MERGED_STATIC ? false : !group.static_mesh,
                group.static_mesh
            )
        );
    }
    if(built_one)
        TR_LOG("Finished building acceleration structures");
}

void scene_stage::assign_group_cache(
    uint64_t id,
    bool static_mesh,
    bool static_transformable,
    entity object_index,
    entity& last_object_index
){
    switch(group_strategy)
    {
    case blas_strategy::PER_MATERIAL:
        group_cache.push_back({id, 1, static_mesh, false});
        break;
    case blas_strategy::PER_MODEL:
        if(last_object_index == object_index)
        {
            instance_group& group = group_cache.back();
            group.id = hash_combine(group.id, id);
            if(!static_mesh) group.static_mesh = false;
            group.size++;
        }
        else group_cache.push_back({id, 1, static_mesh, false});
        break;
    case blas_strategy::STATIC_MERGED_DYNAMIC_PER_MODEL:
        if(group_cache.size() == 0)
        {
            bool is_static = static_mesh && static_transformable;
            group_cache.push_back({
                id, 1, static_mesh, is_static
            });
        }
        else
        {
            instance_group& group = group_cache.back();
            bool prev_is_static = group.static_mesh && group.static_transformable;
            bool is_static = static_mesh && static_transformable;
            if(prev_is_static && is_static)
            {
                group.id = hash_combine(group.id, id);
                group.size++;
            }
            else
            {
                if(last_object_index == object_index)
                {
                    group.id = hash_combine(group.id, id);
                    if(!static_mesh) group.static_mesh = false;
                    group.size++;
                }
                else group_cache.push_back({id, 1, static_mesh, false});
            }
        }
        break;
    case blas_strategy::ALL_MERGED_STATIC:
        {
            if(group_cache.size() == 0)
                group_cache.push_back({0, 0, true, true});
            instance_group& group = group_cache.back();
            group.id = hash_combine(group.id, id);
            if(!static_mesh) group.static_mesh = false;
            group.size++;
        }
        break;
    }
    last_object_index = object_index;
}

bool scene_stage::reserve_pre_transformed_vertices(size_t max_vertex_count)
{
    if(!get_context()->is_ray_tracing_supported())
        return false;

    bool ret = false;
    for(auto[dev, ptv]: pre_transformed_vertices)
    {
        if(ptv.count < max_vertex_count)
        {
            ptv.buf = create_buffer(
                dev,
                {
                    {}, max_vertex_count * sizeof(mesh::vertex),
                    vk::BufferUsageFlagBits::eVertexBuffer|vk::BufferUsageFlagBits::eStorageBuffer,
                    vk::SharingMode::eExclusive
                },
                VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
            );
            ptv.count = max_vertex_count;
            ret = true;
        }
    }
    return ret;
}

void scene_stage::clear_pre_transformed_vertices()
{
    if(!get_context()->is_ray_tracing_supported())
        return;

    for(auto[dev, ptv]: pre_transformed_vertices)
    {
        if(ptv.count != 0)
        {
            ptv.buf.drop();
            ptv.count = 0;
        }
    }
}

void scene_stage::update_temporal_tables(uint32_t frame_index)
{
    forward_instance_ids.clear();
    forward_instance_ids.resize(prev_instance_count, 0xFFFFFFFFu);
    forward_point_light_ids.clear();
    forward_point_light_ids.resize(prev_point_light_count, 0xFFFFFFFFu);

    for(uint32_t new_id = 0; new_id < backward_instance_ids.size(); ++new_id)
    {
        uint32_t& prev_id = backward_instance_ids[new_id];
        if(prev_id < forward_instance_ids.size())
            forward_instance_ids[prev_id] = new_id;
        else prev_id = 0xFFFFFFFFu;
    }
    for(uint32_t new_id = 0; new_id < backward_point_light_ids.size(); ++new_id)
    {
        uint32_t& prev_id = backward_point_light_ids[new_id];
        if(prev_id < forward_point_light_ids.size())
            forward_point_light_ids[prev_id] = new_id;
        else prev_id = 0xFFFFFFFFu;
    }

    size_t alignment = get_device_mask().get_min_storage_buffer_alignment();
    size_t instance_backward_map_offset =  0;
    size_t instance_forward_map_offset = align_up_to(
        instance_backward_map_offset + std::max((size_t)1, backward_instance_ids.size()) * sizeof(uint32_t),
        alignment
    );
    size_t point_light_backward_map_offset = align_up_to(
        instance_forward_map_offset + std::max((size_t)1, forward_instance_ids.size()) * sizeof(uint32_t),
        alignment
    );
    size_t point_light_forward_map_offset = align_up_to(
        point_light_backward_map_offset + std::max((size_t)1, backward_point_light_ids.size()) * sizeof(uint32_t),
        alignment
    );
    size_t end_offset = point_light_forward_map_offset + std::max((size_t)1, forward_point_light_ids.size()) * sizeof(uint32_t);

    geometry_outdated |= temporal_tables.resize(end_offset);
    temporal_tables.update(frame_index, backward_instance_ids.data(), instance_backward_map_offset, backward_instance_ids.size() * sizeof(uint32_t));
    temporal_tables.update(frame_index, forward_instance_ids.data(), instance_forward_map_offset, forward_instance_ids.size() * sizeof(uint32_t));
    temporal_tables.update(frame_index, backward_point_light_ids.data(), point_light_backward_map_offset, backward_point_light_ids.size() * sizeof(uint32_t));
    temporal_tables.update(frame_index, forward_point_light_ids.data(), point_light_forward_map_offset, forward_point_light_ids.size() * sizeof(uint32_t));
}

void scene_stage::update(uint32_t frame_index)
{
    if(!cur_scene) return;

    environment_map* new_envmap = tr::get_environment_map(*cur_scene);
    bool envmap_outdated = false;
    if(new_envmap != envmap)
    {
        envmap = new_envmap;
        envmap_change_counter++;
        envmap_outdated = true;
    }

    vec3 new_ambient = tr::get_ambient_light(*cur_scene);
    if(ambient != new_ambient)
    {
        ambient = new_ambient;
        lights_outdated = true;
    }

    geometry_outdated |= refresh_instance_cache();
    track_shadow_maps(*cur_scene);

    uint64_t frame_counter = get_context()->get_frame_counter();
    cur_scene->foreach([&](model& mod){
        if(mod.has_joints_buffer())
            mod.update_joints(frame_index);
    });

    size_t tri_light_count = 0;
    size_t vertex_count = 0;

    if(geometry_outdated)
    {
        // TODO: This won't catch changing materials! The correct solution would
        // probably be to update the sampler table on every frame, but make it
        // faster than it currently is and report if there was a change.
        s_table.update_scene(this);
    }

    instance_data.resize(sizeof(instance_buffer) * instances.size());
    instance_data.foreach<instance_buffer>(
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
            if(opt.alloc_sh_grids && !get_sh_grid(*cur_scene, model[3], &index))
                get_largest_sh_grid(*cur_scene, &index);
            inst.sh_grid_index = index;
            inst.pad = 0;
            inst.shadow_terminator_mul = 1.0f/(
                1.0f-0.5f * instances[i].mod->get_shadow_terminator_offset()
            );

            const material& mat = *instances[i].mat;
            inst.mat.albedo_factor = mat.albedo_factor;
            inst.mat.metallic_roughness_factor =
                vec4(mat.metallic_factor, mat.roughness_factor, 0, 0);
            inst.mat.emission_factor = vec4(mat.emission_factor, 0.0f);
            inst.mat.flags =
                (mat.double_sided ? MATERIAL_FLAG_DOUBLE_SIDED : 0) |
                (mat.transient ? MATERIAL_FLAG_TRANSIENT : 0);
            inst.mat.transmittance = mat.transmittance;
            inst.mat.ior = mat.ior;
            inst.mat.normal_factor = mat.normal_factor;

            inst.mat.albedo_tex_id = s_table.find_tex_id(mat.albedo_tex);
            inst.mat.metallic_roughness_tex_id =
                s_table.find_tex_id(mat.metallic_roughness_tex);
            inst.mat.normal_tex_id = s_table.find_tex_id(mat.normal_tex);
            inst.mat.emission_tex_id = s_table.find_tex_id(mat.emission_tex);
        }
    );
    if(force_instance_refresh_frames > 0) force_instance_refresh_frames--;

    size_t sh_grid_count = cur_scene->count<sh_grid>();
    sh_grid_data.resize(sizeof(sh_grid_buffer) * sh_grid_count);
    sh_grid_data.map<sh_grid_buffer>(
        frame_index,
        [&](sh_grid_buffer* sh_data){
            size_t i =0;
            cur_scene->foreach([&](transformable& t, sh_grid& s) {
                sh_data[i].grid_clamp = 0.5f/vec3(s.get_resolution());
                sh_data[i].grid_resolution = s.get_resolution();
                mat4 transform = t.get_global_transform();
                quat orientation = get_matrix_orientation(transform);
                sh_data[i].pos_from_world = glm::affineInverse(transform);
                sh_data[i].normal_from_world = mat4(inverse(orientation));

                if(sh_grid_textures.count(&s) == 0)
                {
                    sh_grid_textures.emplace(&s, s.create_texture(get_device_mask()));
                    lights_outdated = true;
                }
                ++i;
            });
        }
    );

    //auto& si = cur_scene->scene_infos[dev->index];

    camera_data_offsets.clear();
    size_t start_offset = 0;
    std::vector<entity> camera_entities = get_sorted_cameras(*cur_scene);
    for(entity id: camera_entities)
    {
        camera* cam = cur_scene->get<camera>(id);
        size_t buf_size = camera::get_projection_type_uniform_buffer_size(cam->get_projection_type()) * 2;
        camera_data_offsets.push_back({start_offset, buf_size});
        start_offset += buf_size;
    }
    camera_data.resize(start_offset);
    old_camera_data.resize(start_offset);
    camera_data.map<uint8_t>(
        frame_index, [&](uint8_t* data){
            uint8_t* old_data = old_camera_data.data();
            size_t i = 0;
            for(entity id: camera_entities)
            {
                camera* cam = cur_scene->get<camera>(id);
                transformable* t = cur_scene->get<transformable>(id);
                uint8_t* cur_data = data + camera_data_offsets[i].first;
                size_t buf_size = camera::get_projection_type_uniform_buffer_size(cam->get_projection_type());
                cam->write_uniform_buffer(*t, cur_data);
                memcpy(cur_data + buf_size, old_data, buf_size);
                memcpy(old_data, cur_data, buf_size);
                old_data += buf_size;
                ++i;
            }
        }
    );

    if(opt.shadow_mapping)
    {
        lights_outdated |= update_shadow_map_params();

        shadow_map_range = sizeof(shadow_map_entry) * total_shadow_map_count;

        shadow_map_cascade_range =
            sizeof(shadow_map_cascade_entry) * total_cascade_count;

        shadow_map_data.resize(shadow_map_range + shadow_map_cascade_range);
        shadow_map_data.map<uint8_t>(
            frame_index, [&](uint8_t* sm_data){
            shadow_map_entry* shadow_map_data =
                reinterpret_cast<shadow_map_entry*>(sm_data);
            shadow_map_cascade_entry* shadow_map_cascade_data =
                reinterpret_cast<shadow_map_cascade_entry*>(
                    sm_data + shadow_map_range
                );

            int cascade_index = 0;
            for(const auto& sm: shadow_maps)
            {
                shadow_map_entry& map = shadow_map_data[sm.map_index];
                const auto& first_cam = sm.faces[0];

                map.projection_info = first_cam.cam.get_projection_info();
                map.range_radius = vec4(
                    first_cam.cam.get_near(),
                    first_cam.cam.get_far(),
                    sm.radius
                );

                // Determine shadow map type from projection
                switch(first_cam.cam.get_projection_type())
                {
                case camera::PERSPECTIVE:
                    { // Cubemap / perspective shadow map
                        if(sm.faces.size() == 6)
                        {
                            map.type = 1;
                            map.world_to_shadow = glm::inverse(
                                sm.faces[5].transform);
                        }
                        else
                        {
                            map.type = 0;
                            map.world_to_shadow = glm::inverse(
                                first_cam.transform);
                        }
                    }
                    break;
                case camera::ORTHOGRAPHIC:
                    { // Directional
                        map.type = sm.cascades.size();
                        map.cascade_index = cascade_index;
                        map.world_to_shadow = first_cam.cam.get_view_projection(first_cam.transform);
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
                    ivec2(shadow_atlas->get_rect_px(sm.atlas_index)),
                    sm.face_size
                )/vec4(
                    shadow_atlas->get_size(),
                    shadow_atlas->get_size()
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
                        ivec2(shadow_atlas->get_rect_px(c.atlas_index)),
                        sm.face_size
                    )/vec4(
                        shadow_atlas->get_size(),
                        shadow_atlas->get_size()
                    );

                    cascade_index++;
                }
            }
        });
    }

    prev_point_light_count = backward_point_light_ids.size();
    backward_point_light_ids.clear();

    size_t point_light_count = cur_scene->count<point_light>() + cur_scene->count<spotlight>();
    lights_outdated |= point_light_data.resize(sizeof(point_light_entry) * point_light_count);
    lights_outdated |= prev_point_light_data.resize(sizeof(point_light_entry) * prev_point_light_count);

    point_light_data.map<uint8_t>(frame_index, [&](uint8_t* light_data){
        point_light_entry* point_light_data =
            reinterpret_cast<point_light_entry*>(light_data);

        uint32_t i = 0;
        cur_scene->foreach([&](entity id, transformable& t, point_light& pl, temporal_light_data* td) {
            if(td)
            {
                backward_point_light_ids.push_back(td->prev_index);
                td->prev_index = i;
            }
            else
            {
                backward_point_light_ids.push_back(0xFFFFFFFFu);
                cur_scene->attach(id, temporal_light_data{i});
            }
            point_light_entry pe = point_light_entry(t, pl, get_shadow_map_index(&pl));
            point_light_data[i] = pe;
            ++i;
        });
        cur_scene->foreach([&](entity id, transformable& t, spotlight& sl, temporal_light_data* td) {
            if(td)
            {
                backward_point_light_ids.push_back(td->prev_index);
                td->prev_index = i;
            }
            else
            {
                backward_point_light_ids.push_back(0xFFFFFFFFu);
                cur_scene->attach(id, temporal_light_data{i});
            }
            point_light_entry pe = point_light_entry(t, sl, get_shadow_map_index(&sl));
            point_light_data[i] = pe;
            ++i;
        });
    });

    size_t directional_light_count = cur_scene->count<directional_light>();
    lights_outdated |= directional_light_data.resize(sizeof(directional_light_entry) * directional_light_count);
    directional_light_data.map<uint8_t>(frame_index, [&](uint8_t* light_data){
        directional_light_entry* directional_light_data =
            reinterpret_cast<directional_light_entry*>(light_data);

        // Punctual directional lights
        size_t i = 0;
        cur_scene->foreach([&](transformable& t, directional_light& dl) {
            directional_light_data[i] = directional_light_entry(t, dl, get_shadow_map_index(&dl));
            ++i;
        });
    });

    if(opt.gather_emissive_triangles)
        lights_outdated |= tri_light_data.resize(tri_light_count * sizeof(tri_light_entry));
    else
        tri_light_data.resize(0);

    scene_metadata.map<scene_metadata_buffer>(
        frame_index, [&](scene_metadata_buffer* data){
            data->instance_count = instances.size();
            data->point_light_count = point_light_count;
            data->directional_light_count = directional_light_count;
            data->tri_light_count = tri_light_count;
            if(envmap)
            {
                data->environment_factor = vec4(envmap->get_factor(), 1);
                data->environment_proj = (int)envmap->get_projection();
            }
            else
            {
                data->environment_factor = vec4(0);
                data->environment_proj = -1;
            }
        }
    );

    size_t light_aabb_count = 0;
    if(get_context()->is_ray_tracing_supported())
    {
        light_aabb_buffer.map<vk::AabbPositionsKHR>(
            frame_index,
            [&](vk::AabbPositionsKHR* aabb){
                size_t i = 0;

                cur_scene->foreach([&](transformable& t, point_light& pl){
                    if(i >= opt.max_lights) return;

                    float radius = pl.get_radius();
                    vec3 pos = radius == 0.0f ? vec3(0) : t.get_global_position();
                    vec3 min = pos - vec3(radius);
                    vec3 max = pos + vec3(radius);
                    aabb[i] = vk::AabbPositionsKHR(
                        min.x, min.y, min.z, max.x, max.y, max.z);
                    i++;
                });
                cur_scene->foreach([&](transformable& t, spotlight& sl){
                    if(i >= opt.max_lights) return;
                    float radius = sl.get_radius();
                    vec3 pos = radius == 0.0f ? vec3(0) : t.get_global_position();
                    vec3 min = pos - vec3(radius);
                    vec3 max = pos + vec3(radius);
                    aabb[i] = vk::AabbPositionsKHR(
                        min.x, min.y, min.z, max.x, max.y, max.z);
                    i++;
                });
                light_aabb_count = i;
            }
        );

        for(device& dev: get_device_mask())
        {
            if(geometry_outdated)
                ensure_blas();

            // Run BLAS matrix updates. Only necessary when merged BLASes have dynamic
            // transformables.
            if(group_strategy == blas_strategy::ALL_MERGED_STATIC)
            {
                size_t offset = 0;
                std::vector<bottom_level_acceleration_structure::entry> entries;
                for(const instance_group& group: group_cache)
                {
                    entries.clear();
                    for(size_t i = 0; i < group.size; ++i, ++offset)
                    {
                        const instance& inst = instances[offset];
                        entries.push_back({
                            inst.m,
                            0, nullptr,
                            group.static_transformable ? inst.transform : mat4(1),
                            !inst.mat->potentially_transparent()
                        });
                    }
                    blas_cache.at(group.id).update_transforms(frame_index, entries);
                }
            }

            auto& instance_buffer = tlas->get_instances_buffer();

            as_instance_count = 0;
            uint32_t total_max_capacity = opt.max_instances + opt.max_lights;
            instance_buffer.map_one<vk::AccelerationStructureInstanceKHR>(
                dev.id,
                frame_index,
                [&](vk::AccelerationStructureInstanceKHR* as_instances){
                    // Update instance staging buffer
                    size_t offset = 0;
                    for(size_t j = 0; j < group_cache.size() && as_instance_count < total_max_capacity; ++j)
                    {
                        const instance_group& group = group_cache[j];
                        const bottom_level_acceleration_structure& blas = blas_cache.at(group.id);
                        vk::AccelerationStructureInstanceKHR inst = vk::AccelerationStructureInstanceKHR(
                            {}, offset, 1<<0, 0, // Hit group 0 for triangle meshes.
                            {}, blas.get_blas_address(dev.id)
                        );
                        if(!blas.is_backface_culled())
                            inst.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

                        mat4 global_transform = group.static_transformable ?
                            mat4(1) : transpose(instances[offset].transform);
                        memcpy(
                            (void*)&inst.transform,
                            (void*)&global_transform,
                            sizeof(inst.transform)
                        );
                        as_instances[as_instance_count++] = inst;
                        offset += group.size;
                    }

                    if(light_aabb_count != 0 && as_instance_count < total_max_capacity && light_blas.has_value())
                    {
                        vk::AccelerationStructureInstanceKHR& inst = as_instances[as_instance_count++];
                        inst = vk::AccelerationStructureInstanceKHR(
                            {}, as_instance_count, 1<<1, 2,
                            vk::GeometryInstanceFlagBitsKHR::eTriangleCullDisable,
                            light_blas->get_blas_address(dev.id)
                        );
                        mat4 id_mat = mat4(1.0f);
                        memcpy((void*)&inst.transform, (void*)&id_mat, sizeof(inst.transform));
                    }
                }
            );
        }

        if(opt.pre_transform_vertices)
            geometry_outdated |= reserve_pre_transformed_vertices(vertex_count);
        else
            clear_pre_transformed_vertices();
    }

    update_temporal_tables(frame_index);
    if(lights_outdated) light_change_counter++;
    if(geometry_outdated) geometry_change_counter++;

    if(lights_outdated || geometry_outdated || envmap_outdated)
        update_descriptor_set();

    if(lights_outdated || geometry_outdated)
    {
        record_command_buffers(light_aabb_count, true);
        prev_was_rebuild = true;
        lights_outdated = false;
        geometry_outdated = false;
    }
    else if(prev_was_rebuild)
    {
        record_command_buffers(light_aabb_count, false);
        prev_was_rebuild = false;
    }
}

void scene_stage::record_command_buffers(size_t light_aabb_count, bool rebuild_as)
{
    clear_commands();

    for(device& dev: get_device_mask())
    {
        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vk::CommandBuffer cb = begin_graphics(dev.id);
            stage_timer.begin(cb, dev.id, i);

            // Copy over previous light data.
            if(prev_point_light_count != 0)
            {
                cb.copyBuffer(point_light_data[dev.id], prev_point_light_data[dev.id], vk::BufferCopy{0, 0, prev_point_light_count * sizeof(point_light_entry)});
            }

            instance_data.upload(dev.id, i, cb);
            directional_light_data.upload(dev.id, i, cb);
            point_light_data.upload(dev.id, i, cb);
            sh_grid_data.upload(dev.id, i, cb);
            shadow_map_data.upload(dev.id, i, cb);
            camera_data.upload(dev.id, i, cb);
            scene_metadata.upload(dev.id, i, cb);
            light_aabb_buffer.upload(dev.id, i, cb);
            temporal_tables.upload(dev.id, i, cb);

            bulk_upload_barrier(cb, vk::PipelineStageFlagBits::eComputeShader);

            record_skinning(dev.id, i, cb);
            if(dev.ctx->is_ray_tracing_supported())
            {
                record_as_build(dev.id, i, cb, light_aabb_count, rebuild_as);
                if(opt.pre_transform_vertices)
                    record_pre_transform(dev.id, cb);
                if(tri_light_data.get_size() != 0)
                    record_tri_light_extraction(dev.id, cb);
            }

            stage_timer.end(cb, dev.id, i);
            end_graphics(cb, dev.id, i);
        }
    }
}

void scene_stage::record_skinning(device_id id, uint32_t frame_index, vk::CommandBuffer cb)
{
    skinning[id].bind(cb);

    // Update vertex buffers
    cur_scene->foreach([&](model& mod){
        if(!mod.has_joints_buffer()) return;

        mod.upload_joints(cb, id, frame_index);
        for(auto& vg: mod)
        {
            mesh* dst = vg.m;
            mesh* src = dst->get_animation_source();
            uint32_t vertex_count = vg.m->get_vertices().size();

            skinning[id].push_constants(cb, skinning_push_constants{vertex_count});
            skinning_desc.set_buffer(id, "source_data", {{src->get_vertex_buffer(id), 0, VK_WHOLE_SIZE}});
            skinning_desc.set_buffer(id, "destination_data", {{dst->get_vertex_buffer(id), 0, VK_WHOLE_SIZE}});
            skinning_desc.set_buffer(id, "skin_data", {{src->get_skin_buffer(id), 0, VK_WHOLE_SIZE}});
            skinning_desc.set_buffer(id, "joint_data", {{mod.get_joint_buffer()[id], 0, VK_WHOLE_SIZE}});
            skinning[id].push_descriptors(cb, skinning_desc, 0);
            cb.dispatch((vertex_count+31u)/32u, 1, 1);
        }
    });

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

        // Run BLAS updates
        size_t offset = 0;
        std::vector<bottom_level_acceleration_structure::entry> entries;
        for(const instance_group& group: group_cache)
        {
            if(group.static_mesh)
            {
                offset += group.size;
                continue;
            }

            entries.clear();
            for(size_t i = 0; i < group.size; ++i, ++offset)
            {
                const instance& inst = instances[offset];
                entries.push_back({
                    inst.m,
                    0, nullptr,
                    group.static_transformable ? inst.transform : mat4(1),
                    !inst.mat->potentially_transparent()
                });
            }
            blas_cache.at(group.id).rebuild(
                id, frame_index, cb, entries,
                group_strategy == blas_strategy::ALL_MERGED_STATIC ? false : true
            );
        }
    }
}

void scene_stage::record_as_build(
    device_id id,
    uint32_t frame_index,
    vk::CommandBuffer cb,
    size_t light_aabb_count,
    bool rebuild
){
    auto& instance_buffer = tlas->get_instances_buffer();
    bool as_update = !rebuild;

    if(opt.track_prev_tlas)
        prev_tlas->copy(id, *tlas, cb);

    if(light_blas.has_value())
    {
        light_blas->rebuild(
            id,
            frame_index,
            cb,
            {bottom_level_acceleration_structure::entry{nullptr, light_aabb_count, &light_aabb_buffer, mat4(1.0f), true}},
            as_update
        );
    }

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

    tlas->rebuild(id, cb, as_instance_count, as_update);
}

void scene_stage::record_tri_light_extraction(
    device_id id,
    vk::CommandBuffer cb
){
    extract_tri_lights[id].bind(cb);
    extract_tri_lights[id].set_descriptors(cb, scene_desc, 0, 0);
    for(size_t i = 0; i < instances.size(); ++i)
    {
        const instance& inst = instances[i];
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
    vk::Buffer ptv_buf = this->pre_transformed_vertices[id].buf;
    pre_transform[id].bind(cb);
    pre_transform[id].set_descriptors(cb, scene_desc, 0, 1);

    size_t offset = 0;
    for(size_t i = 0; i < instances.size(); ++i)
    {
        const instance& inst = instances[i];

        pre_tranform_push_constants pc;
        pc.vertex_count = inst.m->get_vertices().size();
        pc.instance_id = i;

        size_t bytes = pc.vertex_count * sizeof(mesh::vertex);

        pre_transform_desc.set_buffer(id, "input_verts", {{inst.m->get_vertex_buffer(id), 0, bytes}});
        pre_transform_desc.set_buffer(id, "output_verts", {{ptv_buf, offset, bytes}});
        pre_transform[id].push_descriptors(cb, pre_transform_desc, 0);
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
                {}, {}, ptv_buf, 0, VK_WHOLE_SIZE
            },
        }, {}
    );
}

void scene_stage::init_descriptor_set_layout()
{
    scene_desc.add("instances", {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    scene_desc.add("directional_lights", {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    scene_desc.add("point_lights", {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    scene_desc.add("tri_lights", {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    scene_desc.add("vertices", {4, vk::DescriptorType::eStorageBuffer, opt.max_instances, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    scene_desc.add("indices", {5, vk::DescriptorType::eStorageBuffer, opt.max_instances, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    scene_desc.add("textures", {6, vk::DescriptorType::eCombinedImageSampler, opt.max_samplers, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    scene_desc.add("environment_map_tex", {7, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    scene_desc.add("environment_map_alias_table", {8, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    scene_desc.add("scene_metadata", {9, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    scene_desc.add("camera", {10, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);

    device_mask dev = get_device_mask();
    if(dev.get_context()->is_ray_tracing_supported())
        scene_desc.add("tlas", {11, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eAll, nullptr});

    scene_raster_desc.add("sh_grid_data", {0, vk::DescriptorType::eCombinedImageSampler, opt.max_3d_samplers, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    scene_raster_desc.add("sh_grids", {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    scene_raster_desc.add("shadow_maps", {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    scene_raster_desc.add("shadow_map_cascades", {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    scene_raster_desc.add("shadow_map_atlas", {4, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    scene_raster_desc.add("shadow_map_atlas_test", {5, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    scene_raster_desc.add("pcf_noise_vector_2d", {6, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    scene_raster_desc.add("pcf_noise_vector_3d", {7, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    scene_raster_desc.add("brdf_integration", {8, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eAll, nullptr});

    temporal_tables_desc.add("instance_forward_map", {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    temporal_tables_desc.add("instance_backward_map", {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    temporal_tables_desc.add("point_light_forward_map", {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    temporal_tables_desc.add("point_light_backward_map", {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    temporal_tables_desc.add("prev_point_lights", {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    if(dev.get_context()->is_ray_tracing_supported())
        temporal_tables_desc.add("prev_tlas", {5, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eAll, nullptr});
}

void scene_stage::update_descriptor_set()
{
    scene_desc.reset(scene_desc.get_mask(), 1);
    scene_desc.set_buffer(0, "instances", instance_data);
    scene_desc.set_buffer(0, "directional_lights", directional_light_data);
    scene_desc.set_buffer(0, "point_lights", point_light_data);
    scene_desc.set_buffer(0, "tri_lights", tri_light_data);
    scene_desc.set_buffer(0, "scene_metadata", scene_metadata);
    scene_desc.set_buffer(0, "camera", camera_data);

    if(envmap)
        scene_desc.set_texture(0, "environment_map_tex", *envmap, envmap_sampler);

    for(device& dev: scene_desc.get_mask())
    {
        device_id id = dev.id;
        std::vector<vk::DescriptorBufferInfo> dbi_vertex;
        std::vector<vk::DescriptorBufferInfo> dbi_index;
        bool got_pre_transformed_vertices = false;
        if(get_context()->is_ray_tracing_supported())
        {
            auto& ptv = pre_transformed_vertices[id];
            if(ptv.count != 0)
            {
                size_t offset = 0;
                for(size_t i = 0; i < instances.size(); ++i)
                {
                    const mesh* m = instances[i].m;
                    size_t bytes = m->get_vertices().size() * sizeof(mesh::vertex);
                    dbi_vertex.push_back({ptv.buf, offset, bytes});
                    offset += bytes;
                }
                got_pre_transformed_vertices = true;
            }
        }
        for(size_t i = 0; i < instances.size(); ++i)
        {
            const mesh* m = instances[i].m;
            if(!got_pre_transformed_vertices)
            {
                vk::Buffer vertex_buffer = m->get_vertex_buffer(id);
                dbi_vertex.push_back({vertex_buffer, 0, VK_WHOLE_SIZE});
            }
            vk::Buffer index_buffer = m->get_index_buffer(id);
            dbi_index.push_back({index_buffer, 0, VK_WHOLE_SIZE});
        }

        scene_desc.set_buffer(id, 0, "vertices", std::move(dbi_vertex));
        scene_desc.set_buffer(id, 0, "indices", std::move(dbi_index));

        std::vector<vk::DescriptorImageInfo> dii = s_table.get_image_infos(id);
        scene_desc.set_image(id, 0, "textures", std::move(dii));

        if(envmap)
        {
            scene_desc.set_buffer(id, 0, "environment_map_alias_table", {{
                envmap ? envmap->get_alias_table(id) : vk::Buffer{}, 0, VK_WHOLE_SIZE
            }});
        }

        if(dev.ctx->is_ray_tracing_supported())
        {
            scene_desc.set_acceleration_structure(id, 0, "tlas", *this->tlas->get_tlas_handle(id));
        }
    }

    placeholders& pl = get_device_mask().get_context()->get_placeholders();

    scene_raster_desc.reset(scene_raster_desc.get_mask(), 1);
    scene_raster_desc.set_buffer(0, "sh_grids", sh_grid_data);

    if(opt.shadow_mapping)
    {
        scene_raster_desc.set_buffer(0, "shadow_maps", shadow_map_data);
        scene_raster_desc.set_buffer(0, "shadow_map_cascades", shadow_map_data, shadow_map_range);
        scene_raster_desc.set_texture(0, "shadow_map_atlas", *shadow_atlas, pl.default_sampler);
        scene_raster_desc.set_texture(0, "shadow_map_atlas_test", *shadow_atlas, shadow_sampler);
    }

    scene_raster_desc.set_texture(0, "pcf_noise_vector_2d", noise_vector_2d, pl.default_sampler);
    scene_raster_desc.set_texture(0, "pcf_noise_vector_3d", noise_vector_3d, pl.default_sampler);
    scene_raster_desc.set_texture(0, "brdf_integration", brdf_integration, brdf_integration_sampler);

    for(device& dev: scene_desc.get_mask())
    {
        device_id id = dev.id;

        std::vector<vk::DescriptorImageInfo> dii_3d;
        if(opt.alloc_sh_grids)
        {
            cur_scene->foreach([&](sh_grid& sg){
                const texture& tex = sh_grid_textures.at(&sg);
                dii_3d.push_back({
                    sh_grid_sampler.get_sampler(id),
                    tex.get_image_view(id),
                    vk::ImageLayout::eShaderReadOnlyOptimal
                });
            });
        }
        scene_raster_desc.set_image(id, 0, "sh_grid_data", std::move(dii_3d));
    }

    temporal_tables_desc.reset(temporal_tables_desc.get_mask(), 1);
    size_t alignment = get_device_mask().get_min_storage_buffer_alignment();
    size_t instance_backward_map_offset =  0;
    size_t instance_forward_map_offset = align_up_to(
        instance_backward_map_offset + std::max((size_t)1, backward_instance_ids.size()) * sizeof(uint32_t),
        alignment
    );
    size_t point_light_backward_map_offset = align_up_to(
        instance_forward_map_offset + std::max((size_t)1, forward_instance_ids.size()) * sizeof(uint32_t),
        alignment
    );
    size_t point_light_forward_map_offset = align_up_to(
        point_light_backward_map_offset + std::max((size_t)1, backward_point_light_ids.size()) * sizeof(uint32_t),
        alignment
    );
    temporal_tables_desc.set_buffer(0, "instance_forward_map", temporal_tables, instance_forward_map_offset);
    temporal_tables_desc.set_buffer(0, "instance_backward_map", temporal_tables, instance_backward_map_offset);
    temporal_tables_desc.set_buffer(0, "point_light_forward_map", temporal_tables, point_light_forward_map_offset);
    temporal_tables_desc.set_buffer(0, "point_light_backward_map", temporal_tables, point_light_backward_map_offset);
    temporal_tables_desc.set_buffer(0, "prev_point_lights", prev_point_light_data);
    for(device& dev: scene_desc.get_mask())
    {
        if(dev.ctx->is_ray_tracing_supported())
        {
            if(opt.track_prev_tlas)
                temporal_tables_desc.set_acceleration_structure(dev.id, 0, "prev_tlas", *this->prev_tlas->get_tlas_handle(dev.id));
            else
                temporal_tables_desc.set_acceleration_structure(dev.id, 0, "prev_tlas", *this->tlas->get_tlas_handle(dev.id));
        }
    }
}

}
