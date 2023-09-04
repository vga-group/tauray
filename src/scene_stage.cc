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
    light_bvh_outdated(true),
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
    scene_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    scene_metadata(dev, sizeof(scene_metadata_buffer), vk::BufferUsageFlagBits::eUniformBuffer),
    directional_light_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    point_light_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    tri_light_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc),
    sh_grid_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    shadow_map_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    camera_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    light_bvh_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    light_bit_trail_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
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
    return instances;
}

const std::unordered_map<sh_grid*, texture>& scene_stage::get_sh_grid_textures() const
{
    return sh_grid_textures;
}

vk::AccelerationStructureKHR scene_stage::get_acceleration_structure(
    device_id id
) const {
    device_mask dev = get_device_mask();
    if(!dev.get_context()->is_ray_tracing_supported())
        throw std::runtime_error(
            "Trying to use TLAS, but ray tracing is not available!"
        );
    return *tlas->get_tlas_handle(id);
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
    size_t i = 0;
    entity last_object_id = INVALID_ENTITY;
    group_cache.clear();
    bool scene_changed = false;

    auto add_instances = [&](bool static_mesh, bool static_transformable){
        cur_scene->foreach([&](entity id, transformable& t, model& mod){
            // If requesting dynamic meshes, we don't care about the
            // transformable staticness any more.
            if(static_mesh && static_transformable != t.is_static())
                return;

            bool fetched_transforms = false;
            mat4 transform;
            mat4 normal_transform;
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
                    if(!fetched_transforms)
                    {
                        transform = t.get_global_transform();
                        normal_transform = t.get_global_inverse_transpose_transform();
                        fetched_transforms = true;
                    }

                    if(inst.prev_transform != inst.transform)
                    {
                        inst.prev_transform = inst.transform;
                        inst.last_refresh_frame = frame_counter;
                    }
                    if(inst.transform != transform)
                    {
                        inst.transform = transform;
                        inst.normal_transform = normal_transform;
                        inst.last_refresh_frame = frame_counter;
                    }
                }
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

std::vector<descriptor_state> scene_stage::get_descriptor_info(device_id id, int32_t camera_index) const
{
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

    vk::AccelerationStructureKHR tlas = {};
    device_mask dev = get_device_mask();
    if(dev.get_context()->is_ray_tracing_supported())
        tlas = get_acceleration_structure(id);

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

    std::vector<vk::DescriptorImageInfo> dii = s_table.get_image_infos(id);

    std::vector<descriptor_state> descriptors = {
        {"scene", {scene_data[id], 0, VK_WHOLE_SIZE}},
        {"scene_metadata", {scene_metadata[id], 0, VK_WHOLE_SIZE}},
        {"vertices", dbi_vertex},
        {"indices", dbi_index},
        {"textures", dii},
        {"directional_lights", {
            directional_light_data[id], 0, VK_WHOLE_SIZE
        }},
        {"point_lights", {point_light_data[id], 0, VK_WHOLE_SIZE}},
        {"tri_lights", {tri_light_data[id], 0, VK_WHOLE_SIZE}},
        {"environment_map_tex", {
            envmap_sampler.get_sampler(id),
            envmap ? envmap->get_image_view(id) : vk::ImageView{},
            vk::ImageLayout::eShaderReadOnlyOptimal
        }},
        {"environment_map_alias_table", {
            envmap ? envmap->get_alias_table(id) : vk::Buffer{}, 0, VK_WHOLE_SIZE
        }},
        {"textures3d", dii_3d},
        {"sh_grids", {sh_grid_data[id], 0, VK_WHOLE_SIZE}},
        {"light_bvh", {light_bvh_data[id], 0, VK_WHOLE_SIZE}},
        {"light_bit_trail", {light_bit_trail_data[id], 0, VK_WHOLE_SIZE}}
    };

    if(camera_index >= 0)
    {
        std::pair<size_t, size_t> camera_offset = camera_data_offsets[camera_index];
        descriptors.push_back(
            {"camera", {camera_data[id], camera_offset.first, VK_WHOLE_SIZE}}
        );
    }

    if(dev.get_context()->is_ray_tracing_supported())
    {
        descriptors.push_back({"tlas", {1, this->tlas->get_tlas_handle(id)}});
    }

    if(opt.shadow_mapping)
    {
        placeholders& pl = dev.get_context()->get_placeholders();

        descriptors.push_back(
            {"shadow_maps", {shadow_map_data[id], 0, shadow_map_range}}
        );
        descriptors.push_back(
            {"shadow_map_cascades", {
                shadow_map_data[id], shadow_map_range,
                shadow_map_cascade_range
            }}
        );
        descriptors.push_back(
            {"shadow_map_atlas", {
                pl.default_sampler.get_sampler(id),
                shadow_atlas->get_image_view(id),
                vk::ImageLayout::eShaderReadOnlyOptimal
            }}
        );
        descriptors.push_back(
            {"shadow_map_atlas_test", {
                shadow_sampler.get_sampler(id),
                shadow_atlas->get_image_view(id),
                vk::ImageLayout::eShaderReadOnlyOptimal
            }}
        );
    }
    return descriptors;
}

void scene_stage::bind(basic_pipeline& pipeline, uint32_t frame_index, int32_t camera_index)
{
    device* dev = pipeline.get_device();
    std::vector<descriptor_state> descriptors = get_descriptor_info(dev->id, camera_index);
    pipeline.update_descriptor_set(descriptors, frame_index);
}

void scene_stage::push(basic_pipeline& pipeline, vk::CommandBuffer cmd, int32_t camera_index)
{
    device* dev = pipeline.get_device();
    std::vector<descriptor_state> descriptors = get_descriptor_info(dev->id, camera_index);
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
            pl.default_sampler.get_sampler(dev->id),
            pl.depth_test_sample.get_image_view(dev->id),
            vk::ImageLayout::eShaderReadOnlyOptimal
        }},
        {"textures3d", {
            pl.default_sampler.get_sampler(dev->id),
            pl.sample3d.get_image_view(dev->id),
            vk::ImageLayout::eShaderReadOnlyOptimal
        }, max_3d_samplers}
    });
}

void scene_stage::update(uint32_t frame_index)
{
    if(!cur_scene) return;

    environment_map* new_envmap = tr::get_environment_map(*cur_scene);
    if(new_envmap != envmap)
    {
        envmap = new_envmap;
        envmap_change_counter++;
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

    tri_light_count = 0;
    size_t vertex_count = 0;

    if(geometry_outdated)
    {
        // TODO: This won't catch changing materials! The correct solution would
        // probably be to update the sampler table on every frame, but make it
        // faster than it currently is and report if there was a change.
        s_table.update_scene(this);
    }

    scene_data.resize(sizeof(instance_buffer) * instances.size());
    scene_data.foreach<instance_buffer>(
        frame_index, instances.size(),
        [&](instance_buffer& inst, size_t i){
            if(instances[i].mat->emission_factor != vec3(0))
            {
                inst.light_base_id = tri_light_count;
                tri_light_count += instances[i].m->get_indices().size() / 3;

                if(instances[i].mod->has_joints_buffer())
                    light_bvh_outdated = true;
            }
            else inst.light_base_id = -1;

            vertex_count += instances[i].m->get_indices().size();

            // Skip unchanged instances.
            if(
                force_instance_refresh_frames == 0 &&
                instances[i].last_refresh_frame+MAX_FRAMES_IN_FLIGHT < frame_counter
            ) return;

            if(instances[i].mat->emission_factor != vec3(0) && instances[i].transform != instances[i].prev_transform)
                light_bvh_outdated = true;

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
            inst.mat.emission_factor_double_sided = vec4(
                mat.emission_factor, mat.double_sided ? 1.0f : 0.0f
            );
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

                map.clip_info = vec4(
                    first_cam.cam.get_clip_info(),
                    first_cam.cam.get_near()
                );
                map.projection_info_radius = vec4(
                    first_cam.cam.get_projection_info(), sm.radius
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
                        map.clip_info.z = first_cam.cam.get_far();
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

    size_t point_light_count = cur_scene->count<point_light>() + cur_scene->count<spotlight>();
    lights_outdated |= point_light_data.resize(sizeof(point_light_entry) * point_light_count);
    point_light_data.map<uint8_t>(frame_index, [&](uint8_t* light_data){
        point_light_entry* point_light_data =
            reinterpret_cast<point_light_entry*>(light_data);

        size_t i = 0;
        cur_scene->foreach([&](transformable& t, point_light& pl) {
            point_light_entry pe = point_light_entry(t, pl, get_shadow_map_index(&pl));
            point_light_data[i] = pe;
            ++i;
        });
        cur_scene->foreach([&](transformable& t, spotlight& sl) {
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
        lights_outdated |= tri_light_data.resize(tri_light_count * sizeof(gpu_tri_light));
    else
        tri_light_data.resize(0);

    scene_metadata.map<scene_metadata_buffer>(
        frame_index, [&](scene_metadata_buffer* data){
            data->point_light_count = point_light_count;
            data->directional_light_count = directional_light_count;
            data->tri_light_count = tri_light_count;
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

    if(lights_outdated) light_change_counter++;
    if(geometry_outdated) geometry_change_counter++;

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

void scene_stage::post_submit(uint32_t frame_index)
{
    if(light_bvh_outdated)
    {
        light_bvh.build(tri_light_count, tri_light_data);
        light_bvh_outdated = false;

        light_bvh_data.resize(light_bvh.get_gpu_bvh_size());
        light_bvh_data.map<gpu_light_bvh>(
            frame_index,
            [&](gpu_light_bvh* bvh){
                light_bvh.get_gpu_bvh_data(bvh);
            }
        );
        light_bit_trail_data.resize(light_bvh.get_gpu_bit_trail_size());
        light_bit_trail_data.map<uint32_t>(
            frame_index,
            [&](uint32_t* bit_trail){
                light_bvh.get_gpu_bit_trail_data(bit_trail);
            }
        );

        for(device& d: get_device_mask())
        {
            vk::CommandBuffer cmd = begin_command_buffer(d);
            light_bvh_data.upload(d.id, frame_index, cmd);
            light_bit_trail_data.upload(d.id, frame_index, cmd);
            end_command_buffer(d, cmd);
        }

        TR_LOG("Built and uploaded light BVH ", light_bvh.get_gpu_bvh_size());
        light_change_counter++;
    }
}

void scene_stage::record_command_buffers(size_t light_aabb_count, bool rebuild_as)
{
    clear_commands();

    for(device& dev: get_device_mask())
    {
        if(opt.gather_emissive_triangles)
        {
            extract_tri_lights[dev.id].reset_descriptor_sets();
            extract_tri_lights[dev.id].update_descriptor_set({
                {"textures", opt.max_samplers}
            });
            bind(extract_tri_lights[dev.id], 0, 0);
        }

        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vk::CommandBuffer cb = begin_graphics(dev.id);
            stage_timer.begin(cb, dev.id, i);
            scene_data.upload(dev.id, i, cb);
            directional_light_data.upload(dev.id, i, cb);
            point_light_data.upload(dev.id, i, cb);
            sh_grid_data.upload(dev.id, i, cb);
            shadow_map_data.upload(dev.id, i, cb);
            camera_data.upload(dev.id, i, cb);
            scene_metadata.upload(dev.id, i, cb);
            light_aabb_buffer.upload(dev.id, i, cb);

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
            skinning[id].push_descriptors(cb, {
                {"source_data", {src->get_vertex_buffer(id), 0, VK_WHOLE_SIZE}},
                {"destination_data", {dst->get_vertex_buffer(id), 0, VK_WHOLE_SIZE}},
                {"skin_data", {src->get_skin_buffer(id), 0, VK_WHOLE_SIZE}},
                {"joint_data", {mod.get_joint_buffer()[id], 0, VK_WHOLE_SIZE}}
            });
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
    extract_tri_lights[id].bind(cb, 0);
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
    size_t offset = 0;
    for(size_t i = 0; i < instances.size(); ++i)
    {
        const instance& inst = instances[i];

        pre_tranform_push_constants pc;
        pc.vertex_count = inst.m->get_vertices().size();
        pc.instance_id = i;

        size_t bytes = pc.vertex_count * sizeof(mesh::vertex);

        pre_transform[id].push_descriptors(cb, {
            {"input_verts", {inst.m->get_vertex_buffer(id), 0, bytes}},
            {"output_verts", {ptv_buf, offset, bytes}},
            {"scene", {scene_data[id], 0, VK_WHOLE_SIZE}}
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
                {}, {}, ptv_buf, 0, VK_WHOLE_SIZE
            },
        }, {}
    );
}

}
