#ifndef TAURAY_SCENE_STAGE_HH
#define TAURAY_SCENE_STAGE_HH
#include "scene.hh"
#include "mesh.hh"
#include "stage.hh"
#include "light.hh"
#include "model.hh"
#include "compute_pipeline.hh"
#include "radix_sort.hh"
#include "sampler_table.hh"
#include "sampler.hh"
#include "timer.hh"
#include "atlas.hh"
#include "camera.hh"
#include "descriptor_set.hh"

namespace tr
{

enum class blas_strategy
{
    PER_MATERIAL,
    PER_MODEL,
    STATIC_MERGED_DYNAMIC_PER_MODEL,
    ALL_MERGED_STATIC
};

class scene_stage: public multi_device_stage
{
public:
    struct options
    {
        // REFACTOR: Delete all max_instances & max_lights from everywhere
        // other than here.
        uint32_t max_instances = 1024;
        uint32_t max_lights = 128;
        uint32_t max_samplers = 128;
        uint32_t max_3d_samplers = 128;
        bool gather_emissive_triangles = false;
        bool pre_transform_vertices = false;
        bool shadow_mapping = false;
        bool alloc_sh_grids = false;
        blas_strategy group_strategy = blas_strategy::STATIC_MERGED_DYNAMIC_PER_MODEL;
    };

    scene_stage(device_mask dev, const options& opt);

    void set_scene(scene* target);
    scene* get_scene() const;

    // Update categories, these are used for checking if individual aspects of
    // the scene have changed. This is needed so that stages can update their
    // command buffers on-demand.
    static inline constexpr uint32_t ENVMAP = 1<<0;
    static inline constexpr uint32_t GEOMETRY = 1<<1;
    static inline constexpr uint32_t LIGHT = 1<<2;

    bool check_update(uint32_t categories, uint32_t& prev_counter) const;

    environment_map* get_environment_map() const;
    vec3 get_ambient() const;

    struct instance
    {
        mat4 transform;
        mat4 prev_transform;
        mat4 normal_transform;
        const material* mat;
        const mesh* m;
        const model* mod;
        uint64_t last_refresh_frame;
    };
    const std::vector<instance>& get_instances() const;

    const std::unordered_map<sh_grid*, texture>& get_sh_grid_textures() const;

    vk::AccelerationStructureKHR get_acceleration_structure(
        device_id id
    ) const;

    descriptor_set& get_descriptors();
    void bind(basic_pipeline& pipeline, uint32_t frame_index, int32_t camera_offset = 0);
    static void bind_placeholders(
        basic_pipeline& pipeline,
        size_t max_samplers,
        size_t max_3d_samplers
    );

    struct shadow_map_instance
    {
        unsigned atlas_index;
        unsigned map_index;
        uvec2 face_size;
        float min_bias;
        float max_bias;
        vec2 radius;
        struct cam_transform
        {
            camera cam;
            mat4 transform;
        };
        std::vector<cam_transform> faces;
        struct cascade
        {
            unsigned atlas_index;
            vec2 offset;
            float scale;
            float bias_scale;
            cam_transform cam;
        };
        std::vector<cascade> cascades;
    };
    vec2 get_shadow_map_atlas_pixel_margin() const;
    const std::vector<shadow_map_instance>& get_shadow_maps() const;
    atlas* get_shadow_map_atlas() const;

protected:
    void update(uint32_t frame_index) override;

private:
    void record_command_buffers(size_t light_aabb_count, bool rebuild_as);
    void record_skinning(device_id id, uint32_t frame_index, vk::CommandBuffer cb);
    void record_as_build(device_id id, uint32_t frame_index, vk::CommandBuffer cb, size_t light_aabb_count, bool rebuild);
    void record_tri_light_extraction(device_id id, vk::CommandBuffer cb);
    void record_pre_transform(device_id id, vk::CommandBuffer cb);

    void init_descriptor_set_layout();
    void update_descriptor_set();
    std::vector<descriptor_state> get_descriptor_info(device_id id, int32_t camera_index) const;

    bool prev_was_rebuild;
    size_t as_instance_count;

    uint32_t envmap_change_counter;
    uint32_t geometry_change_counter;
    uint32_t light_change_counter;
    bool geometry_outdated;
    bool lights_outdated;

    unsigned force_instance_refresh_frames;
    scene* cur_scene;

    //==========================================================================
    // Light stuff
    //==========================================================================
    environment_map* envmap;
    vec3 ambient;
    std::optional<bottom_level_acceleration_structure> light_blas;
    gpu_buffer light_aabb_buffer;

    //==========================================================================
    // Mesh stuff
    //==========================================================================

    // For acceleration structures, instances are grouped by which ones go into
    // the same BLAS. If two groups share the same ID, they will have the same
    // acceleration structure as well, but are inserted as separate TLAS
    // instances still.
    struct instance_group
    {
        uint64_t id = 0;
        size_t size = 0;
        bool static_mesh = false;
        bool static_transformable = false;
    };

    struct pre_transformed_data
    {
        uint32_t count = 0;
        vkm<vk::Buffer> buf;
    };
    per_device<pre_transformed_data> pre_transformed_vertices;
    std::unordered_map<uint64_t, bottom_level_acceleration_structure> blas_cache;
    std::vector<instance> instances;
    std::vector<instance_group> group_cache;
    blas_strategy group_strategy;

    bool refresh_instance_cache();
    void ensure_blas();
    void assign_group_cache(
        uint64_t id,
        bool static_mesh,
        bool static_transformable,
        entity object_index,
        entity& last_object_index
    );
    bool reserve_pre_transformed_vertices(size_t max_vertex_count);
    void clear_pre_transformed_vertices();

    //==========================================================================
    // Shadow map stuff.
    //==========================================================================
    // The scene_stage acts as the owner of these as they're
    // scene-wide GPU assets, but they're updated by shadow_map_stage. If the
    // scene specifies no shadow maps, these won't exist either.
    size_t total_shadow_map_count;
    size_t total_cascade_count;
    size_t shadow_map_range;
    size_t shadow_map_cascade_range;
    std::unordered_map<const light*, size_t /*index*/> shadow_map_indices;
    std::vector<shadow_map_instance> shadow_maps;
    std::unique_ptr<atlas> shadow_atlas;

    bool update_shadow_map_params();
    int get_shadow_map_index(const light* l);

    //==========================================================================
    // Scene resources
    //==========================================================================
    // Previous values for camera uniform data are tracked here for temporal
    // algorithms.
    std::vector<uint8_t> old_camera_data;

    sampler_table s_table;
    gpu_buffer instance_data;
    gpu_buffer scene_metadata;
    gpu_buffer directional_light_data;
    gpu_buffer point_light_data;
    gpu_buffer tri_light_data;
    gpu_buffer sh_grid_data;
    gpu_buffer shadow_map_data;
    gpu_buffer camera_data;
    sampler envmap_sampler;
    sampler shadow_sampler;
    sampler sh_grid_sampler;
    // Offsets and sizes to the camera uniform buffer.
    std::vector<std::pair<size_t, size_t>> camera_data_offsets;
    std::unordered_map<sh_grid*, texture> sh_grid_textures;

    std::optional<top_level_acceleration_structure> tlas;
    std::optional<event_subscription> events[10];

    descriptor_set scene_desc;

    //==========================================================================
    // Pipelines
    //==========================================================================
    per_device<compute_pipeline> skinning;
    per_device<compute_pipeline> extract_tri_lights;
    per_device<compute_pipeline> pre_transform;

    options opt;
    timer stage_timer;
};

}

#endif
