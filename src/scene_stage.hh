#ifndef TAURAY_SCENE_STAGE_HH
#define TAURAY_SCENE_STAGE_HH
#include "scene.hh"
#include "stage.hh"
#include "compute_pipeline.hh"
#include "radix_sort.hh"

namespace tr
{

class shadow_map_renderer;
class scene_stage: public multi_device_stage
{
public:
    struct options
    {
        uint32_t max_instances = 1024;
        bool gather_emissive_triangles = false;
        bool pre_transform_vertices = false;
    };

    scene_stage(device_mask dev, const options& opt);

    void set_scene(scene* target);
    scene* get_scene() const; // TODO: Remove?

    // Update categories, these are used for checking if individual aspects of
    // the scene have changed. This is needed so that stages can update their
    // command buffers on-demand.
    static inline constexpr uint32_t ENVMAP = 1<<0;
    static inline constexpr uint32_t GEOMETRY = 1<<1;
    static inline constexpr uint32_t LIGHT = 1<<2;

    bool check_update(uint32_t categories, uint32_t& prev_counter) const;

    environment_map* get_environment_map() const;
    vec3 get_ambient() const;

    using instance = mesh_scene::instance;
    const std::vector<instance>& get_instances() const;

    vk::AccelerationStructureKHR get_acceleration_structure(
        device_id id
    ) const;

    void set_shadow_map_renderer(shadow_map_renderer* smr);
    void set_sh_grid_textures(
        std::unordered_map<sh_grid*, texture>* sh_grid_textures
    );

    vec2 get_shadow_map_atlas_pixel_margin() const;

    void bind(basic_pipeline& pipeline, uint32_t frame_index, int32_t camera_offset = 0);
    void push(basic_pipeline& pipeline, vk::CommandBuffer cmd, int32_t camera_offset = 0);
    static void bind_placeholders(
        basic_pipeline& pipeline,
        size_t max_samplers,
        size_t max_3d_samplers
    );

protected:
    void update(uint32_t frame_index) override;

private:
    void record_command_buffers();
    void record_skinning(device_id id, uint32_t frame_index, vk::CommandBuffer cb);
    void record_as_build(device_id id, uint32_t frame_index, vk::CommandBuffer cb);
    void record_tri_light_extraction(device_id id, vk::CommandBuffer cb);
    void record_pre_transform(device_id id, vk::CommandBuffer cb);

    std::vector<descriptor_state> get_descriptor_info(device_id id, int32_t camera_index) const;

    bool as_rebuild;
    size_t as_instance_count;

    uint32_t envmap_change_counter;
    uint32_t geometry_change_counter;
    uint32_t light_change_counter;

    bool command_buffers_outdated;
    unsigned force_instance_refresh_frames;
    scene* cur_scene;

    // Light stuff
    environment_map* envmap;
    vec3 ambient;

    // Previous values for camera uniform data are tracked here for temporal
    // algorithms.
    std::vector<uint8_t> old_camera_data;

    per_device<compute_pipeline> skinning;
    per_device<compute_pipeline> extract_tri_lights;
    per_device<compute_pipeline> pre_transform;

    options opt;
    timer stage_timer;
};

}

#endif
