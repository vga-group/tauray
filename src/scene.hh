#ifndef TAURAY_SCENE_HH
#define TAURAY_SCENE_HH
#include "light_scene.hh"
#include "mesh_scene.hh"
#include "timer.hh"
#include "sampler_table.hh"
#include "descriptor_state.hh"
#include "acceleration_structure.hh"

namespace tr
{

class camera;
class environment_map;
class sh_grid;
class basic_pipeline;
class shadow_map_renderer;

class scene: public light_scene, public mesh_scene
{
public:
    scene(
        context& ctx,
        size_t max_instances = 1024,
        size_t max_lights = 128
    );
    scene(const scene& s) = delete;
    scene(scene&& s) = delete;

    void set_camera(camera* cam);
    camera* get_camera(unsigned index = 0) const;

    using light_scene::add;
    using light_scene::remove;

    using mesh_scene::add;
    using mesh_scene::remove;

    void add(camera& c);
    void remove(camera& c);
    void clear_cameras();
    const std::vector<camera*>& get_cameras() const;
    void reorder_cameras_by_active(const std::set<int>& active_indices);
    void set_camera_jitter(const std::vector<vec2>& jitter);

    void add_control_node(animated_node& o);
    void remove_control_node(animated_node& o);
    void clear_control_nodes();

    void clear();

    void play(
        const std::string& name,
        bool loop = false,
        bool use_fallback = false
    );
    void update(time_ticks dt, bool force_update = false);
    bool is_playing() const;
    void set_animation_time(time_ticks dt);
    time_ticks get_total_ticks() const;

    vk::AccelerationStructureKHR get_acceleration_structure(
        size_t device_index
    ) const;

    void set_shadow_map_renderer(shadow_map_renderer* smr);
    void set_sh_grid_textures(
        std::unordered_map<sh_grid*, texture>* sh_grid_textures
    );
    vec2 get_shadow_map_atlas_pixel_margin() const;

    std::vector<descriptor_state> get_descriptor_info(device_data* dev, int32_t camera_index) const;
    void bind(basic_pipeline& pipeline, uint32_t frame_index, int32_t camera_offset = 0);
    void push(basic_pipeline& pipeline, vk::CommandBuffer cmd, int32_t camera_offset = 0);
    static void bind_placeholders(
        basic_pipeline& pipeline,
        size_t max_samplers,
        size_t max_3d_samplers
    );

private:
    friend class scene_update_stage;

    void init_acceleration_structures();

    context* ctx;
    std::vector<camera*> cameras;
    std::vector<animated_node*> control_nodes;
    time_ticks total_ticks;

    shadow_map_renderer* smr;
    std::unordered_map<sh_grid*, texture>* sh_grid_textures;

    struct scene_buffer
    {
        scene_buffer(device_data& dev);

        sampler_table s_table;
        gpu_buffer scene_data;
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

        std::vector<vk::DescriptorImageInfo> dii;
        // Offsets and sizes to the camera uniform buffer.
        std::vector<std::pair<size_t, size_t>> camera_data_offsets;
        size_t shadow_map_range;
        size_t shadow_map_cascade_range;
    };
    std::vector<scene_buffer> scene_buffers;

    std::optional<top_level_acceleration_structure> tlas;
};

std::vector<uint32_t> get_viewport_reorder_mask(
    const std::set<int>& active_indices,
    size_t viewport_count
);

}

#endif
