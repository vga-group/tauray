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

protected:
    void update(uint32_t frame_index) override;

private:
    void record_command_buffers();
    void record_as_build(
        device_id id,
        uint32_t frame_index,
        vk::CommandBuffer cb
    );

    void record_tri_light_extraction(
        device_id id,
        vk::CommandBuffer cb
    );

    void record_pre_transform(
        device_id id,
        vk::CommandBuffer cb
    );

    bool as_rebuild;
    size_t as_instance_count;
    bool command_buffers_outdated;
    unsigned force_instance_refresh_frames;
    scene* cur_scene;
    // Previous values for camera uniform data are tracked here for temporal
    // algorithms.
    std::vector<uint8_t> old_camera_data;

    per_device<compute_pipeline> extract_tri_lights;
    per_device<compute_pipeline> pre_transform;

    options opt;
    timer stage_timer;
};

}

#endif
