#ifndef TAURAY_SCENE_UPDATE_STAGE_HH
#define TAURAY_SCENE_UPDATE_STAGE_HH
#include "scene.hh"
#include "stage.hh"
#include "compute_pipeline.hh"
#include "radix_sort.hh"

namespace tr
{

class shadow_map_renderer;
class scene_update_stage: public stage
{
public:
    struct options
    {
        uint32_t max_meshes = 1024;
        bool gather_emissive_triangles = false;
    };

    scene_update_stage(device_data& dev, const options& opt);

    void set_scene(scene* target);

protected:
    void update(uint32_t frame_index) override;

private:
    void record_command_buffers();
    void record_as_build(
        uint32_t frame_index,
        vk::CommandBuffer cb
    );
    void record_tri_light_extraction(
        uint32_t frame_index,
        vk::CommandBuffer cb
    );

    bool as_rebuild;
    bool command_buffers_outdated;
    unsigned force_instance_refresh_frames;
    scene* cur_scene;
    // Previous values for camera uniform data are tracked here for temporal
    // algorithms.
    std::vector<uint8_t> old_camera_data;

    compute_pipeline extract_tri_lights;

    options opt;
    timer stage_timer;
};

}

#endif
