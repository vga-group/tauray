#ifndef TAURAY_SHADOW_MAP_STAGE_HH
#define TAURAY_SHADOW_MAP_STAGE_HH
#include "context.hh"
#include "raster_pipeline.hh"
#include "camera.hh"
#include "sampler_table.hh"
#include "timer.hh"
#include "gpu_buffer.hh"
#include "stage.hh"

namespace tr
{

class scene_stage;
class shadow_map_stage: public single_device_stage
{
public:
    struct options
    {
        size_t max_samplers = 128;
    };

    shadow_map_stage(
        device& dev,
        scene_stage& ss,
        uvec4 local_rect,
        render_target& depth_buffer,
        const options& opt
    );

    void set_camera(const camera& cur_cam);

private:
    void update(uint32_t frame_index) override;

    raster_pipeline gfx;
    options opt;
    gpu_buffer camera_data;

    timer shadow_timer;

    camera cur_cam;
    scene_stage* ss;
    uint32_t scene_state_counter;
};

}

#endif

