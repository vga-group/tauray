#ifndef TAURAY_SHADOW_MAP_STAGE_HH
#define TAURAY_SHADOW_MAP_STAGE_HH
#include "context.hh"
#include "raster_pipeline.hh"
#include "camera.hh"
#include "sampler_table.hh"
#include "timer.hh"
#include "gpu_buffer.hh"
#include "stage.hh"
#include "atlas.hh"
#include "scene_stage.hh"

namespace tr
{

class shadow_map_stage: public single_device_stage
{
public:
    struct options
    {
        size_t max_samplers = 128;
    };

    shadow_map_stage(device& dev, scene_stage& ss, const options& opt);

private:
    void update(uint32_t frame_index) override;

    std::optional<raster_pipeline> gfx;
    options opt;
    gpu_buffer camera_data;
    std::vector<scene_stage::shadow_map_instance> shadow_maps;
    uvec2 prev_atlas_size;

    timer shadow_timer;

    scene_stage* ss;
    uint32_t scene_state_counter;
};

}

#endif
