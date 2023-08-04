#ifndef TAURAY_Z_PASS_STAGE_HH
#define TAURAY_Z_PASS_STAGE_HH
#include "context.hh"
#include "timer.hh"
#include "raster_pipeline.hh"
#include "gpu_buffer.hh"
#include "stage.hh"

namespace tr
{

class scene_stage;
class z_pass_stage: public single_device_stage
{
public:
    z_pass_stage(
        device& dev, 
        scene_stage& ss,
        const std::vector<render_target>& depth_buffer_arrays
    );

protected:
    void update(uint32_t frame_index) override;

private:
    std::vector<std::unique_ptr<raster_pipeline>> array_pipelines;

    scene_stage* ss;
    timer z_pass_timer;
    uint32_t scene_state_counter;
};

}

#endif
