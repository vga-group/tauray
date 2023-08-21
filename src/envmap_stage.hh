#ifndef TAURAY_ENVMAP_STAGE_HH
#define TAURAY_ENVMAP_STAGE_HH
#include "stage.hh"
#include "raster_pipeline.hh"
#include "sampler.hh"
#include "timer.hh"
#include "gpu_buffer.hh"

namespace tr
{

class scene_stage;
// Just renders the environment map from the given scene as the sky, if present.
class envmap_stage: public single_device_stage
{
public:
    envmap_stage(
        device& dev,
        scene_stage& ss,
        const std::vector<render_target>& color_arrays
    );

protected:
    void update(uint32_t frame_index) override;

private:
    std::vector<std::unique_ptr<raster_pipeline>> array_pipelines;
    timer envmap_timer;

    uint32_t scene_state_counter;
    scene_stage* ss;
};

}

#endif

