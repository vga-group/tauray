#ifndef TAURAY_ENVMAP_STAGE_HH
#define TAURAY_ENVMAP_STAGE_HH
#include "stage.hh"
#include "raster_pipeline.hh"
#include "sampler.hh"
#include "timer.hh"
#include "gpu_buffer.hh"

namespace tr
{

class scene;
// Just renders the environment map from the given scene as the sky, if present.
class envmap_stage: public stage
{
public:
    envmap_stage(
        device_data& dev,
        const std::vector<render_target>& color_arrays
    );

    void set_scene(scene* s);

protected:
private:
    std::vector<std::unique_ptr<raster_pipeline>> array_pipelines;
    timer envmap_timer;

    scene* cur_scene;
};

}

#endif

