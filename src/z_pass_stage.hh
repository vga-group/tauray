#ifndef TAURAY_Z_PASS_STAGE_HH
#define TAURAY_Z_PASS_STAGE_HH
#include "context.hh"
#include "timer.hh"
#include "raster_pipeline.hh"
#include "gpu_buffer.hh"
#include "stage.hh"

namespace tr
{

class scene;
class z_pass_stage: public stage
{
public:
    z_pass_stage(
        device_data& dev, 
        const std::vector<render_target>& depth_buffer_arrays
    );

    void set_scene(scene* s);
    scene* get_scene();

private:
    std::vector<std::unique_ptr<raster_pipeline>> array_pipelines;

    scene* cur_scene;
    timer z_pass_timer;
};

}

#endif
