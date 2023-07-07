#ifndef TAURAY_SPATIAL_REPROJECTION_STAGE_HH
#define TAURAY_SPATIAL_REPROJECTION_STAGE_HH
#include "context.hh"
#include "texture.hh"
#include "stage.hh"
#include "compute_pipeline.hh"
#include "timer.hh"
#include "gbuffer.hh"
#include "scene.hh"

namespace tr
{

class spatial_reprojection_stage: public single_device_stage
{
public:
    struct options
    {
        size_t active_viewport_count;
    };

    spatial_reprojection_stage(
        device& dev,
        gbuffer_target& target_viewport,
        const options& opt
    );

    void set_scene(scene* s);
    
private:
    void update(uint32_t frame_index) override;

    scene* current_scene;
    
    gbuffer_target target_viewport;

    compute_pipeline comp;
    options opt;
    
    gpu_buffer camera_data;
    timer stage_timer;
};

}

#endif


