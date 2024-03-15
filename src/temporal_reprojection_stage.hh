#ifndef TAURAY_TEMPORAL_REPROJECTION_STAGE_HH
#define TAURAY_TEMPORAL_REPROJECTION_STAGE_HH
#include "context.hh"
#include "texture.hh"
#include "stage.hh"
#include "compute_pipeline.hh"
#include "descriptor_set.hh"
#include "timer.hh"
#include "gbuffer.hh"

namespace tr
{

class temporal_reprojection_stage: public single_device_stage
{
public:
    struct options
    {
        float temporal_ratio = 0.75;
        size_t active_viewport_count = 1;
    };

    temporal_reprojection_stage(
        device& dev,
        gbuffer_target& current_features,
        gbuffer_target& previous_features,
        const options& opt
    );
    temporal_reprojection_stage(const temporal_reprojection_stage& other) = delete;
    temporal_reprojection_stage(temporal_reprojection_stage&& other) = delete;

private:
    push_descriptor_set desc;
    compute_pipeline comp;
    options opt;
    timer stage_timer;
};

}

#endif


