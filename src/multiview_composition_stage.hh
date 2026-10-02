#ifndef TAURAY_MULTIVIEW_COMPOSITION_STAGE_HH
#define TAURAY_MULTIVIEW_COMPOSITION_STAGE_HH
#include "context.hh"
#include "texture.hh"
#include "compute_pipeline.hh"
#include "descriptor_set.hh"
#include "sampler.hh"
#include "timer.hh"
#include "stage.hh"

namespace tr
{

class multiview_composition_stage: public single_device_stage
{
public:
    struct options
    {
        uvec2 views;
    };

    multiview_composition_stage(
        device& dev,
        render_target& input,
        std::vector<render_target>& output_frames,
        const options& opt
    );

private:
    render_target input;
    std::vector<render_target> output_frames;
    push_descriptor_set desc;
    compute_pipeline comp;
    timer stage_timer;
};

}

#endif

