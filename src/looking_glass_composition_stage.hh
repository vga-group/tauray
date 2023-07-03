#ifndef TAURAY_LOOKING_GLASS_COMPOSITION_STAGE_HH
#define TAURAY_LOOKING_GLASS_COMPOSITION_STAGE_HH
#include "context.hh"
#include "texture.hh"
#include "compute_pipeline.hh"
#include "sampler.hh"
#include "timer.hh"
#include "stage.hh"

namespace tr
{

class scene;
class looking_glass_composition_stage: public stage
{
public:
    struct options
    {
        uint32_t viewport_count;
        float pitch;
        float tilt;
        float center;
        bool invert;
    };

    looking_glass_composition_stage(
        device_data& dev,
        render_target& input,
        std::vector<render_target>& output_frames,
        const options& opt
    );

private:
    compute_pipeline comp;
    sampler input_sampler;
    timer stage_timer;
};

}

#endif
