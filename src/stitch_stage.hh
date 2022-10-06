#ifndef TAURAY_STITCH_STAGE_HH
#define TAURAY_STITCH_STAGE_HH
#include "context.hh"
#include "texture.hh"
#include "compute_pipeline.hh"
#include "distribution_strategy.hh"
#include "gbuffer.hh"
#include "timer.hh"
#include "stage.hh"

namespace tr
{

class scene;
class stitch_stage: public stage
{
public:
    struct options
    {
        distribution_strategy strategy = DISTRIBUTION_SCANLINE;
        size_t active_viewport_count = 1;
    };

    stitch_stage(
        device_data& dev, 
        uvec2 size,
        const std::vector<gbuffer_target>& images,
        const std::vector<distribution_params>& params,
        const options& opt
    );

    void set_blend_ratio(float blend_ratio);
    void set_distribution_params(
        const std::vector<distribution_params>& params
    );
    void refresh_params();

private:
    void record_commands();

    compute_pipeline comp;
    options opt;
    uvec2 size;
    float blend_ratio;

    std::vector<gbuffer_target> images;
    std::vector<distribution_params> params;
    timer stitch_timer;
};

}

#endif
