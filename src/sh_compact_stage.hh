#ifndef TAURAY_SH_COMPACT_STAGE_HH
#define TAURAY_SH_COMPACT_STAGE_HH
#include "context.hh"
#include "texture.hh"
#include "compute_pipeline.hh"
#include "descriptor_set.hh"
#include "timer.hh"
#include "stage.hh"

namespace tr
{

class sh_compact_stage: public single_device_stage
{
public:
    sh_compact_stage(
        device& dev, 
        texture& inflated_source,
        texture& compacted_output
    );

private:
    push_descriptor_set desc;
    compute_pipeline comp;
    timer compact_timer;
};

}

#endif
