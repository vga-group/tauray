#ifndef TAURAY_SH_COMPACT_STAGE_HH
#define TAURAY_SH_COMPACT_STAGE_HH
#include "context.hh"
#include "texture.hh"
#include "compute_pipeline.hh"
#include "timer.hh"
#include "stage.hh"

namespace tr
{

class scene;
class sh_compact_stage: public stage
{
public:
    sh_compact_stage(
        device_data& dev, 
        texture& inflated_source,
        texture& compacted_output
    );

private:
    compute_pipeline comp;
    timer compact_timer;
};

}

#endif
