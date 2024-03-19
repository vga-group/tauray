#ifndef TAURAY_GBUFFER_COPY_STAGE_HH
#define TAURAY_GBUFFER_COPY_STAGE_HH
#include "context.hh"
#include "stage.hh"
#include "timer.hh"
#include "gbuffer.hh"

namespace tr
{

// Copies the common entries of G-Buffer A to G-Buffer B.
class gbuffer_copy_stage: public single_device_stage
{
public:
    gbuffer_copy_stage(
        device& dev,
        gbuffer_target& in,
        gbuffer_target& out
    );

private:
    timer copy_timer;
};

}

#endif
