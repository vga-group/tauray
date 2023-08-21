#ifndef TAURAY_FRAME_DELAY_STAGE_HH
#define TAURAY_FRAME_DELAY_STAGE_HH
#include "context.hh"
#include "stage.hh"
#include "timer.hh"
#include "gbuffer.hh"

namespace tr
{

// This stage outputs a G-Buffer that is delayed by one frame. It's needed
// for temporal algorithms, to access the previous frame.
//
// Run this stage directly after post-processing. Additionally, all stages
// which generate the input_features for the next frame must wait for the
// dependency of the frame_delay_stage. These cross-frame dependencies ensure
// that we can avoid synchronization issues.
class frame_delay_stage: public single_device_stage
{
public:
    frame_delay_stage(
        device& dev,
        gbuffer_target& input_features
    );
    gbuffer_target get_output();

protected:
private:
    gbuffer_target output_features;
    std::unique_ptr<gbuffer_texture> textures;
    timer delay_timer;
};

}

#endif
