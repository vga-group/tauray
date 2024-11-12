#ifndef TR_RC_VISUALIZER_STAGE_HH
#define TR_RC_VISUALIZER_STAGE_HH
#include "radiance_cascades_stage.hh"
#include "timer.hh"

namespace tr
{

class rc_visualizer_stage: public single_device_stage
{
public:
    rc_visualizer_stage(
        radiance_cascades_stage& rc,
        render_target& output
    );

    void set_position(int cascade, int z_layer);

private:
    void update(uint32_t frame_index) override;

    radiance_cascades_stage* rc;
    render_target output;

    push_descriptor_set desc;
    compute_pipeline visualize;

    int cascade;
    int layer;
    timer stage_timer;
};

}

#endif


