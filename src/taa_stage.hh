#ifndef TAURAY_TAA_STAGE_HH
#define TAURAY_TAA_STAGE_HH
#include "context.hh"
#include "texture.hh"
#include "stage.hh"
#include "compute_pipeline.hh"
#include "timer.hh"
#include "gbuffer.hh"
#include "gpu_buffer.hh"
#include "scene_stage.hh"
#include "sampler.hh"

namespace tr
{

class taa_stage: public single_device_stage
{
public:
    struct options
    {
        float blending_ratio = 1.0f;
        size_t active_viewport_count = 1;
    };

    taa_stage(
        device& dev,
        scene_stage& ss,
        gbuffer_target& current_features,
        const options& opt
    );
    taa_stage(const taa_stage& other) = delete;
    taa_stage(taa_stage&& other) = delete;

protected:
    void update(uint32_t frame_index) override;

private:
    void record_command_buffers();

    std::vector<vec4> jitter_history;
    scene_stage* ss;
    descriptor_set desc;
    compute_pipeline comp;
    options opt;
    gbuffer_target current_features;
    texture previous_color;
    gpu_buffer jitter_buffer;
    sampler history_sampler;
    timer stage_timer;
};

}

#endif
