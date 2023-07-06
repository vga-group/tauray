#ifndef TAURAY_EXAMPLE_DENOISER_STAGE_HH
#define TAURAY_EXAMPLE_DENOISER_STAGE_HH
#include "context.hh"
#include "texture.hh"
#include "stage.hh"
#include "compute_pipeline.hh"
#include "timer.hh"
#include "gbuffer.hh"

namespace tr
{

// This denoiser works as an example for GBuffer-aware denoising. It assumes
// that there is no MSAA.
class example_denoiser_stage: public single_device_stage
{
public:
    struct options
    {
        int kernel_radius = 1;
        int repeat_count = 1;
        bool albedo_modulation = false;
    };

    example_denoiser_stage(
        device& dev,
        gbuffer_target& input_features,
        render_target& tmp_color1,
        render_target& tmp_color2,
        const options& opt
    );
    example_denoiser_stage(const example_denoiser_stage& other) = delete;
    example_denoiser_stage(example_denoiser_stage&& other) = delete;

    bool need_pingpong_swap() const;

protected:
private:
    void init_resources();
    void record_command_buffers();

    compute_pipeline comp;
    options opt;
    gbuffer_target input_features;
    render_target tmp_color[2];
    timer denoiser_timer;
};

}

#endif


