#ifndef TAURAY_SVGF_STAGE_HH
#define TAURAY_SVGF_STAGE_HH
#include "stage.hh"
#include "gbuffer.hh"
#include "compute_pipeline.hh"
#include "timer.hh"

namespace tr
{

class svgf_stage: public stage 
{
public:
    struct options
    {
        int repeat_count = 1;
    };

    svgf_stage(
        device_data& dev,
        gbuffer_target& input_features,
        gbuffer_target& prev_features,
        render_target& tmp_color1,
        render_target& tmp_color2,
        const options& opt
    );
    svgf_stage(const svgf_stage& other) = delete;
    svgf_stage(svgf_stage&& other) = delete;

    bool need_ping_pongswap() const { return opt.repeat_count % 2 == 1; }

    void init_resources();
    void record_command_buffers();

private:
    compute_pipeline atrous_comp;
    compute_pipeline temporal_comp;
    options opt; 
    gbuffer_target input_features;
    gbuffer_target prev_features;
    render_target tmp_color[2];
    render_target atrous_specular_pingpong[2];
    render_target moments_history[2];
    render_target svgf_color_hist;
    render_target svgf_spec_hist;
    std::unique_ptr<texture> render_target_texture[6];
    timer svgf_timer;
};
}

#endif // TAURAY_SVGF_STAGE_HH