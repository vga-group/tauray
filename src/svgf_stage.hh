#ifndef TAURAY_SVGF_STAGE_HH
#define TAURAY_SVGF_STAGE_HH
#include "stage.hh"
#include "gbuffer.hh"
#include "compute_pipeline.hh"
#include "timer.hh"
#include "scene_stage.hh"

namespace tr
{

class svgf_stage: public single_device_stage
{
public:
    struct options
    {
        size_t active_viewport_count = 1;
        int atrous_diffuse_iters;
        int atrous_spec_iters;
        int atrous_kernel_radius;
        float sigma_l;
        float sigma_z;
        float sigma_n;
        float temporal_alpha_color;
        float temporal_alpha_moments;
    };

    svgf_stage(
        device& dev,
        scene_stage& ss,
        gbuffer_target& input_features,
        gbuffer_target& prev_features,
        const options& opt
    );
    svgf_stage(const svgf_stage& other) = delete;
    svgf_stage(svgf_stage&& other) = delete;

    void update(uint32_t frame_index);

    void init_resources();
    void record_command_buffers();

private:
    compute_pipeline atrous_comp;
    compute_pipeline temporal_comp;
    compute_pipeline estimate_variance_comp;
    options opt;
    gbuffer_target input_features;
    gbuffer_target prev_features;
    render_target atrous_diffuse_pingpong[2];
    render_target atrous_specular_pingpong[2];
    render_target moments_history[2];
    render_target svgf_color_hist;
    render_target svgf_spec_hist;
    std::unique_ptr<texture> render_target_texture[8];
    timer svgf_timer;

    std::vector<vec4> jitter_history;
    gpu_buffer jitter_buffer;
    scene_stage* ss;
    uint32_t scene_state_counter;
};
}

#endif // TAURAY_SVGF_STAGE_HH
