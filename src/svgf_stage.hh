#ifndef TAURAY_SVGF_STAGE_HH
#define TAURAY_SVGF_STAGE_HH
#include "stage.hh"
#include "gbuffer.hh"
#include "compute_pipeline.hh"
#include "descriptor_set.hh"
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
        bool color_buffer_contains_direct_light;
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
    push_descriptor_set atrous_desc;
    compute_pipeline atrous_comp;
    push_descriptor_set temporal_desc;
    compute_pipeline temporal_comp;
    push_descriptor_set firefly_suppression_desc;
    compute_pipeline firefly_suppression_comp;
    push_descriptor_set disocclusion_fix_desc;
    compute_pipeline disocclusion_fix_comp;
    push_descriptor_set hit_dist_reconstruction_desc;
    compute_pipeline hit_dist_reconstruction_comp;
    options opt;
    gbuffer_target input_features;
    gbuffer_target prev_features;
    render_target atrous_diffuse_pingpong[2];
    render_target atrous_specular_pingpong[2];
    render_target history_length[2]; // R: diffuse history length, G: diffuse alpha, G: specular history length A: specular alpha
    render_target svgf_color_hist;
    render_target svgf_spec_hist;
    render_target specular_hit_distance[2];
    static constexpr uint32_t render_target_count = 10;
    std::unique_ptr<texture> render_target_texture[render_target_count];
    timer svgf_timer;

    timer reconstruction_timer;
    timer disocclusion_timer;
    timer firefly_timer;
    timer temporal_timer;
    timer atrous_timer;

    sampler my_sampler;
    scene_stage* ss;
    uint32_t scene_state_counter;

    gpu_buffer uniforms;
};
}

#endif // TAURAY_SVGF_STAGE_HH
