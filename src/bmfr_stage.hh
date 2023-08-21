#ifndef TAURAY_BMFR_STAGE_HH
#define TAURAY_BMFR_STAGE_HH

#include "stage.hh"
#include "compute_pipeline.hh"
#include "gbuffer.hh"
#include "timer.hh"
#include "gpu_buffer.hh"

namespace tr
{

class bmfr_stage: public single_device_stage
{
public:
    enum class bmfr_settings
    {
        DIFFUSE_ONLY = 0,
        DIFFUSE_SPECULAR = 1
    };
    struct options
    {
        bmfr_settings settings;
    };

    bmfr_stage(
        device& dev,
        gbuffer_target& current_features,
        gbuffer_target& prev_features,
        const options& opt
    );
    bmfr_stage(const bmfr_stage& other) = delete;
    bmfr_stage(bmfr_stage&& other) = delete;

    virtual void update(uint32_t frame_index) override;

private:
    void init_resources();
    void record_command_buffers();
    static shader_source load_shader_source(const std::string& path, const options& opt);

    void copy_image(vk::CommandBuffer& cb, render_target& src, render_target& dst);

    compute_pipeline bmfr_preprocess_comp;
    compute_pipeline bmfr_fit_comp;
    compute_pipeline bmfr_weighted_sum_comp;
    compute_pipeline bmfr_accumulate_output_comp;
    gbuffer_target current_features;
    gbuffer_target prev_features;
    render_target tmp_noisy[2];
    render_target tmp_filtered[2];
    render_target diffuse_hist;
    render_target specular_hist;
    render_target filtered_hist[2];
    render_target weighted_sum[2];
    vkm<vk::Buffer> min_max_buffer[MAX_FRAMES_IN_FLIGHT];
    vkm<vk::Buffer> tmp_data[MAX_FRAMES_IN_FLIGHT];
    vkm<vk::Buffer> weights[MAX_FRAMES_IN_FLIGHT];
    vkm<vk::Buffer> accepts[MAX_FRAMES_IN_FLIGHT];
    gpu_buffer ubos;
    std::unique_ptr<texture> rt_textures[10];
    options opt;
    timer stage_timer, bmfr_preprocess_timer, bmfr_fit_timer, bmfr_weighted_sum_timer, bmfr_accumulate_output_timer, image_copy_timer;
};

} // namespace tr

#endif
