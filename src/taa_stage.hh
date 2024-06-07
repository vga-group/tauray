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
        size_t active_viewport_count = 1;

        // Set this to 2.2 if running TAA after gamma/srgb correction.
        // Optimally, you'd run TAA after tonemapping but before gamma/srgb
        // correction, this variable won't fix all issues.
        float gamma = 1.0f;

        // Set this to 1/taa_steps by default
        float alpha = 0.125f;

        // True antialiases harder but can blur a bit more. False can falsely
        // fail on antialiasing some edges.
        bool edge_dilation = true;

        // Avoids shimmering by darkening details that would cause it.
        bool anti_shimmer = false;
    };

    taa_stage(
        device& dev,
        scene_stage& ss,
        render_target& src,
        render_target& motion,
        render_target& depth,
        render_target& dst,
        const options& opt
    );
    taa_stage(
        device& dev,
        scene_stage& ss,
        render_target& src,
        render_target& motion,
        render_target& depth,
        std::vector<render_target>& swapchain_dst,
        const options& opt
    );
    taa_stage(const taa_stage& other) = delete;
    taa_stage(taa_stage&& other) = delete;

protected:
    void init();
    void update(uint32_t frame_index) override;

private:
    options opt;
    scene_stage* scene;

    render_target src, motion, depth;
    std::vector<render_target> dst;

    std::optional<texture> color_history[2];
    compute_pipeline pipeline;
    push_descriptor_set descriptors;
    sampler target_sampler;
    sampler history_sampler;
    timer stage_timer;
};

}

#endif
