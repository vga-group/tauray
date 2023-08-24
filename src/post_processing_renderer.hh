#ifndef TAURAY_POST_PROCESSING_RENDERER_HH
#define TAURAY_POST_PROCESSING_RENDERER_HH
#include "context.hh"
#include "tonemap_stage.hh"
#include "renderer.hh"
#include "example_denoiser_stage.hh"
#include "temporal_reprojection_stage.hh"
#include "spatial_reprojection_stage.hh"
#include "svgf_stage.hh"
#include "taa_stage.hh"
#include "frame_delay_stage.hh"
#include "gbuffer.hh"
#include "bmfr_stage.hh"

namespace tr
{

class post_processing_renderer
{
public:
    struct options
    {
        std::optional<example_denoiser_stage::options> example_denoiser;
        std::optional<temporal_reprojection_stage::options> temporal_reprojection;
        std::optional<spatial_reprojection_stage::options> spatial_reprojection;
        std::optional<svgf_stage::options> svgf_denoiser;
        std::optional<taa_stage::options> taa;
        std::optional<bmfr_stage::options> bmfr;
        tonemap_stage::options tonemap;
        size_t active_viewport_count;
    };

    post_processing_renderer(
        device& dev, scene_stage& ss, uvec2 output_size, const options& opt
    );
    ~post_processing_renderer();

    void set_gbuffer_spec(gbuffer_spec& spec) const;
    void set_display(gbuffer_target input_gbuffer);

    // You must wait for these dependencies before writing to the G-Buffer.
    // There may sometimes be no dependencies, the purpose here is mostly just
    // to handle temporal algorithms.
    dependencies get_gbuffer_write_dependencies() const;

    dependencies render(dependencies deps);

private:
    void init_pipelines();
    void deinit_pipelines();

    device* dev;
    options opt;
    uvec2 output_size;
    scene_stage* ss = nullptr;

    gbuffer_target input_gbuffer;

    std::unique_ptr<texture> pingpong[2];

    // Add the new post processing pipelines here.
    std::unique_ptr<example_denoiser_stage> example_denoiser;
    std::unique_ptr<temporal_reprojection_stage> temporal_reprojection;
    std::unique_ptr<spatial_reprojection_stage> spatial_reprojection;
    std::unique_ptr<svgf_stage> svgf;
    std::unique_ptr<taa_stage> taa;
    std::unique_ptr<bmfr_stage> bmfr;

    // Tonemap should _always_ be the last stage. You can think of its task
    // as simply fixing the mistakes display manufacturers made a long time
    // ago. Displays don't have linear response to the pixel values; a basic
    // tonemapper just does the inverse transform so that the response is
    // linear again.
    std::unique_ptr<tonemap_stage> tonemap;

    // This delayer is for safely getting the gbuffer for the previous frame.
    std::unique_ptr<frame_delay_stage> delay;
    dependencies delay_deps[MAX_FRAMES_IN_FLIGHT];
};

}

#endif
