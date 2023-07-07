#ifndef TAURAY_RT_RENDERER_HH
#define TAURAY_RT_RENDERER_HH
#ifdef WIN32
#include "windows.h"
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#endif
#include "context.hh"
#include "texture.hh"
#include "path_tracer_stage.hh"
#include "direct_stage.hh"
#include "whitted_stage.hh"
#include "raster_stage.hh"
#include "feature_stage.hh"
#include "stitch_stage.hh"
#include "scene_update_stage.hh"
#include "skinning_stage.hh"
#include "renderer.hh"
#include "device_transfer.hh"
#include "post_processing_renderer.hh"
#include <variant>

namespace tr
{

class scene;

template<typename Pipeline>
class rt_renderer: public renderer
{
public:
    struct options: Pipeline::options
    {
        scene_update_stage::options scene_options = {};
        post_processing_renderer::options post_process = {};
        size_t active_viewport_count = 1;
        bool accumulate = false;
    };

    rt_renderer(context& ctx, const options& opt);
    rt_renderer(const rt_renderer& other) = delete;
    rt_renderer(rt_renderer&& other) = delete;
    ~rt_renderer();

    void set_scene(scene* s) override;
    void reset_accumulation(bool reset_sample_counter = true) override;
    void render() override;
    void set_device_workloads(const std::vector<double>& ratios) override;

private:
    void init_resources();
    void prepare_transfers(bool reserve);

    context* ctx;
    options opt;
    post_processing_renderer post_processing;
    bool use_raster_gbuffer = true;
    unsigned accumulated_frames = 0;

    gbuffer_texture gbuffer;

    struct per_device_data
    {
        gbuffer_texture gbuffer_copy;
        std::unique_ptr<device_transfer_interface> transfer;
        std::unique_ptr<Pipeline> ray_tracer;
        distribution_params dist;
    };
    std::vector<per_device_data> per_device;
    std::unique_ptr<skinning_stage> skinning;
    std::unique_ptr<scene_update_stage> scene_update;
    std::unique_ptr<stitch_stage> stitch;
    std::unique_ptr<raster_stage> gbuffer_rasterizer;
    dependencies last_frame_deps;
};

using path_tracer_renderer = rt_renderer<path_tracer_stage>;
using whitted_renderer = rt_renderer<whitted_stage>;
using feature_renderer = rt_renderer<feature_stage>;
using direct_renderer = rt_renderer<direct_stage>;

}

#endif
