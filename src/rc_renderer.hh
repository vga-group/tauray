#ifndef TAURAY_RC_RENDERER_HH
#define TAURAY_RC_RENDERER_HH
#include "context.hh"
#include "renderer.hh"
#include "scene_stage.hh"
#include "path_tracer_stage.hh"
#include "svgf_stage.hh"
#include "envmap_stage.hh"
#include "gbuffer_copy_stage.hh"
#include "taa_stage.hh"
#include "raster_stage.hh"
#include "restir_stage.hh"
#include "tonemap_stage.hh"
#include "light_tree_stage.hh"
#include "rc_visualizer.hh"
#include "radiance_cascades_stage.hh"
#include "shadow_map_stage.hh"
#include "voxelizer.hh"

namespace tr
{

class rc_renderer: public renderer
{
public:
    struct options
    {
        scene_stage::options scene_options;
        radiance_cascades_stage::options rc_options;
        std::optional<light_tree_stage::options> light_tree = {};
        std::optional<path_tracer_stage::options> pt_options;
        std::optional<restir_stage::options> restir_options;
        std::optional<svgf_stage::options> svgf_options;
        std::optional<taa_stage::options> taa_options;
        tonemap_stage::options tonemap_options;

        bool enable_visualizer = false;
    };

    rc_renderer(context& ctx, const options& opt);
    rc_renderer(const rc_renderer& other) = delete;
    rc_renderer(rc_renderer&& other) = delete;

    void set_scene(scene* s) override;
    void set_visualizer_pos(int cascade, int layer);
    void render() override;
    void reset_accumulation(bool reset_sample_counter) override;

private:
    context* ctx;
    options opt;

    gbuffer_texture current_gbuffer;
    gbuffer_texture prev_gbuffer;

    std::optional<texture> taa_input_target;

    std::optional<scene_stage> scene_update;
    std::optional<light_tree_stage> light_tree;
    std::optional<voxelizer_stage> voxelizer;
    std::optional<shadow_map_stage> sms;
    std::optional<radiance_cascades_stage> rc;
    std::optional<envmap_stage> envmap;
    std::optional<raster_stage> gbuffer_rasterizer;
    std::optional<rc_visualizer_stage> rcv;
    std::optional<path_tracer_stage> pt;
    std::optional<restir_stage> restir;
    std::optional<svgf_stage> svgf;
    std::optional<tonemap_stage> tonemap;
    std::optional<taa_stage> taa;
    std::optional<gbuffer_copy_stage> copy;

    dependencies last_frame_deps;
};

}

#endif
