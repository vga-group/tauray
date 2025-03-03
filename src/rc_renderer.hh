#ifndef TAURAY_RC_RENDERER_HH
#define TAURAY_RC_RENDERER_HH
#include "context.hh"
#include "renderer.hh"
#include "scene_stage.hh"
#include "path_tracer_stage.hh"
#include "tonemap_stage.hh"
#include "rc_visualizer.hh"
#include "radiance_cascades_stage.hh"
#include "shadow_map_stage.hh"

namespace tr
{

class rc_renderer: public renderer
{
public:
    struct options
    {
        scene_stage::options scene_options;
        path_tracer_stage::options pt_options;
        tonemap_stage::options tonemap_options;
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

    gbuffer_texture gbuffer;

    std::optional<texture> distance_field;

    std::optional<scene_stage> scene_update;
    std::optional<radiance_cascades_stage> rc;
    std::optional<rc_visualizer_stage> rcv;
    std::optional<path_tracer_stage> pt;
    std::optional<tonemap_stage> tonemap;

    dependencies last_frame_deps;
};

}

#endif
