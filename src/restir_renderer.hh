#ifndef TAURAY_RESTIR_RENDERER_HH
#define TAURAY_RESTIR_RENDERER_HH
#include "context.hh"
#include "gbuffer.hh"
#include "renderer.hh"
#include "raster_stage.hh"
#include "scene_stage.hh"
#include "restir_stage.hh"
#include "envmap_stage.hh"
#include "tonemap_stage.hh"
#include "shadow_map_stage.hh"
#include "gbuffer_copy_stage.hh"
#include "sh_renderer.hh"
#include <variant>

namespace tr
{

class restir_renderer: public renderer
{
public:
    struct options
    {
        scene_stage::options scene_options;
        // If shade_all_explicit_lights is set, hybrid raster rendering is used.
        // If shade_fake_indirect is set and scene contains SH grids, hybrid
        // DDISH-GI rendering is used.
        restir_stage::options restir_options;
        tonemap_stage::options tonemap_options;
        sh_renderer::options sh_options; // For raster hybrid
        shadow_map_filter sm_filter; // For raster hybrid
    };

    restir_renderer(context& ctx, const options& opt);
    restir_renderer(const restir_renderer& other) = delete;
    restir_renderer(restir_renderer&& other) = delete;

    void set_scene(scene* s) override;
    void render() override;

private:
    context* ctx;
    options opt;

    gbuffer_texture current_gbuffer;
    gbuffer_texture prev_gbuffer;

    std::optional<sh_renderer> sh;
    std::optional<scene_stage> scene_update;
    std::optional<shadow_map_stage> sms;

    struct per_view_stages
    {
        std::optional<envmap_stage> envmap;
        std::optional<raster_stage> gbuffer_rasterizer;
        std::optional<restir_stage> restir;
    };

    std::vector<per_view_stages> per_view;

    std::optional<tonemap_stage> tonemap;
    std::optional<gbuffer_copy_stage> copy;

    dependencies last_frame_deps;
};

}

#endif
