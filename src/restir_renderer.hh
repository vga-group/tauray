#ifndef TAURAY_RESTIR_RENDERER_HH
#define TAURAY_RESTIR_RENDERER_HH
#include "context.hh"
#include "gbuffer.hh"
#include "renderer.hh"
#include "raster_stage.hh"
#include "scene_stage.hh"
#include "restir_stage.hh"
#include "tonemap_stage.hh"
#include "gbuffer_copy_stage.hh"
#include <variant>

namespace tr
{

class restir_renderer: public renderer
{
public:
    struct options
    {
        scene_stage::options scene_options = {};
        restir_stage::options restir_options = {};
        tonemap_stage::options tonemap_options = {};
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

    std::optional<scene_stage> scene_update;
    std::optional<raster_stage> gbuffer_rasterizer;
    std::optional<restir_stage> restir;
    std::optional<tonemap_stage> tonemap;
    std::optional<gbuffer_copy_stage> copy;

    dependencies last_frame_deps;
};

}

#endif
