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
#include "gbuffer_copy_stage.hh"
#include "device_transfer.hh"
#include <variant>

namespace tr
{

class restir_renderer: public renderer
{
public:
    struct options
    {
        scene_stage::options scene_options;
        restir_stage::options restir_options;
        tonemap_stage::options tonemap_options;
    };

    restir_renderer(context& ctx, const options& opt);
    restir_renderer(const restir_renderer& other) = delete;
    restir_renderer(restir_renderer&& other) = delete;

    void set_scene(scene* s) override;
    void render() override;

private:
    context* ctx;
    options opt;

    std::optional<scene_stage> scene_update;

    struct per_device_data
    {
        gbuffer_texture current_gbuffer;
        gbuffer_texture prev_gbuffer;
    };

    std::vector<per_device_data> per_device;

    struct per_view_stages
    {
        std::optional<texture> taa_input_target;
        std::optional<texture> tmp_compressed_output_img;

        std::optional<envmap_stage> envmap;
        std::optional<raster_stage> gbuffer_rasterizer;
        std::optional<restir_stage> restir;
        std::vector<std::unique_ptr<device_transfer_interface>> transfer;
        std::optional<tonemap_stage> tonemap;
        std::optional<gbuffer_copy_stage> copy;
    };

    // There's either 1 device with multiple views, or one device per view.
    std::vector<per_view_stages> per_view;

    dependencies last_frame_deps;
};

}

#endif
