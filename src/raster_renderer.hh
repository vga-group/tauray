#ifndef TAURAY_RASTER_RENDERER_HH
#define TAURAY_RASTER_RENDERER_HH
#include "context.hh"
#include "texture.hh"
#include "z_pass_stage.hh"
#include "raster_stage.hh"
#include "envmap_stage.hh"
#include "scene_stage.hh"
#include "shadow_map_renderer.hh"
#include "post_processing_renderer.hh"
#include "renderer.hh"

namespace tr
{

class scene;
class raster_renderer: public renderer
{
public:
    struct options: raster_stage::options
    {
        int msaa_samples = 1;
        scene_stage::options scene_options = {};
        post_processing_renderer::options post_process = {};
        // Enabling the Z pre-pass can help with performance if the scene is
        // overdraw + bandwidth-heavy. It essentially just prevents all overdraw
        // from taking place during final rasterization at the cost of an extra
        // z pass.
        bool z_pre_pass = true;
    };

    raster_renderer(context& ctx, const options& opt);
    raster_renderer(const raster_renderer& other) = delete;
    raster_renderer(raster_renderer&& other) = delete;
    ~raster_renderer();

    void set_scene(scene* s) override;
    void render() override;

protected:
    dependencies render_core(dependencies deps);
    void init_common_resources();
    void init_resources(size_t display_index);

    context* ctx;
    options opt;
    scene* cur_scene = nullptr;
    std::unique_ptr<scene_stage> scene_update;

private:
    shadow_map_renderer smr;
    post_processing_renderer post_processing;
    std::vector<vkm<vk::Semaphore>> shared_resource_semaphores;
    gbuffer_texture gbuffer;
    std::unique_ptr<envmap_stage> envmap;
    std::unique_ptr<z_pass_stage> z_pass;
    std::unique_ptr<raster_stage> raster;
};

}

#endif
