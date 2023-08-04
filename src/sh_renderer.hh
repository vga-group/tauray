#ifndef TAURAY_SH_RENDERER_HH
#define TAURAY_SH_RENDERER_HH
#include "context.hh"
#include "texture.hh"
#include "renderer.hh"
#include "sh_path_tracer_stage.hh"
#include "sh_compact_stage.hh"
#include "renderer.hh"

namespace tr
{

class scene_stage;
class sh_grid;
// This renderer is a bit odd in that it doesn't actually draw anything to the
// context; it only updates SH probe grids. As such, this renderer is not useful
// on its own and must be used as a part of a more comprehensive renderer
// (= dshgi_renderer).
class sh_renderer
{
public:
    using options = sh_path_tracer_stage::options;

    sh_renderer(context& ctx, scene_stage& ss, const options& opt);
    sh_renderer(const sh_renderer& other) = delete;
    sh_renderer(sh_renderer&& other) = delete;
    ~sh_renderer();

    dependencies render(dependencies deps);
    texture& get_sh_grid_texture(sh_grid* grid);

private:
    context* ctx;
    options opt;
    scene_stage* ss = nullptr;

    std::unordered_map<sh_grid*, texture> sh_grid_targets;
    std::unordered_map<sh_grid*, texture> sh_grid_textures;

    struct per_grid_data
    {
        std::unique_ptr<sh_path_tracer_stage> pt;
        std::unique_ptr<sh_compact_stage> compact;
    };
    std::vector<per_grid_data> per_grid;
};

}

#endif
