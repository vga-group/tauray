#ifndef TAURAY_HH
#define TAURAY_HH
#include "options.hh"
#include "environment_map.hh"
#include "scene.hh"
#include "scene_assets.hh"
#include "log.hh"
#include <memory>
#include <vector>

namespace tr
{
    struct scene_data
    {
        std::vector<scene_assets> assets;
        std::unique_ptr<scene> s;
    };

    scene_data load_scenes(context& ctx, const options& opt);
    context* create_context(const options& opt);
    void run(context& ctx, scene_data& s, options& opt);
}

#endif

