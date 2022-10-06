#ifndef TAURAY_HH
#define TAURAY_HH
#include "options.hh"
#include "environment_map.hh"
#include "scene.hh"
#include "ply.hh"
#include <memory>
#include <vector>

namespace tr
{
    struct scene_data
    {
        std::unique_ptr<environment_map> sky;
        std::vector<scene_graph> scenes;
        std::unique_ptr<scene> s;
        std::unique_ptr<ply_streamer> ply_stream;
    };

    scene_data load_scenes(context& ctx, const options& opt);
    context* create_context(const options& opt);
    void run(context& ctx, scene_data& s, options& opt);
}

#endif

