#ifndef TAURAY_GLTF_HH
#define TAURAY_GLTF_HH
#include "scene_graph.hh"

namespace tr
{

scene_graph load_gltf(
    context& ctx,
    const std::string& path,
    bool force_single_sided = false,
    bool force_double_sided = false
);

}

#endif

