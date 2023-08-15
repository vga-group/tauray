#ifndef TAURAY_GLTF_HH
#define TAURAY_GLTF_HH
#include "scene_assets.hh"
#include "scene.hh"

namespace tr
{

scene_assets load_gltf(
    device_mask dev,
    scene& s,
    const std::string& path,
    bool force_single_sided = false,
    bool force_double_sided = false
);

}

#endif

