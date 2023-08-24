#ifndef TAURAY_ASSIMP_HH
#define TAURAY_ASSIMP_HH
#include "scene_assets.hh"
#include "scene.hh"

namespace tr
{

scene_assets load_assimp(device_mask dev, scene& s, const std::string& path);

}

#endif
