#ifndef TAURAY_ASSIMP_HH
#define TAURAY_ASSIMP_HH
#include "scene_graph.hh"

namespace tr
{

scene_graph load_assimp(device_mask dev, const std::string& path);

}

#endif
