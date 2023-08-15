#ifndef TAURAY_SCENE_ASSETS_HH
#define TAURAY_SCENE_ASSETS_HH
#include "animation.hh"
#include "mesh.hh"
#include "texture.hh"
#include <memory>
#include <vector>
#include <string>

namespace tr
{

struct name_component
{
    std::string name;
};

struct scene_assets
{
    std::vector<std::unique_ptr<texture>> textures;
    std::vector<std::unique_ptr<mesh>> meshes;
    std::vector<std::unique_ptr<animation_pool>> animation_pools;
};

}

#endif
