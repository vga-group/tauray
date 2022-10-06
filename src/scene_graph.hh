#ifndef TAURAY_SCENE_GRAPH_HH
#define TAURAY_SCENE_GRAPH_HH
#include "mesh.hh"
#include "animation.hh"
#include "mesh_object.hh"
#include "texture.hh"
#include "sh_grid.hh"
#include "light.hh"
#include "camera.hh"
#include <memory>
#include <vector>
#include <map>

namespace tr
{

class scene;
struct scene_graph
{
    scene_graph() = default;
    scene_graph(const scene_graph&) = delete;
    scene_graph(scene_graph&&) = default;
    scene_graph& operator=(scene_graph&&) = default;

    std::vector<std::unique_ptr<texture>> textures;
    std::vector<std::unique_ptr<mesh>> meshes;
    std::vector<std::unique_ptr<animation_pool>> animation_pools;
    std::map<std::string, model> models;
    std::map<std::string, mesh_object> mesh_objects;
    std::map<std::string, animated_node> control_nodes;
    std::map<std::string, directional_light> directional_lights;
    std::map<std::string, point_light> point_lights;
    std::map<std::string, spotlight> spotlights;
    std::map<std::string, sh_grid> sh_grids;
    std::map<std::string, camera> cameras;
    std::vector<std::unique_ptr<model>> animation_models;

    void to_scene(scene& s);
    void apply_transform(const mat4& transform);
};

}

#endif


