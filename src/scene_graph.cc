#include "scene_graph.hh"
#include "scene.hh"

namespace
{
using namespace tr;

void apply_transform(transformable_node& node, const mat4& transform)
{
    if(node.get_parent() == nullptr)
        node.set_transform(node.get_transform() * transform);
}

}

namespace tr
{

void scene_graph::to_scene(scene& s)
{
    for(auto& pair: mesh_objects) s.add(pair.second);
    for(auto& pair: directional_lights) s.add(pair.second);
    for(auto& pair: point_lights) s.add(pair.second);
    for(auto& pair: spotlights) s.add(pair.second);
    for(auto& pair: sh_grids) s.add(pair.second);
    for(auto& pair: control_nodes) s.add_control_node(pair.second);
}

void scene_graph::apply_transform(const mat4& transform)
{
    for(auto& pair: mesh_objects)
        ::apply_transform(pair.second, transform);
    for(auto& pair: directional_lights)
        ::apply_transform(pair.second, transform);
    for(auto& pair: point_lights)
        ::apply_transform(pair.second, transform);
    for(auto& pair: spotlights)
        ::apply_transform(pair.second, transform);
    for(auto& pair: sh_grids)
        ::apply_transform(pair.second, transform);
    for(auto& pair: control_nodes)
        ::apply_transform(pair.second, transform);
}

}
