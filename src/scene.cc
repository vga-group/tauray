#include "scene.hh"
#include "camera.hh"
#include "misc.hh"
#include "placeholders.hh"
#include "environment_map.hh"

namespace tr
{

scene::scene(
    device_mask dev,
    size_t max_instances,
    size_t max_lights
):  light_scene(dev, max_lights),
    mesh_scene(dev, max_instances),
    dev(dev),
    total_ticks(0)
{
}

void scene::set_camera(camera* cam) { cameras = {cam}; }

camera* scene::get_camera(unsigned index) const
{
    if(index >= cameras.size()) return nullptr;
    return cameras[index];
}

void scene::add(camera& c) { unsorted_insert(cameras, &c); }
void scene::remove(camera& c) { unsorted_erase(cameras, &c); }
void scene::clear_cameras() { cameras.clear(); }
const std::vector<camera*>& scene::get_cameras() const { return cameras; }

void scene::reorder_cameras_by_active(const std::set<int>& active_indices)
{
    std::vector<camera*> new_cameras;
    for(size_t i = 0; i < cameras.size(); ++i)
    {
        if(active_indices.count(i))
            new_cameras.push_back(cameras[i]);
    }
    for(size_t i = 0; i < cameras.size(); ++i)
    {
        if(!active_indices.count(i))
            new_cameras.push_back(cameras[i]);
    }
    cameras = std::move(new_cameras);
}

void scene::set_camera_jitter(const std::vector<vec2>& jitter)
{
    for(camera* cam: cameras)
        cam->set_jitter(jitter);
}

void scene::add_control_node(animated_node& o)
{ sorted_insert(control_nodes, &o); }
void scene::remove_control_node(animated_node& o)
{ sorted_erase(control_nodes, &o); }
void scene::clear_control_nodes() { control_nodes.clear(); }

void scene::clear()
{
    clear_cameras();
    clear_mesh_objects();
    clear_point_lights();
    clear_spotlights();
    clear_directional_lights();
    clear_control_nodes();
}

void scene::play(const std::string& name, bool loop, bool use_fallback)
{
    auto play_handler = [&](animated_node* n){
        n->play(name, loop, use_fallback);
    };
    for(camera* c: cameras) play_handler(c);
    for(animated_node* o: control_nodes) play_handler(o);

    light_scene::visit_animated(play_handler);
    mesh_scene::visit_animated(play_handler);
}

void scene::update(time_ticks dt, bool force_update)
{
    for(camera* c: cameras)
    {
        c->step_jitter();
    }

    if(dt > 0 || force_update)
    {
        auto update_handler = [&](animated_node* n){
            n->update(dt);
        };
        for(camera* c: cameras)
            update_handler(c);
        for(animated_node* o: control_nodes) update_handler(o);
        light_scene::visit_animated(update_handler);
        mesh_scene::visit_animated(update_handler);
    }
    total_ticks += dt;
}

void scene::set_animation_time(time_ticks dt)
{
    auto reset_handler = [&](animated_node* n){
        n->restart();
        n->update(dt);
    };
    for(camera* c: cameras) reset_handler(c);
    for(animated_node* o: control_nodes) reset_handler(o);

    light_scene::visit_animated(reset_handler);
    mesh_scene::visit_animated(reset_handler);
    total_ticks = dt;
}

time_ticks scene::get_total_ticks() const
{
    return total_ticks;
}

bool scene::is_playing() const
{
    bool playing = false;
    auto check_playing = [&](animated_node* n){
        if(n->is_playing()) playing = true;
    };
    for(camera* c: cameras) check_playing(c);
    for(animated_node* o: control_nodes) check_playing(o);

    light_scene::visit_animated(check_playing);
    mesh_scene::visit_animated(check_playing);

    return playing;
}

std::vector<uint32_t> get_viewport_reorder_mask(
    const std::set<int>& active_indices,
    size_t viewport_count
){
    std::vector<uint32_t> reorder;
    for(size_t i = 0; i < viewport_count; ++i)
    {
        if(active_indices.count(i))
            reorder.push_back(i);
    }
    for(size_t i = 0; i < viewport_count; ++i)
    {
        if(!active_indices.count(i))
            reorder.push_back(i);
    }
    return reorder;
}

}
