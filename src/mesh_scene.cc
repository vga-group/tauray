#include "mesh_scene.hh"
#include "misc.hh"
#include <unordered_set>

namespace tr
{

void mesh_scene::add(mesh_object& o)
{
    sorted_insert(objects, &o);
}

void mesh_scene::remove(mesh_object& o)
{
    sorted_erase(objects, &o);
}

void mesh_scene::clear_mesh_objects()
{
    objects.clear();
}

const std::vector<mesh_object*>& mesh_scene::get_mesh_objects() const
{
    return objects;
}

size_t mesh_scene::get_instance_count() const
{
    size_t total = 0;
    for(const mesh_object* o: objects)
    {
        if(!o) continue;
        const model* m = o->get_model();
        if(!m) continue;
        total += m->group_count();
    }
    return total;
}

size_t mesh_scene::get_sampler_count() const
{
    std::unordered_set<
        combined_tex_sampler, combined_tex_sampler_hash
    > samplers;

    for(const mesh_object* o: objects)
    {
        if(!o) continue;
        const model* m = o->get_model();
        if(!m) continue;
        for(const auto& group: *m)
        {
            samplers.insert(group.mat.albedo_tex);
            samplers.insert(group.mat.metallic_roughness_tex);
            samplers.insert(group.mat.normal_tex);
            samplers.insert(group.mat.emission_tex);
        }
    }
    return samplers.size();
}

}
