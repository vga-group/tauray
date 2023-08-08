#ifndef TAURAY_MESH_SCENE_HH
#define TAURAY_MESH_SCENE_HH
#include "mesh.hh"
#include "acceleration_structure.hh"
#include "mesh_object.hh"
#include "timer.hh"
#include <unordered_map>

namespace tr
{

class mesh_scene
{
public:
    void add(mesh_object& o);
    void remove(mesh_object& o);
    void clear_mesh_objects();
    const std::vector<mesh_object*>& get_mesh_objects() const;

    // These can be very slow!
    size_t get_instance_count() const;
    size_t get_sampler_count() const;

protected:
    template<typename F>
    void visit_animated(F&& f) const
    {
        for(mesh_object* o: objects)
        {
            if(!o->is_static())
                f(o);
        }
    }

private:
    std::vector<mesh_object*> objects;
};

}

#endif
