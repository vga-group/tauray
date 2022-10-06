#ifndef TAURAY_MESH_OBJECT_HH
#define TAURAY_MESH_OBJECT_HH
#include "transformable.hh"
#include "animation.hh"
#include "model.hh"
#include <memory>

namespace tr
{

class mesh_object: public animated_node
{
public:
    mesh_object(const model* mod = nullptr);
    ~mesh_object();

    void set_model(const model* mod = nullptr);
    const model* get_model() const;

    void set_shadow_terminator_offset(float offset = 0.0f);
    float get_shadow_terminator_offset() const;

private:
    const model* mod;
    float shadow_terminator_offset;
};

}
#endif

