#include "mesh_object.hh"

namespace tr
{

mesh_object::mesh_object(const model* mod): mod(mod), shadow_terminator_offset(0.0f) {}
mesh_object::~mesh_object() {}

void mesh_object::set_model(const model* mod)
{
    this->mod = mod;
}

const model* mesh_object::get_model() const
{
    return mod;
}

void mesh_object::set_shadow_terminator_offset(float offset)
{
    shadow_terminator_offset = offset;
}

float mesh_object::get_shadow_terminator_offset() const
{
    return shadow_terminator_offset;
}

}
