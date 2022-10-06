#include "material.hh"
#include "texture.hh"

namespace tr
{

bool material::potentially_transparent() const
{
    return transmittance > 0.0f || albedo_factor.a < 1.0f ||
        (albedo_tex.first && albedo_tex.first->potentially_transparent());
}

}
