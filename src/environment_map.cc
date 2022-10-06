#include "environment_map.hh"

namespace tr
{

environment_map::environment_map(
    context& ctx, const std::string& path, projection proj, vec3 factor
): texture(ctx, path), factor(factor), proj(proj)
{
}

void environment_map::set_factor(vec3 factor)
{
    this->factor = factor;
}

vec3 environment_map::get_factor() const
{
    return factor;
}

environment_map::projection environment_map::get_projection() const
{
    return proj;
}

}
