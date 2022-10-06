#ifndef TAURAY_ENVIRONMENT_MAP_HH
#define TAURAY_ENVIRONMENT_MAP_HH
#include "texture.hh"
#include "transformable.hh"

namespace tr
{

class environment_map: public texture, public transformable_node
{
public:
    enum projection
    {
        LAT_LONG = 0
    };

    environment_map(
        context& ctx,
        const std::string& path,
        projection proj = LAT_LONG,
        vec3 factor = vec3(1.0f)
    );

    void set_factor(vec3 factor);
    vec3 get_factor() const;

    projection get_projection() const;

private:
    vec3 factor;
    projection proj;
};

}

#endif

