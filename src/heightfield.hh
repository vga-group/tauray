#ifndef TAURAY_HEIGHT_FIELD_HH
#define TAURAY_HEIGHT_FIELD_HH
#include "mesh.hh"

namespace tr
{

class heightfield: public mesh
{
public:
    heightfield(context& ctx, const std::string& path, vec3 scale = vec3(1.0f));
};

}

#endif

