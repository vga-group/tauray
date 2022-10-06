#ifndef TAURAY_PLANE_HH
#define TAURAY_PLANE_HH
#include "mesh.hh"

namespace tr
{

class plane: public mesh
{
public:
    plane(context& ctx, vec2 size = vec2(1.0f));
};

}

#endif
