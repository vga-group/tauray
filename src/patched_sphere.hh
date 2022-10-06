#ifndef TAURAY_PATCHED_SPHERE_HH
#define TAURAY_PATCHED_SPHERE_HH
#include "mesh.hh"

namespace tr
{

// https://www.iquilezles.org/www/articles/patchedsphere/patchedsphere.htm
// Sphere particularly suitable for cubemap visualization.
class patched_sphere: public mesh
{
public:
    patched_sphere(
        context& ctx, unsigned subdivisions = 8, float radius = 1.0f
    );
};

}

#endif
