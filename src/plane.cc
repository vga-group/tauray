#include "plane.hh"
#include "stb_image.h"

namespace tr
{

plane::plane(context& ctx, vec2 size)
: mesh(ctx)
{
    std::vector<mesh::vertex>& vertex_data = get_vertices();
    std::vector<uint32_t>& index_data = get_indices();

    vec2 r = size*0.5f;
    vertex_data = {
        {pvec3(-r.x, 0, -r.y), pvec3(0,1,0), pvec2(0,0), vec4(1,0,0,1)},
        {pvec3( r.x, 0, -r.y), pvec3(0,1,0), pvec2(1,0), vec4(1,0,0,1)},
        {pvec3(-r.x, 0,  r.y), pvec3(0,1,0), pvec2(0,1), vec4(1,0,0,1)},
        {pvec3( r.x, 0,  r.y), pvec3(0,1,0), pvec2(1,1), vec4(1,0,0,1)}
    };
    index_data = {0,2,1,1,2,3};

    refresh_buffers();
}

}

