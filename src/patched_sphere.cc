#include "patched_sphere.hh"

namespace
{

using namespace tr;

pvec3 swizzle_for_cube_face(pvec3 p, int face)
{
    switch(face)
    {
    case 0:
        return pvec3(p.z, p.y, -p.x);
    case 1:
        return pvec3(-p.z, p.y, p.x);
    case 2:
        return pvec3(p.x, p.z, -p.y);
    case 3:
        return pvec3(p.x, -p.z, p.y);
    default:
    case 4:
        return p;
    case 5:
        return pvec3(-p.x, p.y, -p.z);
    }
}

}

namespace tr
{

patched_sphere::patched_sphere(
    context& ctx, unsigned subdivisions, float radius
): mesh(ctx)
{
    std::vector<mesh::vertex>& vertex_data = get_vertices();
    std::vector<uint32_t>& index_data = get_indices();
    float start = -1.0f;
    float step = 2.0f / subdivisions;

    // Generate indexed points on a subdivided cube, but normalize points.
    for(unsigned face = 0; face < 6; ++face)
    {
        // Iterate subdivided faces
        for(unsigned j = 0; j < subdivisions; ++j)
        {
            for(unsigned i = 0; i < subdivisions; ++i)
            {
                float s1 = start + step * j;
                float t1 = start + step * i;
                float s2 = s1 + step;
                float t2 = t1 + step;

                pvec3 face_vertices[] = {
                    pvec3(s1,t1,1.0f),
                    pvec3(s1,t2,1.0f),
                    pvec3(s2,t1,1.0f),
                    pvec3(s2,t2,1.0f)
                };

                pvec2 face_uvs[] = {
                    pvec2(s1,-t1)*0.5f+0.5f,
                    pvec2(s1,-t2)*0.5f+0.5f,
                    pvec2(s2,-t1)*0.5f+0.5f,
                    pvec2(s2,-t2)*0.5f+0.5f
                };

                uint32_t face_indices[] = {
                    2, 1, 0, 3, 1, 2
                };

                for(unsigned k = 0; k < 4; ++k)
                {
                    pvec3& v = face_vertices[k];
                    v = swizzle_for_cube_face(
                        normalize(v) * radius,
                        face
                    );
                }

                for(uint32_t index: face_indices)
                {
                    pvec3 p = face_vertices[index];
                    pvec2 uv = face_uvs[index];
                    auto it = std::find_if(
                        vertex_data.begin(), vertex_data.end(),
                        [p, uv](vertex& v){
                            return v.uv == uv && dot(v.pos, p) >= 0.9999f;
                        }
                    );
                    index_data.push_back(
                        std::distance(vertex_data.begin(), it)
                    );
                    if(it == vertex_data.end())
                    {
                        vec3 normal = normalize(p);
                        vec3 tangent = normalize(
                            cross(normal, face != 2 && face != 3 ? vec3(0,1,0) : vec3(0,0,1))
                        );
                        vertex_data.push_back({
                            p, normal, uv, vec4(tangent, 1)
                        });
                    }
                }
            }
        }
    }

    refresh_buffers();
}

}
