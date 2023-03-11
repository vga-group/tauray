#include "heightfield.hh"
#include "stb_image.h"

namespace tr
{

heightfield::heightfield(context& ctx, const std::string& path, vec3 scale)
: mesh(ctx)
{
    std::vector<mesh::vertex>& vertex_data = get_vertices();
    std::vector<uint32_t>& index_data = get_indices();

    stbi_set_flip_vertically_on_load(false);
    int w, h;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, nullptr, 1);

    if(!data)
        throw std::runtime_error("Failed to load height field ");

    vec3 origin = vec3(w*0.5f, 0.5, h*0.5f);

    vertex_data.reserve(w*h);
    for(int y = 0; y < h; ++y)
    for(int x = 0; x < w; ++x)
    {
        float height = data[y*w+x] / 255.0f;
        vec3 pos = (vec3(x, height, y)-origin) * scale;
        vertex_data.push_back({
            pos, pvec3(0),
            pvec2(x+0.5f, y+0.5f)/pvec2(w, h),
            pvec4(0)
        });
    }

    stbi_image_free(data);

    // Calculate normals
    for(int y = 0; y < h; ++y)
    for(int x = 0; x < w; ++x)
    {
        int prev_x = max(x-1, 0);
        int next_x = min(x+1, w-1);
        int prev_y = max(y-1, 0);
        int next_y = min(y+1, h-1);
        vec3 xdelta = vertex_data[y*w+prev_x].pos-vertex_data[y*w+next_x].pos;
        vec3 ydelta = vertex_data[next_y*w+x].pos-vertex_data[prev_y*w+x].pos;
        vertex_data[y*w+x].normal = normalize(cross(xdelta, ydelta));
        vertex_data[y*w+x].tangent = pvec4(normalize(xdelta), 1);
    }

    index_data.reserve(max(6*(h-1)*(w-1), 0));
    for(int y = 0; y < h-1; ++y)
    for(int x = 0; x < w-1; ++x)
    {
        int indices[] = {
            y*w+x, (y+1)*w+x, y*w+x+1,
            y*w+x+1, (y+1)*w+x, (y+1)*w+x+1
        };
        index_data.insert(index_data.end(), indices, indices+6);
    }

    refresh_buffers();
}

}
