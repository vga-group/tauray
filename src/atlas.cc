#include "atlas.hh"
#include "rectangle_packer.hh"

namespace tr
{

atlas::atlas(
    device_mask dev,
    const std::vector<uvec2>& sub_sizes,
    unsigned pad_size,
    vk::Format fmt,
    vk::ImageTiling tiling,
    vk::ImageUsageFlags usage,
    vk::ImageLayout layout
): texture(dev, uvec2(1,1), 1, fmt, 0, nullptr, tiling, usage, layout)
{
    set_sub_textures(sub_sizes, pad_size);
}

atlas::~atlas()
{
}

bool atlas::set_sub_textures(
    const std::vector<uvec2>& sub_sizes,
    unsigned pad_size
){
    // Check if nothing changed
    if(rects.size() == sub_sizes.size())
    {
        bool all_same = true;
        for(size_t i = 0; i < rects.size(); ++i)
        {
            if(uvec2(rects[i].z, rects[i].w) != sub_sizes[i])
            {
                all_same = false;
                break;
            }
        }
        if(all_same) return false;
    }

    // Figure out attempted texture size
    uvec2 attempt = get_size();
    if(get_size().x == 0 || get_size().y == 0)
    {
        uvec2 min_sides(0);
        unsigned min_area = 0;
        for(uvec2 sz: sub_sizes)
        {
            sz += pad_size;
            min_sides = max(sz, min_sides);
            min_area += sz.x*sz.y;
        }
        unsigned scale = 64;
        // Find the smallest plausible power-of-two atlas size
        while(
            scale < min_sides.x ||
            scale < min_sides.y ||
            scale*scale < min_area
        ) scale *= 2;
        attempt = uvec2(scale);
    }

    // Attempt to fit all rects into continually larger areas until it succeeds
    rects.clear();
    for(;;)
    {
        rect_packer rp(attempt.x, attempt.y, false);
        std::vector<rect_packer::rect> rp_rects;
        rp_rects.reserve(sub_sizes.size());
        for(uvec2 sz: sub_sizes)
        {
            sz += pad_size;
            rp_rects.push_back({(int)sz.x, (int)sz.y});
        }

        if((size_t)rp.pack(rp_rects.data(), rp_rects.size()) != rp_rects.size())
        {
            attempt *= 2;
            continue;
        }

        for(auto& r: rp_rects) rects.push_back(uvec4(r.x, r.y, r.w, r.h));
        break;
    }

    if(attempt != get_size())
        resize(attempt);
    return true;
}

uvec4 atlas::get_rect_px(unsigned i) const
{
    return rects[i];
}

vec4 atlas::get_rect(unsigned i) const
{
    return vec4(rects[i])/vec4(vec2(get_size()), vec2(get_size()));
}

unsigned atlas::get_sub_texture_count() const
{
    return rects.size();
}

}
