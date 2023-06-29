#ifndef TR_ATLAS_HH
#define TR_ATLAS_HH
#include "texture.hh"
#include <vector>

namespace tr
{

// Texture atlas.
class atlas: public texture
{
public:
    // If sub_sizes is empty, a 1x1 texture is created so that samplers don't
    // cause errors.
    atlas(
        device_mask dev,
        const std::vector<uvec2>& sub_sizes = {},
        unsigned pad_size = 1,
        vk::Format fmt = vk::Format::eR8G8B8A8Unorm,
        vk::ImageTiling tiling = vk::ImageTiling::eOptimal,
        vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled,
        vk::ImageLayout layout = vk::ImageLayout::eGeneral
    );
    ~atlas();

    // This may be a no-op if the sub_sizes match current sizes. First tries to
    // re-accommodate all sub sizes in the current texture, then allocates a
    // larger one if that doesn't work. Returns true if the layout was
    // changed or the texture was recreated.
    bool set_sub_textures(
        const std::vector<uvec2>& sub_sizes,
        unsigned pad_size = 1
    );

    uvec4 get_rect_px(unsigned i) const;
    vec4 get_rect(unsigned i) const;
    unsigned get_sub_texture_count() const;

private:
    std::vector<uvec4> rects;
};

}

#endif
