#ifndef TAURAY_RENDER_TARGET_HH
#define TAURAY_RENDER_TARGET_HH
#include "vkm.hh"
#include "math.hh"

namespace tr
{

struct render_target
{
public:
    render_target() = default;

    render_target(
        uvec2 size,
        unsigned base_layer,
        unsigned layer_count,
        vk::Image image,
        vk::ImageView view,
        vk::ImageLayout layout,
        vk::Format format,
        vk::SampleCountFlagBits msaa = vk::SampleCountFlagBits::e1
    );

    operator bool() const;

    // Does not save the new layout, but assumes that it's reset every time.
    void transition_layout_temporary(
        vk::CommandBuffer cb,
        vk::ImageLayout layout,
        bool ignore_src_stage_mask = false,
        bool ignore_dst_stage_mask = false
    );

    void transition_layout(
        vk::CommandBuffer cb,
        vk::ImageLayout from,
        vk::ImageLayout to,
        bool ignore_src_stage_mask = false,
        bool ignore_dst_stage_mask = false
    );

    vk::ImageSubresourceLayers get_layers() const;
    vk::ImageSubresourceRange get_range() const;

    uvec2 size;
    unsigned base_layer;
    unsigned layer_count;
    vk::SampleCountFlagBits msaa;
    vk::Format format;
    vk::Image image = VK_NULL_HANDLE;
    vk::ImageView view = VK_NULL_HANDLE;
    vk::ImageLayout layout;
};

}

#endif
