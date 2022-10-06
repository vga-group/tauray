#ifndef TAURAY_RENDER_TARGET_HH
#define TAURAY_RENDER_TARGET_HH
#include "vkm.hh"
#include "math.hh"

namespace tr
{

class render_target
{
public:
    struct frame
    {
        vk::Image image;
        vk::ImageView view;
        vk::ImageLayout layout;
    };

    render_target() = default;

    render_target(
        uvec2 size,
        unsigned base_layer,
        unsigned layer_count,
        const std::vector<frame>& frames,
        vk::Format format,
        vk::SampleCountFlagBits msaa = vk::SampleCountFlagBits::e1
    );

    operator bool() const;
    frame operator[](size_t index) const;
    size_t get_frame_count() const;

    vk::ImageLayout get_layout(size_t index = 0) const;
    void transition_layout_save(
        vk::CommandBuffer cb,
        size_t index,
        vk::ImageLayout layout,
        bool ignore_src_stage_mask = false,
        bool ignore_dst_stage_mask = false
    );
    // Does not save the new layout, but assumes that it's reset every time.
    void transition_layout_temporary(
        vk::CommandBuffer cb,
        size_t index,
        vk::ImageLayout layout,
        bool ignore_src_stage_mask = false,
        bool ignore_dst_stage_mask = false
    );
    // Use this to mark the target's layout from that point on, if it's not
    // being changed through transition_layout.
    void set_layout(vk::ImageLayout layout);

    // Rotates the internal frame buffer by one step.
    void delay();

    vk::Format get_format() const;
    vk::SampleCountFlagBits get_msaa() const;
    uvec2 get_size() const;
    unsigned get_base_layer() const;
    unsigned get_layer_count() const;
    vk::ImageSubresourceLayers get_layers() const;
    vk::ImageSubresourceRange get_range() const;

private:
    uvec2 size;
    unsigned base_layer;
    unsigned layer_count;
    vk::SampleCountFlagBits msaa;
    vk::Format format;
    std::vector<frame> frames;
};

}

#endif
