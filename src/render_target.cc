#include "render_target.hh"
#include "misc.hh"

namespace tr
{

render_target::render_target(
    uvec2 size,
    unsigned base_layer,
    unsigned layer_count,
    const std::vector<frame>& frames,
    vk::Format format,
    vk::SampleCountFlagBits msaa
):  size(size), base_layer(base_layer), layer_count(layer_count),
    msaa(msaa), format(format), frames(frames)
{
}

render_target::operator bool() const
{
    return frames.size();
}

render_target::frame render_target::operator[](size_t index) const
{
    index = std::min(index, frames.size()-1);
    return frames[index];
}

size_t render_target::get_frame_count() const
{
    return frames.size();
}

vk::ImageLayout render_target::get_layout(size_t index) const
{
    index = std::min(index, frames.size()-1);
    return frames[index].layout;
}

void render_target::set_layout(vk::ImageLayout layout)
{
    for(frame& f: frames)
        f.layout = layout;
}

void render_target::transition_layout_save(
    vk::CommandBuffer cb,
    size_t index,
    vk::ImageLayout layout,
    bool ignore_src_stage_mask,
    bool ignore_dst_stage_mask
){
    index = std::min(index, frames.size()-1);
    transition_layout_temporary(
        cb, index, layout, ignore_src_stage_mask, ignore_dst_stage_mask
    );
    frames[index].layout = layout;
}

void render_target::transition_layout_temporary(
    vk::CommandBuffer cb,
    size_t index,
    vk::ImageLayout layout,
    bool ignore_src_stage_mask,
    bool ignore_dst_stage_mask
){
    index = std::min(index, frames.size()-1);
    transition_image_layout(
        cb, frames[index].image, format, frames[index].layout, layout, 0, 1,
        base_layer, layer_count, ignore_src_stage_mask, ignore_dst_stage_mask
    );
}

void render_target::delay()
{
    std::rotate(frames.rbegin(), frames.rbegin() + 1, frames.rend());
}

vk::Format render_target::get_format() const
{
    return format;
}

vk::SampleCountFlagBits render_target::get_msaa() const
{
    return msaa;
}

uvec2 render_target::get_size() const
{
    return size;
}

unsigned render_target::get_base_layer() const
{
    return base_layer;
}

unsigned render_target::get_layer_count() const
{
    return layer_count;
}

vk::ImageSubresourceLayers render_target::get_layers() const
{
    return {vk::ImageAspectFlagBits::eColor, 0, base_layer, layer_count};
}

vk::ImageSubresourceRange render_target::get_range() const
{
    return {vk::ImageAspectFlagBits::eColor, 0, 1, base_layer, layer_count};
}

}
