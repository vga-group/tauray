#include "render_target.hh"
#include "misc.hh"

namespace tr
{

render_target::render_target(
    uvec2 size,
    unsigned base_layer,
    unsigned layer_count,
    vk::Image image,
    vk::ImageView view,
    vk::ImageLayout layout,
    vk::Format format,
    vk::SampleCountFlagBits msaa
):  size(size), base_layer(base_layer), layer_count(layer_count),
    msaa(msaa), format(format), image(image), view(view), layout(layout)
{
}

render_target::operator bool() const
{
    return !!image;
}

void render_target::transition_layout_temporary(
    vk::CommandBuffer cb,
    vk::ImageLayout layout,
    bool ignore_src_stage_mask,
    bool ignore_dst_stage_mask
){
    transition_image_layout(
        cb, image, format, this->layout, layout, 0, 1,
        base_layer, layer_count, ignore_src_stage_mask, ignore_dst_stage_mask
    );
}

vk::ImageSubresourceLayers render_target::get_layers() const
{
    return {deduce_aspect_mask(format), 0, base_layer, layer_count};
}

vk::ImageSubresourceRange render_target::get_range() const
{
    return {deduce_aspect_mask(format), 0, 1, base_layer, layer_count};
}

}
