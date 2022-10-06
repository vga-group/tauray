#include "gbuffer.hh"
#include "misc.hh"

namespace tr
{

size_t gbuffer_target::entry_count() const
{
    size_t count = 0;
#define TR_GBUFFER_ENTRY(name, ...) if(name) count++;
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
    return count;
}

render_target& gbuffer_target::operator[](size_t i)
{
    size_t index = 0;
#define TR_GBUFFER_ENTRY(name, ...) \
    if(i == index) return name;\
    index++;
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
    throw std::out_of_range("index out of range for gbuffer");
}

const render_target& gbuffer_target::operator[](size_t i) const
{
    return (*const_cast<gbuffer_target*>(this))[i];
}

void gbuffer_target::set_raster_layouts()
{
    visit([&](render_target& img){
        img.set_layout(vk::ImageLayout::eColorAttachmentOptimal);
    });
    depth.set_layout(vk::ImageLayout::eDepthAttachmentOptimal);
}

void gbuffer_target::set_layout(vk::ImageLayout layout)
{
    visit([&](render_target& img){ img.set_layout(layout); });
}

uvec2 gbuffer_target::get_size() const
{
#define TR_GBUFFER_ENTRY(name, ...) \
    if(name) return name.get_size();
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
    return uvec2(0);
}

unsigned gbuffer_target::get_layer_count() const
{
#define TR_GBUFFER_ENTRY(name, ...) \
    if(name) return name.get_layer_count();
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
    return 0;
}

vk::SampleCountFlagBits gbuffer_target::get_msaa() const
{
    vk::SampleCountFlagBits flags = vk::SampleCountFlagBits::e1;
#define TR_GBUFFER_ENTRY(name, ...) \
    if(name) flags = std::max(flags, name.get_msaa());
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
    return flags;
}

void gbuffer_target::get_location_defines(
    std::map<std::string, std::string>& defines,
    int start_index
) const {
#define TR_GBUFFER_ENTRY(name, ...) \
    if(name) {\
        defines[to_uppercase(#name) + "_TARGET_LOCATION"] = std::to_string(start_index);\
        start_index++;\
    }
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
}

gbuffer_spec gbuffer_target::get_spec() const
{
    gbuffer_spec ret;
#define TR_GBUFFER_ENTRY(name, format) \
    if(name) {\
        ret.name##_present = true; \
        ret.name##_format = name.get_format(); \
    }
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
    return ret;
}

void gbuffer_spec::set_all_usage(vk::ImageUsageFlags usage)
{
#define TR_GBUFFER_ENTRY(name, ...) name##_usage = usage;
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
}

size_t gbuffer_spec::present_count() const
{
    size_t count = 0;
#define TR_GBUFFER_ENTRY(name, ...) if(name##_present) count++;
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
    return count;
}

gbuffer_texture::gbuffer_texture(): dev(nullptr), size(0) {}
gbuffer_texture::gbuffer_texture(
    device_data& dev,
    uvec2 size,
    unsigned layer_count,
    vk::SampleCountFlagBits msaa
){
    reset(dev, size, layer_count, msaa);
}

void gbuffer_texture::reset(
    device_data& dev,
    uvec2 size,
    unsigned layer_count,
    vk::SampleCountFlagBits msaa
){
#define TR_GBUFFER_ENTRY(name, ...) name.reset();
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
    this->dev = &dev;
    this->size = size;
    this->layer_count = layer_count;
    this->msaa = msaa;
}

#define TR_GBUFFER_ENTRY(name, ...) \
    void gbuffer_texture::add_##name(\
        vk::ImageUsageFlags usage, vk::Format fmt\
    ){\
        auto layout = vk::ImageLayout::eGeneral;\
        if(usage & vk::ImageUsageFlagBits::eColorAttachment)\
            layout = vk::ImageLayout::eColorAttachmentOptimal;\
        else if(usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) \
            layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;\
        name.reset(new texture(\
            *dev,\
            size,\
            layer_count,\
            fmt,\
            0, nullptr,\
            vk::ImageTiling::eOptimal,\
            usage,\
            layout,\
            msaa\
        ));\
    }\
    bool gbuffer_texture::has_##name() const\
    {\
        return name.get() != nullptr;\
    }
TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY

void gbuffer_texture::add(gbuffer_spec spec)
{
#define TR_GBUFFER_ENTRY(name, ...) \
    if(spec.name##_present) add_##name(spec.name##_usage, spec.name##_format);
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
}

gbuffer_target gbuffer_texture::get_array_target() const
{
    gbuffer_target gbuf;
#define TR_GBUFFER_ENTRY(name, ...) \
    if(name) gbuf.name = name->get_array_render_target(0);
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
    return gbuf;
}

gbuffer_target gbuffer_texture::get_layer_target(uint32_t layer_index) const
{
    gbuffer_target gbuf;
#define TR_GBUFFER_ENTRY(name, ...) \
    if(name) gbuf.name = name->get_layer_render_target(layer_index, 0);
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
    return gbuf;
}

gbuffer_target gbuffer_texture::get_multiview_block_target(uint32_t block_index) const
{
    gbuffer_target gbuf;
#define TR_GBUFFER_ENTRY(name, ...) \
    if(name) gbuf.name = name->get_multiview_block_render_target(block_index, 0);
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
    return gbuf;
}

size_t gbuffer_texture::entry_count() const
{
    size_t count = 0;
#define TR_GBUFFER_ENTRY(name, ...) if(name.get() != nullptr) count++;
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
    return count;
}

size_t gbuffer_texture::get_layer_count() const
{
    return layer_count;
}

size_t gbuffer_texture::get_multiview_block_count() const
{
    gbuffer_target gbuf;
#define TR_GBUFFER_ENTRY(name, ...) \
    if(name) return name->get_multiview_block_count();
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
    return 0;
}

}
