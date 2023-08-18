#ifndef TAURAY_TEXTURE_HH
#define TAURAY_TEXTURE_HH
#include "context.hh"
#include "render_target.hh"

namespace tr
{

class texture
{
public:
    texture(device_mask dev, const std::string& path);
    // If no data is given, it is assumed that the texture will be a render
    // target!
    texture(
        device_mask dev,
        uvec2 size,
        unsigned array_layers,
        vk::Format fmt,
        size_t data_size = 0,
        void* data = nullptr,
        vk::ImageTiling tiling = vk::ImageTiling::eOptimal,
        vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled,
        vk::ImageLayout layout = vk::ImageLayout::eGeneral,
        vk::SampleCountFlagBits msaa = vk::SampleCountFlagBits::e1
    );
    texture(
        device_mask dev,
        uvec3 dim,
        vk::Format fmt,
        vk::ImageTiling tiling = vk::ImageTiling::eOptimal,
        vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled,
        vk::ImageLayout layout = vk::ImageLayout::eGeneral
    );
    texture(const context& other) = delete;
    texture(texture&& other);

    vk::ImageView get_array_image_view(device_id id) const;
    vk::ImageView get_layer_image_view(device_id id, uint32_t layer_index) const;
    vk::ImageView get_image_view(device_id id) const;
    vk::Image get_image(device_id id) const;

    vk::Format get_format() const;
    vk::SampleCountFlagBits get_msaa() const;
    vk::ImageLayout get_layout() const;

    void set_opaque(bool opaque);
    bool potentially_transparent() const;

    uvec2 get_size() const;
    uvec3 get_dimensions() const;

    render_target get_array_render_target(device_id id) const;
    render_target get_layer_render_target(device_id id, uint32_t layer_index) const;
    render_target get_multiview_block_render_target(device_id id, uint32_t block_index) const;
    size_t get_multiview_block_count() const;

    device_mask get_mask() const;

protected:
    // This can be dangerous, it discards the previous image & view along with
    // the data.
    void resize(uvec2 size);

private:
    // Also creates mip chain.
    void load_from_file(const std::string& path);
    void create(size_t data_size, void* data);

    uvec3 dim;
    unsigned array_layers;
    vk::Format fmt;
    vk::ImageType type;
    vk::ImageTiling tiling;
    vk::ImageUsageFlags usage;
    vk::ImageLayout layout;
    vk::SampleCountFlagBits msaa;
    std::vector<uint8_t> pixel_data;
    bool opaque;

    struct buffer_data
    {
        vkm<vk::Image> img;
        vkm<vk::ImageView> array_view;
        std::vector<vkm<vk::ImageView>> layer_views;
        std::vector<vkm<vk::ImageView>> multiview_block_views;
    };
    per_device<buffer_data> buffers;
};

}

#endif
