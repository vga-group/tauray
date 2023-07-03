#include "texture.hh"
#include "stb_image.h"
#include "misc.hh"
#include <filesystem>
#include "tinyexr.h"
namespace fs = std::filesystem;

namespace
{
using namespace tr;

void create_tex(
    device_data& dev,
    unsigned array_layers,
    vkm<vk::Image>& img,
    vkm<vk::ImageView>& array_view,
    std::vector<vkm<vk::ImageView>>& layer_views,
    std::vector<vkm<vk::ImageView>>& multiview_block_views,
    vk::ImageCreateInfo info,
    vk::ImageLayout layout,
    size_t data_size,
    void* pixel_data
){
    layer_views.clear();
    multiview_block_views.clear();
    img = sync_create_gpu_image(
        dev,
        info,
        layout,
        data_size,
        pixel_data
    );

    vk::ImageViewType base_type = info.imageType == vk::ImageType::e3D ?
            vk::ImageViewType::e3D :
            vk::ImageViewType::e2D;
    vk::ImageViewType array_type = base_type;
    if(base_type == vk::ImageViewType::e2D)
        array_type = vk::ImageViewType::e2DArray;

    vk::ImageViewCreateInfo view_info = {
        {},
        img,
        array_type,
        info.format,
        {},
        {
            deduce_aspect_mask(info.format),
            0, info.mipLevels, 0, VK_REMAINING_ARRAY_LAYERS
        }
    };
    array_view = vkm(dev, dev.dev.createImageView(view_info));

    view_info.viewType = base_type;
    for(unsigned i = 0; i < array_layers; ++i)
    {
        view_info.subresourceRange.baseArrayLayer = i;
        view_info.subresourceRange.layerCount = 1;
        layer_views.emplace_back(dev, dev.dev.createImageView(view_info));
    }

    view_info.viewType = array_type;
    for(unsigned i = 0; i < array_layers; i += dev.mv_props.maxMultiviewViewCount)
    {
        view_info.subresourceRange.baseArrayLayer = i;
        view_info.subresourceRange.layerCount = min(
            dev.mv_props.maxMultiviewViewCount,
            array_layers - i
        );
        multiview_block_views.emplace_back(
            dev,
            dev.dev.createImageView(view_info)
        );
    }
}

void insert_strided(
    std::vector<uint8_t>& data,
    size_t entry_size,
    size_t fill_data_size,
    void* fill_data
){
    size_t new_entry_size = entry_size + fill_data_size;
    size_t entry_count = (data.size()/entry_size);
    std::vector<uint8_t> new_data(entry_count*new_entry_size);
    for(size_t i = 0; i < entry_count; ++i)
    {
        memcpy(
            new_data.data() + i * new_entry_size,
            data.data() + i * entry_size,
            entry_size
        );
        memcpy(
            new_data.data() + i * new_entry_size + entry_size,
            fill_data,
            fill_data_size
        );
    }

    data = std::move(new_data);
}

void float_to_half(std::vector<uint8_t>& buffer)
{
    std::vector<uint8_t> new_buffer(buffer.size()/2);

    for(
        size_t i = 0, j = 0;
        i < buffer.size();
        i += sizeof(float), j += sizeof(float)/2
    ){
        // Clamp to a range that shouldn't overflow in a half.
        float f = clamp(
            *(float*)(buffer.data() + i),
            -65000.0f, 65000.0f
        );
        *(uint16_t*)(new_buffer.data() + j) = tr::float_to_half(f);
    }

    buffer = std::move(new_buffer);
}

float* read_exr(
    int& width,
    int& height,
    int& channel_count,
    const std::string& path
){
    EXRVersion exr_version;

    int ret = 0;

    ret = ParseEXRVersionFromFile(&exr_version, path.c_str());
    if(ret != TINYEXR_SUCCESS || exr_version.multipart || exr_version.non_image)
        return nullptr;

    EXRHeader exr_header;
    InitEXRHeader(&exr_header);

    ret = ParseEXRHeaderFromFile(&exr_header, &exr_version, path.c_str(), nullptr);
    if(ret != TINYEXR_SUCCESS)
        return nullptr;

    for(int i = 0; i < exr_header.num_channels; ++i)
        if(exr_header.pixel_types[i] == TINYEXR_PIXELTYPE_HALF)
            exr_header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;

    EXRImage exr_image;
    InitEXRImage(&exr_image);

    ret = LoadEXRImageFromFile(&exr_image, &exr_header, path.c_str(), nullptr);
    if(ret != TINYEXR_SUCCESS)
        return nullptr;

    int cid[4] = {-1, -1, -1, -1};
    bool naming_rgba = true;
    channel_count = min(exr_header.num_channels, 4);
    if(channel_count == 0) return nullptr;

    for(int c = 0; c < exr_header.num_channels; ++c)
    {
        auto name = exr_header.channels[c].name;
        if(name[0] == 'R')
            cid[0] = c;
        else if(name[0] == 'G')
            cid[1] = c;
        else if(name[0] == 'B')
            cid[2] = c;
        else if(name[0] == 'A')
            cid[3] = c;
        else naming_rgba = false;
    }

    if(!naming_rgba) for(int c = 0; c < exr_header.num_channels; ++c)
        cid[c] = c;

    float* data = (float*)malloc(
        channel_count * exr_image.width * exr_image.height * sizeof(float)
    );

    if(exr_header.tiled)
    {
        for(int t = 0; t < exr_image.num_tiles; ++t)
        for(int ty = 0; ty < exr_header.tile_size_y; ++ty)
        for(int tx = 0; tx < exr_header.tile_size_x; ++tx)
        {
            int x = exr_image.tiles[t].offset_x * exr_header.tile_size_x + tx;
            int y = exr_image.tiles[t].offset_y * exr_header.tile_size_y + ty;
            int i = x + y * exr_image.width;

            if(x >= exr_image.width) continue;
            if(y >= exr_image.height) continue;

            float** src = (float**)exr_image.tiles[t].images;
            int src_i = tx + ty * exr_header.tile_size_x;

            for(int c = 0; c < channel_count; ++c)
                data[channel_count * i + c] = src[cid[c]][src_i];
        }
    }
    else
    {
        float** images = (float**)exr_image.images;
        for(int i = 0; i < exr_image.width * exr_image.height; i++)
        for(int c = 0; c < channel_count; ++c)
            data[channel_count * i + c] = images[cid[c]][i];
    }

    width = exr_image.width;
    height = exr_image.height;

    FreeEXRHeader(&exr_header);
    FreeEXRImage(&exr_image);

    return data;
}

}

namespace tr
{

texture::texture(device_mask dev, const std::string& path)
: opaque(false), buffers(dev)
{
    load_from_file(path);
}

texture::texture(
    device_mask dev,
    uvec2 size,
    unsigned array_layers,
    vk::Format fmt,
    size_t data_size,
    void* data,
    vk::ImageTiling tiling,
    vk::ImageUsageFlags usage,
    vk::ImageLayout layout,
    vk::SampleCountFlagBits msaa
):  dim(size, 1), array_layers(array_layers),
    fmt(fmt), type(vk::ImageType::e2D), tiling(tiling), usage(usage),
    layout(layout), msaa(msaa), opaque(false), buffers(dev)
{
    create(data_size, data);
}

texture::texture(
    device_mask dev,
    uvec3 dim,
    vk::Format fmt,
    vk::ImageTiling tiling,
    vk::ImageUsageFlags usage,
    vk::ImageLayout layout
):  dim(dim), array_layers(1), fmt(fmt), type(vk::ImageType::e3D),
    tiling(tiling), usage(usage), layout(layout),
    msaa(vk::SampleCountFlagBits::e1), opaque(false), buffers(dev)
{
    create(0, nullptr);
}

texture::texture(texture&& other)
:   dim(other.dim), array_layers(other.array_layers), fmt(other.fmt),
    type(other.type), tiling(other.tiling), usage(other.usage),
    layout(other.layout), msaa(other.msaa),
    pixel_data(std::move(other.pixel_data)), opaque(other.opaque),
    buffers(std::move(other.buffers))
{
    other.buffers.clear();
}

vk::ImageView texture::get_array_image_view(device_id id) const
{
    return buffers[id].array_view;
}

vk::ImageView texture::get_layer_image_view(
    uint32_t layer_index, device_id id
) const
{
    return buffers[id].layer_views[layer_index];
}

vk::ImageView texture::get_image_view(device_id id) const
{
    return get_layer_image_view(0, id);
}

vk::Image texture::get_image(device_id id) const
{
    return buffers[id].img;
}

vk::Format texture::get_format() const
{
    return fmt;
}

vk::SampleCountFlagBits texture::get_msaa() const
{
    return msaa;
}

vk::ImageLayout texture::get_layout() const
{
    return layout;
}

void texture::set_opaque(bool opaque)
{
    this->opaque = opaque;
}

bool texture::potentially_transparent() const
{
    if(opaque) return false;
    switch(fmt)
    {
    case vk::Format::eR4G4B4A4UnormPack16:
    case vk::Format::eB4G4R4A4UnormPack16:
    case vk::Format::eR5G5B5A1UnormPack16:
    case vk::Format::eB5G5R5A1UnormPack16:
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Snorm:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Snorm:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eA8B8G8R8UnormPack32:
    case vk::Format::eA8B8G8R8SnormPack32:
    case vk::Format::eA8B8G8R8SrgbPack32:
    case vk::Format::eA2R10G10B10UnormPack32:
    case vk::Format::eA2R10G10B10SnormPack32:
    case vk::Format::eA2B10G10R10UnormPack32:
    case vk::Format::eA2B10G10R10SnormPack32:
    case vk::Format::eR16G16B16A16Unorm:
    case vk::Format::eR16G16B16A16Snorm:
    case vk::Format::eR16G16B16A16Sfloat:
    case vk::Format::eR32G32B32A32Sfloat:
    case vk::Format::eR64G64B64A64Sfloat:
        return true;
    default:
        return false;
    }
}

uvec2 texture::get_size() const
{
    return dim;
}

uvec3 texture::get_dimensions() const
{
    return dim;
}

render_target texture::get_array_render_target(device_id id) const
{
    const auto& buf = buffers[id];
    return render_target(dim, 0, array_layers, buf.img, buf.array_view, layout, fmt, msaa);
}

render_target texture::get_layer_render_target(
    uint32_t layer_index,
    device_id id
) const {
    const auto& buf = buffers[id];
    return render_target(dim, layer_index, 1, buf.img, buf.layer_views[layer_index], layout, fmt, msaa);
}

render_target texture::get_multiview_block_render_target(
    uint32_t block_index,
    device_id id
) const {
    const auto& buf = buffers[id];
    uint32_t block_size = buffers.get_device(id).mv_props.maxMultiviewViewCount;
    return render_target(
        dim,
        block_index * block_size,
        min(
            array_layers - block_index * block_size,
            block_size
        ),
        buf.img, buf.multiview_block_views[block_index], layout,
        fmt,
        msaa
    );
}

device_mask texture::get_mask() const
{
    return buffers.get_mask();
}

size_t texture::get_multiview_block_count() const
{
    return buffers[0].multiview_block_views.size();
}

void texture::resize(uvec2 size)
{
    pixel_data.clear();
    dim = uvec3(size, 1u);
    create(0, nullptr);
}

void texture::load_from_file(const std::string& path)
{
    array_layers = 1;
    fs::path fp(path);
    if(fp.extension().string() == ".exr")
    {
        int n = 0, w = 0, h = 0;
        float* data = read_exr(w, h, n, path);
        if(!data)
            throw std::runtime_error("Failed to load texture " + path);

        VkDeviceSize size = w*h*n*sizeof(float);
        pixel_data.resize(size);
        memcpy(pixel_data.data(), data, size);
        free(data);

        this->dim = uvec3(w, h, 1);
        opaque = n < 4;

        if(n == 3)
        {
            float alpha = 1.0f;
            insert_strided(pixel_data, n*sizeof(float), sizeof(float), &alpha);
            n = 4;
        }

        switch(n)
        {
        default:
        case 1:
            fmt = vk::Format::eR32Sfloat;
            break;
        case 2:
            fmt = vk::Format::eR32G32Sfloat;
            break;
        case 4:
            fmt = vk::Format::eR32G32B32A32Sfloat;
            break;
        }

        type = vk::ImageType::e2D;
        tiling = vk::ImageTiling::eOptimal;
        msaa = vk::SampleCountFlagBits::e1;
        usage = vk::ImageUsageFlagBits::eSampled;
        layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }
    else
    {
        stbi_set_flip_vertically_on_load(false);
        bool hdr = stbi_is_hdr(path.c_str());
        int n = 0, w = 0, h = 0;

        void* data = nullptr;
        VkDeviceSize size = 0;
        if(hdr)
        {
            data = stbi_loadf(path.c_str(), &w, &h, &n, 0);
            size = w*h*n*sizeof(float);
        }
        else
        {
            data = stbi_load(path.c_str(), &w, &h, &n, 0);
            size = w*h*n;
        }
        this->dim = uvec3(w, h, 1);

        if(!data)
            throw std::runtime_error("Failed to load texture " + path);

        pixel_data.resize(size);
        memcpy(pixel_data.data(), data, size);
        stbi_image_free(data);

        // If there's no alpha channel, the texture is opaque.
        opaque = n < 4;

        // Support for 3-channel textures in Vulkan implementations is basically
        // nonexistent, so turn those into 4-channels.
        if(n == 3)
        {
            if(hdr)
            {
                float alpha = 1.0f;
                insert_strided(pixel_data, n*sizeof(float), sizeof(float), &alpha);
            }
            else
            {
                uint8_t alpha = 255;
                insert_strided(pixel_data, n*sizeof(uint8_t), sizeof(uint8_t), &alpha);
            }
            n = 4;
        }

        // Let's use 16-bit floats for hdr images instead of wasting memory with
        // 32-bit data.
        if(hdr) ::float_to_half(pixel_data);

        switch(n)
        {
        default:
        case 1:
            fmt = hdr ? vk::Format::eR16Sfloat : vk::Format::eR8Unorm;
            break;
        case 2:
            fmt = hdr ? vk::Format::eR16G16Sfloat : vk::Format::eR8G8Unorm;
            break;
        case 3:
            fmt = hdr ? vk::Format::eR16G16B16Sfloat : vk::Format::eR8G8B8Unorm;
            break;
        case 4:
            fmt = hdr ? vk::Format::eR16G16B16A16Sfloat : vk::Format::eR8G8B8A8Unorm;
            break;
        }

        type = vk::ImageType::e2D;
        tiling = vk::ImageTiling::eOptimal;
        msaa = vk::SampleCountFlagBits::e1;
        usage = vk::ImageUsageFlagBits::eSampled;
        layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }

    create(pixel_data.size(), pixel_data.data());
}

void texture::create(size_t data_size, void* data)
{
    vk::ImageCreateInfo img_info{
        {},
        type,
        fmt,
        {(uint32_t)dim.x, (uint32_t)dim.y, (uint32_t)dim.z},
        data ? calculate_mipmap_count(uvec2(dim.x, dim.y)) : 1,
        array_layers,
        msaa,
        tiling,
        usage,
        vk::SharingMode::eExclusive
    };

    buffers([&](device& dev, buffer_data& buf){
        create_tex(
            dev, array_layers, buf.img, buf.array_view,
            buf.layer_views, buf.multiview_block_views, img_info,
            layout, data_size, data
        );
    });
}

}
