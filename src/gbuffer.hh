#ifndef TAURAY_GBUFFER_HH
#define TAURAY_GBUFFER_HH
#include "texture.hh"
#include "render_target.hh"
#include <memory>

// If you want to add a new G-Buffer entry, add it in the list here. This macro
// is used to automatically generate code for the entries everywhere. Pretty
// literally everywhere, you should only need to modify renderers to populate
// the entries and shaders to use these afterwards.
// NOTE: If you can, don't use TR_GBUFFER_ENTRIES to iterate over the entries
// yourself. It is clearer to use the member functions of gbuffer_texture &
// gbuffer_target for this.
//
// Spatial things are in world-space unless otherwise mentioned. This makes it
// more straightforward to handle temporal and multi-viewport algorithms, as all
// G-Buffers are in the same reference space.
#define TR_GBUFFER_ENTRIES \
    /* RGB: total color in linear color space. */\
    TR_GBUFFER_ENTRY(color, vk::Format::eR16G16B16A16Sfloat)\
    /* RGB: direct light in linear color space. */\
    TR_GBUFFER_ENTRY(direct, vk::Format::eR16G16B16A16Sfloat)\
    /* RGB:diffusedirect light in linear color space. */\
    TR_GBUFFER_ENTRY(diffuse, vk::Format::eR16G16B16A16Sfloat)\
    /* RGB: Material albedo in linear color space. */\
    TR_GBUFFER_ENTRY(albedo, vk::Format::eR16G16B16A16Sfloat)\
    /* R: Metallicness, G: Roughness */\
    TR_GBUFFER_ENTRY(material, vk::Format::eR16G16Unorm)\
    /* RG: Packed world-space normal (octahedral mapping) */\
    TR_GBUFFER_ENTRY(normal, vk::Format::eR16G16Snorm)\
    /* R: X-coordinate, G: Y-coordinate, B: Z-coordinate (in world-space)*/\
    TR_GBUFFER_ENTRY(pos, vk::Format::eR32G32B32A32Sfloat)\
    /* RG: Position of the same point in the previous frame. On-screen between [0,1] */\
    TR_GBUFFER_ENTRY(screen_motion, vk::Format::eR32G32Sfloat)\
    /* R: ID of the instance covering each pixel. */\
    TR_GBUFFER_ENTRY(instance_id, vk::Format::eR32Sint)\
    /* R: View-space Z-coordinate (hyperbolic, depending on projection) */\
    TR_GBUFFER_ENTRY(depth, vk::Format::eD32Sfloat)\

namespace tr
{
    // This specification can be used to create a gbuffer_texture in one call.
    // It's useful when multiple parties need to take part into defining which
    // gbuffer entries are wanted.
    struct gbuffer_spec
    {
#define TR_GBUFFER_ENTRY(name, format) \
        bool name##_present = false; \
        vk::Format name##_format = format; \
        vk::ImageUsageFlags name##_usage = vk::ImageUsageFlagBits::eStorage;

        TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
        void set_all_usage(vk::ImageUsageFlags usage);
        size_t present_count() const;
    };

    // Only the render targets that are valid are used. This is why there are
    // a lot of very specific targets here, just don't set them to anything if
    // you don't use them. Also, not every renderer knows how to fill every
    // entry here. They just use 'color' by default.
    struct gbuffer_target
    {
#define TR_GBUFFER_ENTRY(name, ...) render_target name;
        TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY

        size_t entry_count() const;
        // NOTE: entry_count()-1 is not the maximum index, it's always
        // MAX_GBUFFER_ENTRIES-1. These functions can return a false
        // render_target if it's not present for the given index.
        render_target& operator[](size_t i);
        const render_target& operator[](size_t i) const;

        void set_raster_layouts();
        void set_layout(vk::ImageLayout layout);
        uvec2 get_size() const;
        unsigned get_layer_count() const;
        vk::SampleCountFlagBits get_msaa() const;
        void get_location_defines(
            std::map<std::string, std::string>& defines,
            int start_index = 0
        ) const;

        gbuffer_spec get_spec() const;

        template<typename F>
        void visit(F&& f) const
        {
#define TR_GBUFFER_ENTRY(name, ...) if(name) f(name);
        TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
        }

        template<typename F>
        void visit(F&& f)
        {
#define TR_GBUFFER_ENTRY(name, ...) if(name) f(name);
        TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
        }
    };

    class gbuffer_texture
    {
    public:
        gbuffer_texture();
        gbuffer_texture(
            device_data& dev,
            uvec2 size,
            unsigned layer_count,
            vk::SampleCountFlagBits msaa = vk::SampleCountFlagBits::e1
        );
        gbuffer_texture(gbuffer_texture&& other) = default;

        gbuffer_texture& operator=(gbuffer_texture&& other) = default;

        void reset(
            device_data& dev,
            uvec2 size,
            unsigned layer_count,
            vk::SampleCountFlagBits msaa = vk::SampleCountFlagBits::e1
        );

#define TR_GBUFFER_ENTRY(name, format) \
        void add_##name(\
            vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eStorage,\
            vk::Format fmt = format\
        );\
        bool has_##name() const;
        TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY

        void add(gbuffer_spec spec);

        gbuffer_target get_array_target() const;
        gbuffer_target get_layer_target(uint32_t layer_index) const;
        gbuffer_target get_multiview_block_target(uint32_t block_index) const;
        size_t entry_count() const;
        size_t get_layer_count() const;
        size_t get_multiview_block_count() const;

    private:
        device_data* dev;
        uvec2 size;
        unsigned layer_count;
        vk::SampleCountFlagBits msaa;

#define TR_GBUFFER_ENTRY(name, ...) std::unique_ptr<texture> name;
        TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
    };

    constexpr size_t calc_max_gbuffer_entries()
    {
        size_t i = 0;
#define TR_GBUFFER_ENTRY(name, ...) i++;
        TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
        return i;
    }
    constexpr size_t MAX_GBUFFER_ENTRIES = calc_max_gbuffer_entries();
}
#endif
