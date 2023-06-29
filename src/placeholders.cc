#include "placeholders.hh"
#include "misc.hh"

namespace tr
{

placeholders::placeholders(context& ctx)
:   ctx(&ctx),
    sample2d(
        device_mask::all(ctx),
        uvec2(1),
        1,
        vk::Format::eR8G8B8A8Unorm,
        0, nullptr,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled,
        vk::ImageLayout::eShaderReadOnlyOptimal
    ),
    sample3d(
        device_mask::all(ctx),
        uvec3(1),
        vk::Format::eR8G8B8A8Unorm,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled,
        vk::ImageLayout::eShaderReadOnlyOptimal
    ),
    depth_test_sample(
        device_mask::all(ctx),
        uvec2(1),
        1,
        vk::Format::eD32Sfloat,
        0, nullptr,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled,
        vk::ImageLayout::eShaderReadOnlyOptimal
    ),
    default_sampler(
        device_mask::all(ctx),
        vk::Filter::eNearest, vk::Filter::eNearest,
        vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode::eRepeat,
        vk::SamplerMipmapMode::eNearest, 0,
        true, false
    ),
    buffers(device_mask::all(ctx))
{
    buffers([&](device& d, buffer& buf){
        buf.storage_buffer = create_buffer(
            d,
            {
                {},
                4,
                vk::BufferUsageFlagBits::eStorageBuffer,
                vk::SharingMode::eExclusive
            },
            VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
        );

        buf.img_2d_info = {
            default_sampler.get_sampler(d.index),
            sample2d.get_image_view(d.index),
            vk::ImageLayout::eShaderReadOnlyOptimal
        };
        buf.img_3d_info = {
            default_sampler.get_sampler(d.index),
            sample3d.get_image_view(d.index),
            vk::ImageLayout::eShaderReadOnlyOptimal
        };
        buf.storage_info = vk::DescriptorBufferInfo{
            buf.storage_buffer, 0, VK_WHOLE_SIZE
        };
    });
}

}
