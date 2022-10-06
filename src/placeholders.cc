#include "placeholders.hh"
#include "misc.hh"

namespace tr
{

placeholders::placeholders(context& ctx)
:   ctx(&ctx),
    sample2d(
        ctx,
        uvec2(1),
        1,
        vk::Format::eR8G8B8A8Unorm,
        0, nullptr,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled,
        vk::ImageLayout::eShaderReadOnlyOptimal
    ),
    sample3d(
        ctx,
        uvec3(1),
        vk::Format::eR8G8B8A8Unorm,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled,
        vk::ImageLayout::eShaderReadOnlyOptimal
    ),
    depth_test_sample(
        ctx,
        uvec2(1),
        1,
        vk::Format::eD32Sfloat,
        0, nullptr,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled,
        vk::ImageLayout::eShaderReadOnlyOptimal
    ),
    default_sampler(
        ctx, vk::Filter::eNearest, vk::Filter::eNearest,
        vk::SamplerAddressMode::eRepeat, vk::SamplerMipmapMode::eNearest, 0,
        true, false
    )
{
    std::vector<device_data>& devices = ctx.get_devices();
    per_device.resize(devices.size());
    for(size_t i = 0; i < per_device.size(); ++i)
    {
        per_device[i].storage_buffer = create_buffer(
            devices[i],
            {
                {},
                4,
                vk::BufferUsageFlagBits::eStorageBuffer,
                vk::SharingMode::eExclusive
            },
            VMA_MEMORY_USAGE_CPU_ONLY
        );

        per_device[i].img_2d_info = {
            default_sampler.get_sampler(i),
            sample2d.get_image_view(i),
            vk::ImageLayout::eShaderReadOnlyOptimal
        };
        per_device[i].img_3d_info = {
            default_sampler.get_sampler(i),
            sample3d.get_image_view(i),
            vk::ImageLayout::eShaderReadOnlyOptimal
        };
        per_device[i].storage_info = vk::DescriptorBufferInfo{
            per_device[i].storage_buffer, 0, VK_WHOLE_SIZE
        };
    }
}

}
