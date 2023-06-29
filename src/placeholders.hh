#ifndef TAURAY_PLACEHOLDERS_HH
#define TAURAY_PLACEHOLDERS_HH
#include "texture.hh"
#include "sampler.hh"

namespace tr
{

// These placeholder assets are used when some resource is missing.
struct placeholders
{
    placeholders(context& ctx);

    context* ctx;
    texture sample2d;
    texture sample3d;
    texture depth_test_sample;
    sampler default_sampler;

    struct per_device_data
    {
        vkm<vk::Buffer> storage_buffer;

        vk::DescriptorImageInfo img_2d_info;
        vk::DescriptorImageInfo img_3d_info;
        vk::DescriptorBufferInfo storage_info;
    };
    std::vector<per_device_data> per_device;
};

}

#endif

