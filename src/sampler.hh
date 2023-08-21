#ifndef TAURAY_SAMPLER_HH
#define TAURAY_SAMPLER_HH
#include "context.hh"

namespace tr
{

class sampler
{
public:
    sampler(
        device_mask dev,
        vk::Filter min = vk::Filter::eLinear,
        vk::Filter mag = vk::Filter::eLinear,
        vk::SamplerAddressMode extend_x = vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode extend_y = vk::SamplerAddressMode::eRepeat,
        vk::SamplerMipmapMode mip = vk::SamplerMipmapMode::eLinear,
        int anisotropy = 16,
        bool normalized = true,
        bool use_mipmaps = true,
        bool shadow = false,
        float mip_bias = 0.0f
    );
    sampler(const sampler& other) = delete;
    sampler(sampler&& other) = default;

    vk::Sampler get_sampler(device_id id) const;

private:
    per_device<vkm<vk::Sampler>> samplers;
};

}

#endif
