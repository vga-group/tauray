#ifndef TAURAY_SAMPLER_HH
#define TAURAY_SAMPLER_HH
#include "context.hh"

namespace tr
{

class sampler
{
public:
    sampler(
        context& ctx,
        vk::Filter min = vk::Filter::eLinear,
        vk::Filter mag = vk::Filter::eLinear,
        vk::SamplerAddressMode extend = vk::SamplerAddressMode::eRepeat,
        vk::SamplerMipmapMode mip = vk::SamplerMipmapMode::eLinear,
        int anisotropy = 16,
        bool normalized = true,
        bool use_mipmaps = true,
        bool shadow = false,
        float mip_bias = 0.0f
    );
    sampler(const context& other) = delete;
    sampler(sampler&& other) = default;

    vk::Sampler get_sampler(size_t device_index) const;

private:
    context* ctx;
    vk::SamplerCreateInfo info;
    std::vector<vkm<vk::Sampler>> samplers;
};

}

#endif
