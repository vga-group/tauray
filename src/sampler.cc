#include "sampler.hh"

namespace tr
{

sampler::sampler(
    device_mask dev, vk::Filter min, vk::Filter mag,
    vk::SamplerAddressMode extend_x,
    vk::SamplerAddressMode extend_y,
    vk::SamplerMipmapMode mip,
    int anisotropy, bool normalized, bool use_mipmaps,
    bool shadow, float mip_bias
){
    vk::SamplerCreateInfo info = vk::SamplerCreateInfo{
        {},
        min, mag, mip, extend_x, extend_y, extend_x, mip_bias,
        anisotropy > 0, (float)anisotropy,
        shadow, shadow ? vk::CompareOp::eLess : vk::CompareOp::eAlways,
        0.0f, use_mipmaps ? 1000.0f : 0.0f,
        vk::BorderColor::eFloatTransparentBlack,
        !normalized
    };
    samplers.init(dev, [&](device& d){
        return vkm(d, d.logical.createSampler(info));
    });
}

vk::Sampler sampler::get_sampler(device_id id) const
{
    return samplers[id];
}

}
