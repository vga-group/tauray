#include "sampler.hh"

namespace tr
{

sampler::sampler(
    context& ctx, vk::Filter min, vk::Filter mag,
    vk::SamplerAddressMode extend, vk::SamplerMipmapMode mip,
    int anisotropy, bool normalized, bool use_mipmaps,
    bool shadow, float mip_bias
): ctx(&ctx)
{
    std::vector<device_data>& devices = ctx.get_devices();
    samplers.resize(devices.size());
    info = vk::SamplerCreateInfo{
        {},
        min, mag, mip, extend, extend, extend, mip_bias,
        anisotropy > 0, (float)anisotropy,
        shadow, shadow ? vk::CompareOp::eLess : vk::CompareOp::eAlways,
        0.0f, use_mipmaps ? 1000.0f : 0.0f,
        vk::BorderColor::eFloatTransparentBlack,
        !normalized
    };
    for(size_t i = 0; i < devices.size(); ++i)
        samplers[i] = vkm(devices[i], devices[i].dev.createSampler(info));
}

vk::Sampler sampler::get_sampler(size_t device_index) const
{
    return samplers[device_index];
}

}
