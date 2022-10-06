#include "sampler_table.hh"
#include "placeholders.hh"
#include "scene.hh"

namespace tr
{

sampler_table::sampler_table(device_data& dev, bool use_mipmaps)
:   dev(&dev), default_sampler(
        *dev.ctx, vk::Filter::eLinear, vk::Filter::eLinear,
        vk::SamplerAddressMode::eRepeat, vk::SamplerMipmapMode::eLinear,
        use_mipmaps ? 16 : 0, true, use_mipmaps, false, -1.0f
    )
{
}

std::vector<vk::DescriptorImageInfo> sampler_table::update_scene(scene* s)
{
    const std::vector<scene::instance>& instances = s->get_instances();
    table.clear();
    std::vector<vk::DescriptorImageInfo> dii;
    for(size_t i = 0; i < instances.size(); ++i)
    {
        const material& mat = *instances[i].mat;
        register_tex_id(mat.albedo_tex, dii);
        register_tex_id(mat.metallic_roughness_tex, dii);
        register_tex_id(mat.normal_tex, dii);
        register_tex_id(mat.emission_tex, dii);
    }
    return dii;
}

void sampler_table::register_tex_id(
    combined_tex_sampler cs,
    std::vector<vk::DescriptorImageInfo>& dii
){
    if(cs.first)
    {
        const sampler* s = cs.second ?  cs.second : &default_sampler;
        combined_tex_sampler key(cs.first, s);

        auto it = table.find(key);
        if(it == table.end())
        {
            table[key] = dii.size();
            dii.push_back({
                s->get_sampler(dev->index),
                cs.first->get_image_view(dev->index),
                vk::ImageLayout::eShaderReadOnlyOptimal
            });
        }
    }
}

int sampler_table::find_tex_id(combined_tex_sampler cs)
{
    if(cs.first)
    {
        const sampler* s = cs.second ? cs.second : &default_sampler;
        combined_tex_sampler key(cs.first, s);
        auto it = table.find(key);
        if(it == table.end())
            throw std::runtime_error("Sampler table is out of date!");
        return it->second;
    }
    else return -1;
}

}
