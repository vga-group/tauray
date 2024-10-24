#include "sampler_table.hh"
#include "placeholders.hh"
#include "scene_stage.hh"

namespace tr
{

sampler_table::sampler_table(device_mask dev, bool use_mipmaps)
:   default_sampler(
        dev, vk::Filter::eLinear, vk::Filter::eLinear,
        vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode::eRepeat,
        vk::SamplerMipmapMode::eLinear,
        use_mipmaps ? 16 : 0, true, use_mipmaps, false, 0.0f
    )
{
}

void sampler_table::update_scene(scene_stage* s)
{
    const std::vector<scene_stage::instance>& instances = s->get_instances();
    table.clear();
    index_counter = 0;
    for(size_t i = 0; i < instances.size(); ++i)
    {
        const material& mat = *instances[i].mat;
        register_tex_id(mat.albedo_tex);
        register_tex_id(mat.metallic_roughness_tex);
        register_tex_id(mat.normal_tex);
        register_tex_id(mat.emission_tex);
    }
}

std::vector<vk::DescriptorImageInfo> sampler_table::get_image_infos(device_id id) const
{
    std::vector<vk::DescriptorImageInfo> dii(index_counter);
    for(const auto& pair: table)
    {
        dii[pair.second] = {
            pair.first.second->get_sampler(id),
            pair.first.first->get_image_view(id),
            vk::ImageLayout::eShaderReadOnlyOptimal
        };
    }
    return dii;
}

void sampler_table::register_tex_id(combined_tex_sampler cs)
{
    if(cs.first)
    {
        const sampler* s = cs.second ?  cs.second : &default_sampler;
        combined_tex_sampler key(cs.first, s);

        auto it = table.find(key);
        if(it == table.end())
        {
            table[key] = index_counter++;
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
