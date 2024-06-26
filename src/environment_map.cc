#include "environment_map.hh"
#include "compute_pipeline.hh"
#include "placeholders.hh"
#include "descriptor_set.hh"
#include "misc.hh"
#include "sampler.hh"

namespace tr
{

environment_map::environment_map(
    device_mask dev, const std::string& path, projection proj, vec3 factor
): texture(dev, path), factor(factor), proj(proj)
{
    generate_alias_table();
}

void environment_map::set_factor(vec3 factor)
{
    this->factor = factor;
}

vec3 environment_map::get_factor() const
{
    return factor;
}

environment_map::projection environment_map::get_projection() const
{
    return proj;
}

vk::Buffer environment_map::get_alias_table(size_t device_index) const
{
    return alias_table_buffers[device_index];
}

// Based on CC0 code from https://gist.github.com/juliusikkala/6c8c186f0150fe877a55cee4d266b1b0
void environment_map::generate_alias_table()
{
    alias_table.clear();
    device& dev = *get_mask().begin();
    shader_source src("shader/alias_table_importance.comp");
    push_descriptor_set desc(dev);
    desc.add(src);
    compute_pipeline importance_pipeline(dev);
    importance_pipeline.init(src, {&desc});
    sampler envmap_sampler(dev);
    ivec2 size = texture::get_size();
    unsigned pixel_count = size.x * size.y;
    size_t bytes = sizeof(float) * pixel_count;
    vkm<vk::Buffer> readback_buffer = create_download_buffer(dev, bytes);

    vk::CommandBuffer cb = begin_command_buffer(dev);
    importance_pipeline.bind(cb);
    desc.set_texture("environment", *this, dev.ctx->get_placeholders().default_sampler);
    desc.set_buffer(dev.id, "importances", {{*readback_buffer, 0, VK_WHOLE_SIZE}});
    importance_pipeline.push_descriptors(cb, desc, 0);
    cb.dispatch((size.x+15u)/16u, (size.y+15u)/16u, 1);
    end_command_buffer(dev, cb);

    float* importance = nullptr;
    vmaMapMemory(dev.allocator, readback_buffer.get_allocation(), (void**)&importance);

    double sum_importance = 0;
    for(unsigned i = 0; i < pixel_count; ++i)
        sum_importance += importance[i];

    double average = sum_importance / double(pixel_count);
    average_luminance = sum_importance;
    float inv_average = 1.0 / average;
    for(unsigned i = 0; i < pixel_count; ++i)
        importance[i] *= inv_average;

    // Average of 'probability' is now 1.
    // Sweeping alias table build idea from: https://arxiv.org/pdf/1903.00227.pdf
    alias_table.resize(pixel_count);
    for(unsigned i = 0; i < pixel_count; ++i)
        alias_table[i] = {i, 0xFFFFFFFF, 1.0f, 1.0f};

    // i tracks light items, j tracks heavy items.
    unsigned i = 0, j = 0;
    while(i < pixel_count && importance[i] > 1.0f) ++i;
    while(j < pixel_count && importance[j] <= 1.0f) ++j;

    float weight = j < pixel_count ? importance[j] : 0.0f;
    while(j < pixel_count)
    {
        if(weight > 1.0f)
        {
            if(i > pixel_count) break;
            alias_table[i].probability = ldexp(importance[i], 32);
            alias_table[i].alias_id = j;
            weight = (weight + importance[i]) - 1.0f;
            ++i;
            while(i < pixel_count && importance[i]> 1.0f) ++i;
        }
        else
        {
            alias_table[j].probability = ldexp(weight, 32);
            unsigned old_j = j;
            ++j;
            while(j < pixel_count && importance[j] <= 1.0f) ++j;
            if(j < pixel_count)
            {
                alias_table[old_j].alias_id = j;
                weight = (weight + importance[j]) - 1.0f;
            }
        }
    }

    std::vector<float> sin_theta(size.y);
    for(int i = 0; i < size.y; ++i)
        sin_theta[i] = sin((i+0.5f) / float(size.y) * M_PI);
    // Write pdfs.
    for(unsigned i = 0; i < pixel_count; ++i)
    {
        unsigned j = alias_table[i].alias_id;
        alias_table[i].pdf = importance[i] / (2.0f * M_PI * M_PI * sin_theta[i/size.x]);
        alias_table[i].alias_pdf = importance[j] / (2.0f * M_PI * M_PI * sin_theta[j/size.x]);
    }

    vmaUnmapMemory(dev.allocator, readback_buffer.get_allocation());

    alias_table_buffers.init(
        get_mask(),
        [&](device& dev){
            return create_buffer(
                dev,
                vk::BufferCreateInfo{
                    {},
                    sizeof(alias_table_entry) * pixel_count,
                    vk::BufferUsageFlagBits::eStorageBuffer
                },
                VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                alias_table.data()
            );
        }
    );
}

}
