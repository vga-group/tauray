#include "basic_pipeline.hh"
#include "descriptor_set.hh"
#include "misc.hh"

namespace tr
{

basic_pipeline::basic_pipeline(
    device& dev,
    std::vector<std::vector<vk::DescriptorSetLayoutBinding>>&& sets,
    std::map<std::string, std::pair<uint32_t, uint32_t>>&& binding_names,
    std::vector<vk::PushConstantRange>&& push_constant_ranges,
    uint32_t max_descriptor_sets,
    vk::PipelineBindPoint bind_point,
    bool use_push_descriptors,
    const std::vector<tr::descriptor_set_layout*>& layout
):  dev(&dev),
    bind_point(bind_point),
    sets(std::move(sets)),
    binding_names(std::move(binding_names)),
    push_constant_ranges(std::move(push_constant_ranges))
{
    vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_info(
        {}, this->sets[0].size(), this->sets[0].data()
    );

    if(use_push_descriptors)
        descriptor_set_layout_info.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;

    descriptor_set_layout = vkm(
        dev, dev.logical.createDescriptorSetLayout(descriptor_set_layout_info)
    );

    if(!use_push_descriptors && max_descriptor_sets > 0)
    {
        descriptor_sets.resize(max_descriptor_sets);
        reset_descriptor_sets();
    }

    std::vector<vk::DescriptorSetLayout> descriptor_sets;

    if(max_descriptor_sets > 0)
        descriptor_sets.insert(descriptor_sets.begin(), descriptor_set_layout);
    for(tr::descriptor_set_layout* layout: layout)
        descriptor_sets.push_back(layout->get_layout(dev.id));

    vk::PipelineLayoutCreateInfo pipeline_layout_info(
        {}, descriptor_sets.size(), descriptor_sets.data(),
        this->push_constant_ranges.size(),
        this->push_constant_ranges.data()
    );

    pipeline_layout = vkm(dev, dev.logical.createPipelineLayout(pipeline_layout_info));
}

void basic_pipeline::reset_descriptor_sets()
{
    if(descriptor_sets.size() == 0)
        return;

    std::map<vk::DescriptorType, uint32_t> type_count;
    for(auto& b: sets[0]) type_count[b.descriptorType] += b.descriptorCount;

    std::vector<vk::DescriptorPoolSize> pool_sizes;
    for(auto& pair: type_count)
    {
        pool_sizes.push_back({
            pair.first, (uint32_t)(pair.second * descriptor_sets.size())
        });
    }

    descriptor_pool = vkm(*dev,
        dev->logical.createDescriptorPool({
            {}, (uint32_t)descriptor_sets.size(),
            (uint32_t)pool_sizes.size(), pool_sizes.data()
        })
    );

    for(size_t i = 0; i < descriptor_sets.size(); ++i)
    {
        vk::DescriptorSetAllocateInfo dset_alloc_info(
            descriptor_pool, 1, descriptor_set_layout
        );
        vk::Result res = dev->logical.allocateDescriptorSets(
            &dset_alloc_info, &descriptor_sets[i]
        );
        if(res != vk::Result::eSuccess)
            throw std::runtime_error(vk::to_string(res));
    }
}

void basic_pipeline::update_descriptor_set(
    const std::vector<descriptor_state>& descriptor_states,
    int32_t index
){
    if(descriptor_sets.size() == 0) return;
    if(index < 0)
    {
        for(index = 0; index < (int32_t)descriptor_sets.size(); ++index)
            update_descriptor_set(descriptor_states, index);
    }
    else
    {
        // These just hold the temporary arrays needed by some placeholder
        // situations.
        std::vector<std::vector<vk::DescriptorBufferInfo>> buffer_holder;
        std::vector<std::vector<vk::DescriptorImageInfo>> image_holder;

        placeholders& pl = dev->ctx->get_placeholders();

        std::vector<vk::WriteDescriptorSet> writes;
        writes.reserve(descriptor_states.size());
        for(const auto& dstate: descriptor_states)
        {
            const vk::DescriptorSetLayoutBinding* binding =
                find_descriptor_binding(dstate.get_binding_name());
            if(binding && !dstate.is_empty())
                writes.push_back(
                    dstate.get_write(
                        pl, dev->id, descriptor_sets[index],
                        *binding, buffer_holder, image_holder
                    )
                );
        }
        dev->logical.updateDescriptorSets(writes, nullptr);
    }
}

void basic_pipeline::push_descriptors(
    vk::CommandBuffer cb,
    const std::vector<descriptor_state>& descriptor_states
){
    // These just hold the temporary arrays needed by some placeholder
    // situations.
    std::vector<std::vector<vk::DescriptorBufferInfo>> buffer_holder;
    std::vector<std::vector<vk::DescriptorImageInfo>> image_holder;

    placeholders& pl = dev->ctx->get_placeholders();

    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(descriptor_states.size());
    for(const auto& dstate: descriptor_states)
    {
        const vk::DescriptorSetLayoutBinding* binding =
            find_descriptor_binding(dstate.get_binding_name());
        if(binding && !dstate.is_empty())
            writes.push_back(
                dstate.get_write(
                    pl, dev->id, VK_NULL_HANDLE,
                    *binding, buffer_holder, image_holder
                )
            );
    }

    cb.pushDescriptorSetKHR(
        bind_point,
        pipeline_layout,
        0,
        writes
    );
}

const vk::DescriptorSetLayoutBinding* basic_pipeline::find_descriptor_binding(
    const std::string& binding_name
) const
{
    auto it = binding_names.find(binding_name);
    if(it == binding_names.end())
    {
        //printf("Could not find descriptor binding %s\n", binding_name.c_str());
        return nullptr;
    }

    for(const auto& b: sets[0])
    {
        if(b.binding == it->second.second)
            return &b;
    }
    return nullptr;
}

device* basic_pipeline::get_device() const
{
    return dev;
}

void basic_pipeline::bind(vk::CommandBuffer cmd, uint32_t descriptor_set_index) const
{
    bind(cmd);
    if(descriptor_sets.size() > 0)
    {
        cmd.bindDescriptorSets(
            bind_point, pipeline_layout,
            0, {descriptor_sets[descriptor_set_index]}, {}
        );
    }
}

void basic_pipeline::set_descriptors(vk::CommandBuffer cmd, descriptor_set& set, uint32_t index, uint32_t set_index) const
{
    set.bind(dev->id, cmd, pipeline_layout, bind_point, index, set_index);
}

void basic_pipeline::push_descriptors(vk::CommandBuffer cmd, push_descriptor_set& set,  uint32_t set_index) const
{
    set.push(dev->id, cmd, pipeline_layout, bind_point, set_index);
}

void basic_pipeline::bind(vk::CommandBuffer cmd) const
{
    cmd.bindPipeline(bind_point, *pipeline);
}

void basic_pipeline::load_shader_module(
    shader_source src,
    vk::ShaderStageFlagBits stage,
    std::vector<vk::PipelineShaderStageCreateInfo>& stages,
    const vk::SpecializationInfo& specialization
){
    if(src.data.empty()) return;

    vkm<vk::ShaderModule> mod = vkm(*dev, dev->logical.createShaderModule({
        {}, src.data.size()*sizeof(uint32_t), src.data.data()
    }));
    stages.push_back({{}, stage, mod, "main", specialization.pData != nullptr ? &specialization : nullptr});
}


}
