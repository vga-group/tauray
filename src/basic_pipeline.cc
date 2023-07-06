#include "basic_pipeline.hh"
#include "misc.hh"

namespace tr
{

basic_pipeline::basic_pipeline(
    device& dev,
    std::vector<vk::DescriptorSetLayoutBinding>&& bindings,
    std::map<std::string, uint32_t>&& binding_names,
    std::vector<vk::PushConstantRange>&& push_constant_ranges,
    uint32_t max_descriptor_sets,
    vk::PipelineBindPoint bind_point,
    bool use_push_descriptors
):  dev(&dev),
    bind_point(bind_point),
    bindings(std::move(bindings)),
    binding_names(std::move(binding_names)),
    push_constant_ranges(std::move(push_constant_ranges))
{
    vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_info(
        {}, this->bindings.size(), this->bindings.data()
    );

    if(use_push_descriptors)
        descriptor_set_layout_info.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;

    descriptor_set_layout = vkm(
        dev, dev.dev.createDescriptorSetLayout(descriptor_set_layout_info)
    );

    if(!use_push_descriptors)
    {
        descriptor_sets.resize(max_descriptor_sets);
        reset_descriptor_sets();
    }

    vk::PipelineLayoutCreateInfo pipeline_layout_info(
        {}, 1, descriptor_set_layout,
        this->push_constant_ranges.size(),
        this->push_constant_ranges.data()
    );

    pipeline_layout = vkm(dev, dev.dev.createPipelineLayout(pipeline_layout_info));
}

void basic_pipeline::reset_descriptor_sets()
{
    if(descriptor_sets.size() == 0)
        return;

    std::map<vk::DescriptorType, uint32_t> type_count;
    for(auto& b: bindings) type_count[b.descriptorType] += b.descriptorCount;

    std::vector<vk::DescriptorPoolSize> pool_sizes;
    for(auto& pair: type_count)
    {
        pool_sizes.push_back({
            pair.first, (uint32_t)(pair.second * descriptor_sets.size())
        });
    }

    descriptor_pool = vkm(*dev,
        dev->dev.createDescriptorPool({
            {}, (uint32_t)descriptor_sets.size(),
            (uint32_t)pool_sizes.size(), pool_sizes.data()
        })
    );

    for(size_t i = 0; i < descriptor_sets.size(); ++i)
    {
        vk::DescriptorSetAllocateInfo dset_alloc_info(
            descriptor_pool, 1, descriptor_set_layout
        );
        vk::Result res = dev->dev.allocateDescriptorSets(
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
                        pl, dev->index, descriptor_sets[index],
                        *binding, buffer_holder, image_holder
                    )
                );
        }
        dev->dev.updateDescriptorSets(writes, nullptr);
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
                    pl, dev->index, VK_NULL_HANDLE,
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

    for(const auto& b: bindings)
    {
        if(b.binding == it->second)
            return &b;
    }
    return nullptr;
}

device_data* basic_pipeline::get_device() const
{
    return dev;
}

void basic_pipeline::bind(vk::CommandBuffer cmd, uint32_t descriptor_set_index) const
{
    bind(cmd);
    cmd.bindDescriptorSets(
        bind_point, pipeline_layout,
        0, {descriptor_sets[descriptor_set_index]}, {}
    );
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

    vkm<vk::ShaderModule> mod = vkm(*dev, dev->dev.createShaderModule({
        {}, src.data.size()*sizeof(uint32_t), src.data.data()
    }));
    stages.push_back({{}, stage, mod, "main", specialization.pData != nullptr ? &specialization : nullptr});
}


}
