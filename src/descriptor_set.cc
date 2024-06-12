#include "descriptor_set.hh"
#include "shader_source.hh"
#include "misc.hh"
#include "texture.hh"
#include "sampler.hh"
#include "gpu_buffer.hh"
#include <stdexcept>

namespace
{

size_t calculate_descriptor_pool_sizes(
    size_t binding_count,
    vk::DescriptorPoolSize* pool_sizes,
    const vk::DescriptorSetLayoutBinding* bindings,
    uint32_t multiplier
){
    size_t count = 0;
    for(size_t i = 0; i < binding_count; ++i)
    {
        vk::DescriptorSetLayoutBinding b = bindings[i];
        bool found = false;
        for(size_t j = 0; j < count; ++j)
        {
            vk::DescriptorPoolSize& size = pool_sizes[j];
            if(size.type == b.descriptorType)
            {
                found = true;
                size.descriptorCount += b.descriptorCount * multiplier;
            }
        }

        if(!found)
        {
            pool_sizes[count] = {b.descriptorType, b.descriptorCount * multiplier};
            count++;
        }
    }
    return count;
}

}

namespace tr
{

descriptor_set_layout::descriptor_set_layout(device_mask dev, bool push_descriptor_set)
: push_descriptor_set(push_descriptor_set), layout(dev)
{
}

descriptor_set_layout::~descriptor_set_layout() {}

void descriptor_set_layout::add(
    std::string_view name,
    const vk::DescriptorSetLayoutBinding& binding,
    vk::DescriptorBindingFlags flags
){
    auto name_it = descriptor_names.insert(std::string(name)).first;
    auto it = named_bindings.find(*name_it);
    if(it != named_bindings.end() && it->second.binding != binding.binding)
        throw std::runtime_error(
            std::string("Binding ") + std::string(name) + " has conflicting binding indices: " +
            std::to_string(it->second.binding) + " and " + std::to_string(binding.binding) + "."
        );

    if(it == named_bindings.end())
        named_bindings[*name_it] = {binding, flags};
    else
    {
        it->second.stageFlags |= binding.stageFlags;
        it->second.descriptorCount = max(binding.descriptorCount, it->second.descriptorCount);
    }
    for(const auto&[dev, data]: layout)
        data.dirty = true;
}

void descriptor_set_layout::add(const shader_source& in_data, uint32_t target_set_index)
{
    for(auto[name, info]: in_data.bindings)
    {
        if(info.set != target_set_index)
            continue;

        add(name, info.binding);
    }
}

void descriptor_set_layout::add(const raster_shader_sources& data, uint32_t target_set_index)
{
    add(data.vert, target_set_index);
    add(data.frag, target_set_index);
}

void descriptor_set_layout::add(const rt_shader_sources& data, uint32_t target_set_index)
{
    add(data.rgen, target_set_index);
    for(const rt_shader_sources::hit_group& hg: data.rhit)
    {
        add(hg.rchit, target_set_index);
        add(hg.rahit, target_set_index);
        add(hg.rint, target_set_index);
    }
    for(const shader_source& src: data.rmiss)
        add(src, target_set_index);
}

void descriptor_set_layout::set_binding_params(
    std::string_view name,
    uint32_t count,
    vk::DescriptorBindingFlags flags
){
    auto it = named_bindings.find(name);
    if(it != named_bindings.end())
    {
        it->second.descriptorCount = count;
        it->second.flags = flags;
        for(const auto&[dev, data]: layout)
            data.dirty = true;
    }
}

descriptor_set::set_binding
descriptor_set_layout::find_binding(std::string_view name) const
{
    auto it = named_bindings.find(name);
    if(it == named_bindings.end())
        throw std::runtime_error("Missing binding " + std::string(name));

    return it->second;
}

vk::DescriptorSetLayout descriptor_set_layout::get_layout(device_id id) const
{
    refresh(id);
    return layout[id].layout;
}

void descriptor_set_layout::refresh(device_id id) const
{
    layout_data& ld = layout[id];
    if(ld.dirty)
    {
        bindings.clear();
        std::vector<vk::DescriptorBindingFlags> binding_flags;
        for(const auto& [name, binding]: named_bindings)
        {
            binding_flags.push_back(binding.flags);
            bindings.push_back(binding);
        }

        vk::DescriptorSetLayoutBindingFlagsCreateInfo flag_info = {
            (uint32_t)binding_flags.size(), binding_flags.data()
        };
        vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_info = {
            {}, (uint32_t)bindings.size(), bindings.data()
        };
        descriptor_set_layout_info.pNext = &flag_info;

        if(push_descriptor_set)
            descriptor_set_layout_info.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;

        device& dev = layout.get_device(id);
        ld.layout = vkm(
            dev, dev.logical.createDescriptorSetLayout(descriptor_set_layout_info)
        );
        ld.descriptor_pool_capacity = 0;

        ld.dirty = false;
    }
}

device_mask descriptor_set_layout::get_mask() const
{
    return layout.get_mask();
}

descriptor_set::descriptor_set(device_mask dev)
: descriptor_set_layout(dev, false), data(dev)
{
}

descriptor_set::~descriptor_set()
{
    reset(data.get_mask(), 0);
}

void descriptor_set::reset(device_mask mask, uint32_t count)
{
    for(const device& dev: mask)
        reset(dev.id, count);
}

void descriptor_set::reset(device_id id, uint32_t count)
{
    refresh(id);

    if(named_bindings.size() == 0)
        return;

    set_data& sd = data[id];
    layout_data& ld = layout[id];
    device& dev = data.get_device(id);
    for(vk::DescriptorSet set: sd.alternatives)
    {
        dev.ctx->queue_frame_finish_callback([
            descriptor_set=set,
            pool=*sd.pool,
            logical_device=dev.logical
        ](){
            logical_device.freeDescriptorSets(pool, 1, &descriptor_set);
        });
    }
    sd.alternatives.clear();

    if(ld.descriptor_pool_capacity < count)
    {
        uint32_t safe_count = count * (MAX_FRAMES_IN_FLIGHT + 2);
        std::vector<vk::DescriptorPoolSize> pool_sizes(bindings.size());

        size_t pool_count = calculate_descriptor_pool_sizes(
            bindings.size(), pool_sizes.data(), bindings.data(), safe_count
        );
        vk::DescriptorPoolCreateInfo pool_create_info = {
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            (uint32_t)safe_count,
            (uint32_t)pool_count,
            pool_sizes.data()
        };
        vk::DescriptorPool tmp_pool = dev.logical.createDescriptorPool(pool_create_info);
        sd.pool = vkm(dev, tmp_pool);
        ld.descriptor_pool_capacity = count;
    }

    // Create descriptor sets
    if(count > 0)
    {
        std::vector<vk::DescriptorSetLayout> layouts(count);
        std::fill(layouts.begin(), layouts.end(), layout[id].layout);
        vk::DescriptorSetAllocateInfo descriptor_alloc_info = {
            sd.pool,
            (uint32_t)count,
            layouts.data()
        };

        sd.alternatives.resize(count);
        if(dev.logical.allocateDescriptorSets(&descriptor_alloc_info, sd.alternatives.data()) != vk::Result::eSuccess)
            throw std::runtime_error("Failed to allocate descriptor sets for some reason");
    }
}

void descriptor_set::set_image(
    device_id id,
    uint32_t index,
    std::string_view name,
    std::vector<vk::DescriptorImageInfo>&& image_infos
){
    if(named_bindings.count(name) == 0 || image_infos.size() == 0) return;

    set_binding bind = find_binding(name);

    set_data& sd = data[id];
    if(!(bind.flags&vk::DescriptorBindingFlagBits::ePartiallyBound) && image_infos.size() != bind.descriptorCount)
        throw std::runtime_error(
            "Image view count does not match descriptor count, and "
            "VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT isn't set."
        );

    if(image_infos.size() > bind.descriptorCount)
        throw std::runtime_error("More images than descriptor allows!");

    if(
        bind.descriptorType != vk::DescriptorType::eSampledImage &&
        bind.descriptorType != vk::DescriptorType::eCombinedImageSampler &&
        bind.descriptorType != vk::DescriptorType::eStorageImage
    ) throw std::runtime_error("Cannot set non-image descriptor as an image!");

    for(size_t i = 0; i < image_infos.size(); ++i)
    {
        if(image_infos[i].imageLayout == vk::ImageLayout::eUndefined)
            image_infos[i].imageLayout = bind.descriptorType == vk::DescriptorType::eStorageImage ?
                vk::ImageLayout::eGeneral :
                vk::ImageLayout::eShaderReadOnlyOptimal;
    }
    vk::WriteDescriptorSet write = {
        sd.alternatives[index], (uint32_t)bind.binding, 0,
        (uint32_t)image_infos.size(), bind.descriptorType,
        image_infos.data(), nullptr, nullptr
    };
    data.get_device(id).logical.updateDescriptorSets(
        1, &write, 0, nullptr
    );
}

void descriptor_set::set_texture(
    uint32_t index,
    std::string_view name,
    const texture& tex,
    const sampler& s
){
    for(device& dev: layout.get_mask())
    {
        vk::DescriptorImageInfo info;
        info.sampler = s.get_sampler(dev.id);
        info.imageView = tex.get_image_view(dev.id);
        info.imageLayout = vk::ImageLayout::eUndefined;
        if(data.get_mask().contains(dev.id))
            set_image(dev.id, index, name, {{info}});
    }
}

void descriptor_set::set_image(
    uint32_t index,
    std::string_view name,
    const texture& tex
){
    for(device& dev: layout.get_mask())
    {
        vk::DescriptorImageInfo info;
        info.sampler = VK_NULL_HANDLE;
        info.imageView = tex.get_image_view(dev.id);
        info.imageLayout = vk::ImageLayout::eUndefined;
        if(data.get_mask().contains(dev.id))
            set_image(dev.id, index, name, {{info}});
    }
}

void descriptor_set::set_image_array(
    uint32_t index,
    std::string_view name,
    const texture& tex
){
    for(device& dev: layout.get_mask())
    {
        vk::DescriptorImageInfo info;
        info.sampler = VK_NULL_HANDLE;
        info.imageView = tex.get_array_image_view(dev.id);
        info.imageLayout = vk::ImageLayout::eUndefined;
        if(data.get_mask().contains(dev.id))
            set_image(dev.id, index, name, {{info}});
    }
}

void descriptor_set::set_buffer(
    device_id id,
    uint32_t index,
    std::string_view name,
    std::vector<vk::DescriptorBufferInfo>&& infos
){
    if(named_bindings.count(name) == 0 || infos.size() == 0) return;

    set_binding bind = find_binding(name);

    set_data& sd = data[id];
    if(infos.size() > bind.descriptorCount)
        throw std::runtime_error("More buffers than descriptor allows!");

    if(bind.descriptorType != vk::DescriptorType::eStorageBuffer && bind.descriptorType != vk::DescriptorType::eUniformBuffer)
        throw std::runtime_error("Cannot set non-buffer descriptor as a buffer!");

    for(size_t i = 0;;)
    {
        while(i < infos.size() && !infos[i].buffer) ++i;
        uint32_t begin = i;
        while(i < infos.size() && infos[i].buffer) ++i;
        uint32_t end = i;

        if(end == begin)
            break;

        vk::WriteDescriptorSet write = {
            sd.alternatives[index], (uint32_t)bind.binding, (uint32_t)begin,
            end-begin, bind.descriptorType,
            nullptr, infos.data()+begin, nullptr
        };
        data.get_device(id).logical.updateDescriptorSets(
            1, &write, 0, nullptr
        );
    }
}

void descriptor_set::set_buffer(
    uint32_t index,
    std::string_view name,
    const gpu_buffer& buffer,
    uint32_t offset
){
    for(device& dev: buffer.get_mask())
    {
        if(data.get_mask().contains(dev.id))
            set_buffer(dev.id, index, name, {{buffer[dev.id], offset, VK_WHOLE_SIZE}});
    }
}

void descriptor_set::set_acceleration_structure(
    device_id id,
    uint32_t index,
    std::string_view name,
    vk::AccelerationStructureKHR tlas
){
    if(named_bindings.count(name) == 0) return;

    set_data& sd = data[id];

    set_binding bind = find_binding(name);
    if(bind.descriptorType != vk::DescriptorType::eAccelerationStructureKHR)
        throw std::runtime_error(
            "Cannot set non-acceleration structure descriptor as an acceleration "
            "structure!"
        );

    vk::WriteDescriptorSetAccelerationStructureKHR as_write = {
        1, &tlas
    };

    vk::WriteDescriptorSet write = {
        sd.alternatives[index], (uint32_t)bind.binding, 0,
        (uint32_t)1, bind.descriptorType,
        nullptr, nullptr, nullptr
    };
    write.pNext = &as_write;
    data.get_device(id).logical.updateDescriptorSets(
        1, &write, 0, nullptr
    );
}

void descriptor_set::bind(
    device_id id,
    vk::CommandBuffer buf,
    vk::PipelineLayout pipeline_layout,
    vk::PipelineBindPoint bind_point,
    uint32_t alternative_index,
    uint32_t set_index
) const {
    const set_data& sd = data[id];
    if(sd.alternatives.size() == 0)
        return;
    if(alternative_index >= sd.alternatives.size())
        throw std::runtime_error("Alternative index is higher than number of alternatives");

    buf.bindDescriptorSets(
        bind_point,
        pipeline_layout,
        set_index, 1,
        &sd.alternatives[alternative_index],
        0, nullptr
    );
}

push_descriptor_set::push_descriptor_set(device_mask dev):
    descriptor_set_layout(dev, true), data(dev)
{
}

push_descriptor_set::~push_descriptor_set()
{
}

void push_descriptor_set::set_image(
    device_id id,
    std::string_view name,
    std::vector<vk::DescriptorImageInfo>&& infos
){
    if(named_bindings.count(name) == 0 || infos.size() == 0) return;

    set_data& sd = data[id];

    set_binding bind = find_binding(name);
    auto& image_infos = sd.image_info_index < sd.tmp_image_infos.size() ?
        sd.tmp_image_infos[sd.image_info_index] :
        sd.tmp_image_infos.emplace_back(std::vector<vk::DescriptorImageInfo>());
    image_infos = std::move(infos);
    sd.image_info_index++;

    if(
        !(bind.flags&vk::DescriptorBindingFlagBits::ePartiallyBound) &&
        image_infos.size() != bind.descriptorCount
    ) throw std::runtime_error(
        "Image view count does not match descriptor count, and "
        "VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT isn't set."
    );

    if(image_infos.size() > bind.descriptorCount)
        throw std::runtime_error("More images than descriptor allows!");

    if(
        bind.descriptorType != vk::DescriptorType::eSampledImage &&
        bind.descriptorType != vk::DescriptorType::eCombinedImageSampler &&
        bind.descriptorType != vk::DescriptorType::eStorageImage
    ) throw std::runtime_error("Cannot set non-image descriptor as an image!");

    for(size_t i = 0; i < image_infos.size(); ++i)
    {
        if(image_infos[i].imageLayout == vk::ImageLayout::eUndefined)
            image_infos[i].imageLayout = bind.descriptorType == vk::DescriptorType::eStorageImage ?
                vk::ImageLayout::eGeneral :
                vk::ImageLayout::eShaderReadOnlyOptimal;
    }
    vk::WriteDescriptorSet write = {
        VK_NULL_HANDLE, (uint32_t)bind.binding, 0,
        (uint32_t)image_infos.size(), bind.descriptorType,
        image_infos.data(), nullptr, nullptr
    };
    sd.writes.push_back(write);
}

void push_descriptor_set::set_texture(
    std::string_view name,
    const texture& tex,
    const sampler& s
){
    for(device& dev: layout.get_mask())
    {
        vk::DescriptorImageInfo info;
        info.sampler = s.get_sampler(dev.id);
        info.imageView = tex.get_image_view(dev.id);
        info.imageLayout = vk::ImageLayout::eUndefined;
        if(data.get_mask().contains(dev.id))
            set_image(dev.id, name, {{info}});
    }
}

void push_descriptor_set::set_image(
    std::string_view name,
    const texture& tex
){
    for(device& dev: layout.get_mask())
    {
        vk::DescriptorImageInfo info;
        info.sampler = VK_NULL_HANDLE;
        info.imageView = tex.get_image_view(dev.id);
        info.imageLayout = vk::ImageLayout::eUndefined;
        if(data.get_mask().contains(dev.id))
            set_image(dev.id, name, {{info}});
    }
}

void push_descriptor_set::set_image_array(
    std::string_view name,
    const texture& tex
){
    for(device& dev: layout.get_mask())
    {
        vk::DescriptorImageInfo info;
        info.sampler = VK_NULL_HANDLE;
        info.imageView = tex.get_array_image_view(dev.id);
        info.imageLayout = vk::ImageLayout::eUndefined;
        if(data.get_mask().contains(dev.id))
            set_image(dev.id, name, {{info}});
    }
}

void push_descriptor_set::set_buffer(
    device_id id,
    std::string_view name,
    std::vector<vk::DescriptorBufferInfo>&& buffers
){
    if(named_bindings.count(name) == 0 || buffers.size() == 0) return;

    set_binding bind = find_binding(name);

    set_data& sd = data[id];
    if(buffers.size() > bind.descriptorCount)
        throw std::runtime_error("More buffers than descriptor allows!");

    auto& infos = sd.buffer_info_index < sd.tmp_buffer_infos.size() ?
        sd.tmp_buffer_infos[sd.buffer_info_index] :
        sd.tmp_buffer_infos.emplace_back(std::vector<vk::DescriptorBufferInfo>());
    sd.buffer_info_index++;
    infos = std::move(buffers);

    if(bind.descriptorType != vk::DescriptorType::eStorageBuffer && bind.descriptorType != vk::DescriptorType::eUniformBuffer)
        throw std::runtime_error("Cannot set non-buffer descriptor as a buffer!");

    for(size_t i = 0;;)
    {
        while(i < infos.size() && !infos[i].buffer) ++i;
        uint32_t begin = i;
        while(i < infos.size() && infos[i].buffer) ++i;
        uint32_t end = i;

        if(end == begin)
            break;

        vk::WriteDescriptorSet write = {
            VK_NULL_HANDLE, (uint32_t)bind.binding, (uint32_t)begin,
            end-begin, bind.descriptorType,
            nullptr, infos.data()+begin, nullptr
        };
        sd.writes.push_back(write);
    }
}

void push_descriptor_set::set_buffer(
    std::string_view name,
    const gpu_buffer& buffer,
    uint32_t offset
){
    for(device& dev: buffer.get_mask())
    {
        if(data.get_mask().contains(dev.id))
            set_buffer(dev.id, name, {{buffer[dev.id], offset, VK_WHOLE_SIZE}});
    }
}

void push_descriptor_set::set_acceleration_structure(
    device_id id,
    std::string_view name,
    vk::AccelerationStructureKHR tlas
){
    if(named_bindings.count(name) == 0) return;

    set_binding bind = find_binding(name);
    set_data& sd = data[id];

    if(bind.descriptorType != vk::DescriptorType::eAccelerationStructureKHR)
        throw std::runtime_error(
            "Cannot set non-acceleration structure descriptor as an acceleration "
            "structure!"
        );

    if(sd.tmp_as.size() < bindings.size())
    {
        sd.tmp_as.resize(bindings.size());
        sd.tmp_as_infos.resize(bindings.size());
    }

    vk::WriteDescriptorSetAccelerationStructureKHR& as_write = sd.tmp_as_infos[sd.as_info_index];
    vk::AccelerationStructureKHR& as = sd.tmp_as[sd.as_info_index];
    sd.as_info_index++;
    as = tlas;
    as_write.accelerationStructureCount = 1;
    as_write.pAccelerationStructures = &as;

    vk::WriteDescriptorSet write = {
        VK_NULL_HANDLE, (uint32_t)bind.binding, 0,
        (uint32_t)1, bind.descriptorType,
        nullptr, nullptr, nullptr
    };
    write.pNext = &as_write;
    sd.writes.push_back(write);
}

void push_descriptor_set::push(
    device_id id,
    vk::CommandBuffer buf,
    vk::PipelineLayout pipeline_layout,
    vk::PipelineBindPoint bind_point,
    uint32_t set_index
){
    set_data& sd = data[id];
    if(sd.writes.size() != 0)
    {
        buf.pushDescriptorSetKHR(
            bind_point,
            pipeline_layout,
            set_index,
            sd.writes.size(),
            sd.writes.data()
        );
    }
    sd.image_info_index = 0;
    sd.buffer_info_index = 0;
    sd.as_info_index = 0;
    sd.writes.clear();
}

}

