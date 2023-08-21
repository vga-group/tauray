#include "descriptor_state.hh"
#include "placeholders.hh"

namespace tr
{

descriptor_state::descriptor_state(
    const std::string& binding_name,
    size_t length
): binding_name(binding_name), length(length)
{
}

descriptor_state::descriptor_state(
    const std::string& binding_name,
    vk::DescriptorBufferInfo buffer,
    size_t repeat
): binding_name(binding_name), length(repeat), buffers(repeat, buffer)
{
}

descriptor_state::descriptor_state(
    const std::string& binding_name,
    vk::DescriptorImageInfo image,
    size_t repeat
): binding_name(binding_name), length(repeat), images(repeat, image)
{
}

descriptor_state::descriptor_state(
    const std::string& binding_name,
    vk::WriteDescriptorSetAccelerationStructureKHR as_info
): binding_name(binding_name), length(1), as_info(as_info)
{
}

descriptor_state::descriptor_state(
    const std::string& binding_name,
    const std::vector<vk::DescriptorBufferInfo>& buffers
): binding_name(binding_name), length(buffers.size()), buffers(buffers)
{
}

descriptor_state::descriptor_state(
    const std::string& binding_name,
    const std::vector<vk::DescriptorImageInfo>& images
): binding_name(binding_name), length(images.size()), images(images)
{
}

const std::string& descriptor_state::get_binding_name() const
{
    return binding_name;
}

bool descriptor_state::is_empty() const
{
    return length == 0;
}

vk::WriteDescriptorSet descriptor_state::get_write(
    placeholders& pl,
    uint32_t device_index,
    const vk::DescriptorSet& ds,
    const vk::DescriptorSetLayoutBinding& binding,
    std::vector<std::vector<vk::DescriptorBufferInfo>>& buffer_holder,
    std::vector<std::vector<vk::DescriptorImageInfo>>& image_holder
) const
{
    if(buffers.size() != 0)
    {
        // Can't write zero-sized buffers...
        if(buffers.size() == 1 && (buffers[0].range == 0 || !buffers[0].buffer))
            return get_placeholder_write(
                pl, device_index, ds, binding, buffer_holder, image_holder
            );
        return {
            ds, binding.binding, 0, (uint32_t)buffers.size(),
            binding.descriptorType, nullptr, buffers.data(), nullptr
        };
    }
    else if(images.size() != 0)
    {
        if(images.size() == 1 && !images[0].imageView)
            return get_placeholder_write(
                pl, device_index, ds, binding, buffer_holder, image_holder
            );
        return {
            ds, binding.binding, 0, (uint32_t)images.size(),
            binding.descriptorType, images.data(), nullptr, nullptr
        };
    }
    else if(as_info.has_value())
    {
        vk::WriteDescriptorSet ws{
            ds, binding.binding, 0, 1, binding.descriptorType,
            nullptr, nullptr, nullptr
        };
        ws.pNext = &as_info.value();
        return ws;
    }
    else
    {
        return get_placeholder_write(
            pl, device_index, ds, binding, buffer_holder, image_holder
        );
    }
}

vk::WriteDescriptorSet descriptor_state::get_placeholder_write(
    placeholders& pl,
    device_id id,
    const vk::DescriptorSet& ds,
    const vk::DescriptorSetLayoutBinding& binding,
    std::vector<std::vector<vk::DescriptorBufferInfo>>& buffer_holder,
    std::vector<std::vector<vk::DescriptorImageInfo>>& image_holder
) const
{
    // Placeholder time!
    if(binding.descriptorType == vk::DescriptorType::eCombinedImageSampler)
    {
        image_holder.emplace_back(
            length, pl.buffers[id].img_2d_info
        );
        return {
            ds, binding.binding, 0, (uint32_t)length, binding.descriptorType,
            image_holder.back().data(), nullptr, nullptr
        };
    }
    else if(binding.descriptorType == vk::DescriptorType::eStorageBuffer)
    {
        buffer_holder.emplace_back(
            length, pl.buffers[id].storage_info
        );
        return {
            ds, binding.binding, 0, (uint32_t)length, binding.descriptorType,
            nullptr, buffer_holder.back().data(), nullptr
        };
    }
    else
        throw std::runtime_error(
            "I don't know the correct placeholder for this descriptor type!"
        );
}

}
