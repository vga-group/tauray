#ifndef TAURAY_DESCRIPTOR_STATE_HH
#define TAURAY_DESCRIPTOR_STATE_HH
#include "context.hh"
#include <optional>

namespace tr
{

class descriptor_state
{
public:
    descriptor_state(const std::string& binding, size_t length = 1);

    descriptor_state(
        const std::string& binding,
        vk::DescriptorBufferInfo buffer,
        size_t repeat = 1
    );

    descriptor_state(
        const std::string& binding,
        vk::DescriptorImageInfo image,
        size_t repeat = 1
    );

    descriptor_state(
        const std::string& binding,
        vk::WriteDescriptorSetAccelerationStructureKHR as_info
    );

    descriptor_state(
        const std::string& binding,
        const std::vector<vk::DescriptorBufferInfo>& buffers
    );

    descriptor_state(
        const std::string& binding,
        const std::vector<vk::DescriptorImageInfo>& images
    );

    const std::string& get_binding_name() const;

    bool is_empty() const;

    vk::WriteDescriptorSet get_write(
        placeholders& pl,
        device_id id,
        const vk::DescriptorSet& ds,
        const vk::DescriptorSetLayoutBinding& binding,
        std::vector<std::vector<vk::DescriptorBufferInfo>>& buffer_holder,
        std::vector<std::vector<vk::DescriptorImageInfo>>& image_holder
    ) const;

    vk::WriteDescriptorSet get_placeholder_write(
        placeholders& pl,
        device_id id,
        const vk::DescriptorSet& ds,
        const vk::DescriptorSetLayoutBinding& binding,
        std::vector<std::vector<vk::DescriptorBufferInfo>>& buffer_holder,
        std::vector<std::vector<vk::DescriptorImageInfo>>& image_holder
    ) const;

private:
    std::string binding_name;
    size_t length;
    std::vector<vk::DescriptorBufferInfo> buffers;
    std::vector<vk::DescriptorImageInfo> images;
    std::optional<vk::WriteDescriptorSetAccelerationStructureKHR> as_info;
};

}

#endif
