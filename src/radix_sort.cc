#include "radix_sort.hh"
#include "radix_sort/radix_sort_vk.h"
#include "context.hh"
#include "misc.hh"

namespace
{
using namespace tr;

struct reorder_push_constants
{
    uint32_t item_size;
    uint32_t item_count;
};

}

namespace tr
{

radix_sort::radix_sort(device& dev)
: dev(&dev), desc(dev), reorder(dev)
{
    shader_source src("shader/array_reorder.comp");
    desc.add(src);
    reorder.init(src, {&desc});

    radix_sort_vk_target* rs_target =  radix_sort_vk_target_auto_detect(
        (VkPhysicalDeviceProperties*)&dev.props,
        (VkPhysicalDeviceSubgroupProperties*)&dev.subgroup_props,
        2
    );

    rs_instance = radix_sort_vk_create(
        dev.logical,
        nullptr,
        VK_NULL_HANDLE,
        rs_target
    );

    free(rs_target);
}

radix_sort::~radix_sort()
{
    dev->ctx->queue_frame_finish_callback([rs_instance = rs_instance, dev = dev->logical](){
        radix_sort_vk_destroy((radix_sort_vk_t*)rs_instance, dev, nullptr);
    });
}

vkm<vk::Buffer> radix_sort::create_keyval_buffer(size_t max_items)
{
    radix_sort_vk_memory_requirements_t memory_requirements;
    radix_sort_vk_get_memory_requirements(
        (const radix_sort_vk_t*)rs_instance, max_items, &memory_requirements
    );

    size_t total_alignment = max(
        memory_requirements.internal_alignment,
        memory_requirements.keyvals_alignment
    );
    size_t keyval_buf_size = (memory_requirements.keyvals_size + total_alignment-1)/total_alignment*total_alignment;
    size_t internal_buf_size = memory_requirements.internal_size;

    return create_buffer_aligned(
        *dev,
        {
            {},
            keyval_buf_size * 2 + internal_buf_size,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress| vk::BufferUsageFlagBits::eTransferDst,
            vk::SharingMode::eExclusive
        },
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        total_alignment
    );
}

vk::DescriptorBufferInfo radix_sort::sort(
    vk::CommandBuffer cmd,
    vk::Buffer item_keyvals,
    size_t item_count,
    size_t key_bits
){
    size_t total_alignment;
    size_t keyval_buf_size;
    size_t internal_buf_size;
    get_alloc_info(
        item_count,
        total_alignment,
        keyval_buf_size,
        internal_buf_size
    );

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, {}, {{
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite,
            {}, {}, item_keyvals, 0, keyval_buf_size
        }}, {}
    );

    radix_sort_vk_memory_requirements_t memory_requirements;
    radix_sort_vk_get_memory_requirements(
        (const radix_sort_vk_t*)rs_instance, item_count, &memory_requirements);

    radix_sort_vk_sort_info_t sort_info = {
        nullptr,
        (uint32_t)key_bits,
        (uint32_t)item_count,
        {item_keyvals, 0, keyval_buf_size},
        {item_keyvals, keyval_buf_size, keyval_buf_size},
        {item_keyvals, 2 * keyval_buf_size, internal_buf_size}
    };
    radix_sort_vk_sort(
        (const radix_sort_vk_t*)rs_instance,
        &sort_info,
        dev->logical,
        cmd,
        (VkDescriptorBufferInfo*)&sorted_keyvals_buf
    );
    return sorted_keyvals_buf;
}

void radix_sort::sort(
    vk::CommandBuffer cb,
    vk::Buffer input_items,
    vk::Buffer output_items,
    vk::Buffer item_keyvals,
    size_t item_size,
    size_t item_count,
    size_t key_bits
){
    assert(item_size % sizeof(uint32_t) == 0);

    sort(cb, item_keyvals, item_count, key_bits);
    resort(cb, input_items, output_items, item_size, item_count);
}

void radix_sort::resort(
    vk::CommandBuffer cmd,
    vk::Buffer input_items,
    vk::Buffer output_items,
    size_t item_size,
    size_t item_count
){
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, {}, {
            {
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite,
                {}, {}, sorted_keyvals_buf.buffer, sorted_keyvals_buf.offset, sorted_keyvals_buf.range
            },
            {
                vk::AccessFlagBits::eMemoryWrite,
                vk::AccessFlagBits::eShaderRead,
                {}, {}, input_items, 0, item_count * item_size
            },
            {
                vk::AccessFlagBits::eShaderRead,
                vk::AccessFlagBits::eShaderWrite,
                {}, {}, output_items, 0, item_count * item_size
            }
        }, {}
    );

    reorder.bind(cmd);
    desc.set_buffer(dev->id, "keyval_data", {sorted_keyvals_buf});
    desc.set_buffer(dev->id, "input_data", {{input_items, 0, item_size * item_count}});
    desc.set_buffer(dev->id, "output_data", {{output_items, 0, item_size * item_count}});
    reorder.push_descriptors(cmd, desc, 0);
    reorder.push_constants(cmd, reorder_push_constants{(uint32_t)(item_size/sizeof(uint32_t)), (uint32_t)item_count});
    cmd.dispatch((item_count * (item_size / sizeof(uint32_t)) + 255u)/256u, 1, 1);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eAllCommands,
        {}, {}, {
            {
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead,
                {}, {}, output_items, 0, item_count * item_size
            }
        }, {}
    );
}

void radix_sort::get_alloc_info(
    size_t max_count,
    size_t& alloc_alignment,
    size_t& alloc_keyvals_size,
    size_t& alloc_internal_size
) const
{
    radix_sort_vk_memory_requirements_t memory_requirements;
    radix_sort_vk_get_memory_requirements(
        (const radix_sort_vk_t*)rs_instance, max_count, &memory_requirements
    );

    assert(memory_requirements.keyval_size == 8);

    alloc_alignment = max(
        memory_requirements.internal_alignment,
        memory_requirements.keyvals_alignment
    );
    alloc_keyvals_size = (memory_requirements.keyvals_size + alloc_alignment-1)/alloc_alignment*alloc_alignment;
    alloc_internal_size = memory_requirements.internal_size;
}

}
