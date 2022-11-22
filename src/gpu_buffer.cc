#include "gpu_buffer.hh"
#include "misc.hh"

namespace tr
{

gpu_buffer::gpu_buffer()
:   dev(nullptr), capacity(0), size(0), flags(0)
{}

gpu_buffer::gpu_buffer(
    device_data& dev,
    size_t size,
    vk::BufferUsageFlags flags
):  dev(&dev), capacity(0), size(size), flags(flags)
{
    resize(size);
}

// May reallocate buffers. Returns true if so.
bool gpu_buffer::resize(size_t size)
{
    this->size = size;
    if(this->capacity >= size) return false;

    this->capacity = size;

    vk::BufferCreateInfo gpu_buffer_info(
        {}, this->capacity,
        flags|vk::BufferUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive
    );

    buffer = create_buffer(*dev, gpu_buffer_info, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        staging[i] = create_staging_buffer(*dev, this->capacity, nullptr);
    return true;
}

size_t gpu_buffer::get_size() const
{
    return size;
}

gpu_buffer::operator bool() const
{
    return dev && size > 0;
}

vk::Buffer gpu_buffer::operator*() const
{
    return buffer;
}

vk::DeviceAddress gpu_buffer::get_address() const
{
    return dev->dev.getBufferAddress({*buffer});
}

void gpu_buffer::update(uint32_t frame_index, const void* data, size_t offset, size_t bytes)
{
    if(!*this) return;

    if(bytes == 0 || bytes > size - offset)
        bytes = size - offset;

    char* ptr = nullptr;
    vkm<vk::Buffer>& staging_buffer = staging[frame_index];
    vmaMapMemory(dev->allocator, staging_buffer.get_allocation(), (void**)&ptr);

    memcpy(ptr + offset, data, bytes);

    vmaUnmapMemory(dev->allocator, staging_buffer.get_allocation());
}

void gpu_buffer::upload(uint32_t frame_index, vk::CommandBuffer cb)
{
    if(size > 0)
    {
        vk::Buffer src = staging[frame_index];
        cb.copyBuffer(src, *buffer, {{0, 0, size}});
    }
}

size_t gpu_buffer::calc_buffer_entry_alignment(size_t entry_size) const
{
    uint32_t min_uniform_offset = dev->props.limits.minUniformBufferOffsetAlignment;
    return (entry_size+min_uniform_offset-1)/min_uniform_offset
        *min_uniform_offset;
}

}
