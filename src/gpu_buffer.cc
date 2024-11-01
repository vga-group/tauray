#include "gpu_buffer.hh"
#include "misc.hh"

namespace tr
{

gpu_buffer::gpu_buffer()
: capacity(0), size(0), flags(0)
{}

gpu_buffer::gpu_buffer(
    device_mask dev,
    size_t size,
    vk::BufferUsageFlags flags
):  capacity(0), size(size), flags(flags), buffers(dev)
{
    resize(size);
}

// May reallocate buffers. Returns true if so.
bool gpu_buffer::resize(size_t size)
{
    this->size = size;
    if(this->capacity >= size) return false;

    if(buffers.get_mask().size() > 1)
        shared_data.reset();

    this->capacity = size;

    vk::BufferCreateInfo gpu_buffer_info(
        {}, this->capacity,
        flags|vk::BufferUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive
    );

    for(auto[dev, buf]: buffers)
    {
        buf.buffer = create_buffer(dev, gpu_buffer_info, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
            buf.staging[i] = create_staging_buffer(dev, this->capacity, nullptr);
    }
    return true;
}

size_t gpu_buffer::get_size() const
{
    return size;
}

gpu_buffer::operator bool() const
{
    return buffers.get_mask().size() > 0 && size > 0;
}

vk::Buffer gpu_buffer::operator[](device_id id) const
{
    return buffers[id].buffer;
}

vk::DeviceAddress gpu_buffer::get_address(device_id id) const
{
    return buffers.get_device(id).logical.getBufferAddress({*buffers[id].buffer});
}

device_mask gpu_buffer::get_mask() const
{
    return buffers.get_mask();
}

void gpu_buffer::update(uint32_t frame_index, const void* data, size_t offset, size_t bytes)
{
    if(!*this) return;

    offset = std::min(size, offset);

    if(bytes > size - offset)
        bytes = size - offset;

    if(bytes == 0)
        return;

    for(auto[dev, buf]: buffers)
    {
        vkm<vk::Buffer>& staging_buffer = buf.staging[frame_index];
        char* ptr = nullptr;
        vmaMapMemory(dev.allocator, staging_buffer.get_allocation(), (void**)&ptr);
        memcpy(ptr + offset, data, bytes);
        vmaUnmapMemory(dev.allocator, staging_buffer.get_allocation());
    }
}

void gpu_buffer::update_one(device_id id, uint32_t frame_index, const void* data, size_t offset, size_t bytes)
{
    if(!*this) return;

    offset = std::min(size, offset);

    if(bytes == 0 || bytes > size - offset)
        bytes = size - offset;

    if(bytes == 0)
        return;

    device& dev = buffers.get_device(id);
    buffer_data& buf = buffers[id];

    vkm<vk::Buffer>& staging_buffer = buf.staging[frame_index];
    char* ptr = nullptr;
    vmaMapMemory(dev.allocator, staging_buffer.get_allocation(), (void**)&ptr);
    memcpy(ptr + offset, data, bytes);
    vmaUnmapMemory(dev.allocator, staging_buffer.get_allocation());
}

void gpu_buffer::upload(device_id id, uint32_t frame_index, vk::CommandBuffer cb)
{
    if(size > 0)
    {
        buffer_data& buf = buffers[id];
        vk::Buffer src = buf.staging[frame_index];
        cb.copyBuffer(src, *buf.buffer, {{0, 0, size}});
    }
}

size_t gpu_buffer::calc_buffer_entry_alignment(device_id id, size_t entry_size) const
{
    uint32_t min_uniform_offset = buffers.get_device(id).props.limits.minUniformBufferOffsetAlignment;
    return (entry_size+min_uniform_offset-1)/min_uniform_offset*min_uniform_offset;
}

void gpu_buffer::ensure_shared_data()
{
    if(!shared_data)
        shared_data.reset(new char[size]);
}

}
