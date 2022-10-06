#ifndef TAURAY_GPU_BUFFER_HH
#define TAURAY_GPU_BUFFER_HH
#include "context.hh"

namespace tr
{

// If you need to give a buffer to the GPU that gets updated often, this class
// is the one you want. It wraps staging buffers and other buffer handling
// shenanigans into one simple package.
// Replaces the struct buffer_data of yore, along with manual staging buffers.
class gpu_buffer
{
public:
    gpu_buffer();
    gpu_buffer(
        device_data& dev,
        size_t size,
        vk::BufferUsageFlags flags
    );

    // May reallocate buffers. Returns true if so.
    bool resize(size_t size);
    size_t get_size() const;

    operator bool() const;
    vk::Buffer operator*() const;
    vk::DeviceAddress get_address() const;

    void update(uint32_t frame_index, const void* data, size_t offset = 0, size_t bytes = 0);
    template<typename T, typename F>
    void foreach(uint32_t frame_index, size_t entries, F&& f);
    template<typename T, typename F>
    void map(uint32_t frame_index, F&& f);
    void upload(uint32_t frame_index, vk::CommandBuffer cb);

    size_t calc_buffer_entry_alignment(size_t entry_size) const;

private:
    device_data* dev;
    size_t capacity;
    size_t size;
    // These initial parameters must be stored so that resize() can reallocate.
    // There's no other use for them.
    vk::BufferUsageFlags flags;

    vkm<vk::Buffer> buffer;
    vkm<vk::Buffer> staging[MAX_FRAMES_IN_FLIGHT];
};

template<typename T, typename F>
void gpu_buffer::foreach(uint32_t frame_index, size_t entries, F&& f)
{
    if(!*this) return;

    size_t alignment = sizeof(T);
    if(flags & vk::BufferUsageFlagBits::eUniformBuffer)
        alignment = calc_buffer_entry_alignment(alignment);

    char* data = nullptr;
    vkm<vk::Buffer>& staging_buffer = staging[frame_index];
    vmaMapMemory(dev->allocator, staging_buffer.get_allocation(), (void**)&data);

    for(size_t i = 0; i < entries; ++i)
    {
        T& entry = *reinterpret_cast<T*>(data + alignment * i);
        f(entry, i);
    }

    vmaUnmapMemory(dev->allocator, staging_buffer.get_allocation());
}

template<typename T, typename F>
void gpu_buffer::map(uint32_t frame_index, F&& f)
{
    if(!*this) return;

    T* data = nullptr;
    vkm<vk::Buffer>& staging_buffer = staging[frame_index];
    vmaMapMemory(dev->allocator, staging_buffer.get_allocation(), (void**)&data);

    f(data);

    vmaUnmapMemory(dev->allocator, staging_buffer.get_allocation());
}

}

#endif
