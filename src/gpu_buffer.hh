#ifndef TAURAY_GPU_BUFFER_HH
#define TAURAY_GPU_BUFFER_HH
#include "context.hh"

namespace tr
{

// If you need to give a buffer to the GPU that gets updated often, this class
// is the one you want. It wraps staging buffers and other buffer handling
// shenanigans into one simple package. It also automatically handles
// duplicating data to all specified devices, although this adds some overhead
// (only present if there are more than one device involved, though.)
class gpu_buffer
{
public:
    gpu_buffer();
    gpu_buffer(
        device_mask dev,
        size_t size,
        vk::BufferUsageFlags flags
    );

    // May reallocate buffers. Returns true if so.
    bool resize(size_t size);
    size_t get_size() const;

    operator bool() const;
    vk::Buffer operator[](device_id id) const;
    vk::DeviceAddress get_address(device_id id) const;

    device_mask get_mask() const;

    void update(uint32_t frame_index, const void* data, size_t offset = 0, size_t bytes = SIZE_MAX);
    void update_one(device_id id, uint32_t frame_index, const void* data, size_t offset = 0, size_t bytes = SIZE_MAX);
    template<typename T, typename F>
    void foreach(uint32_t frame_index, size_t entries, F&& f);
    template<typename T, typename F>
    void map(uint32_t frame_index, F&& f);
    template<typename T, typename F>
    void map_one(device_id id, uint32_t frame_index, F&& f);
    void upload(device_id id, uint32_t frame_index, vk::CommandBuffer cb);

    size_t calc_buffer_entry_alignment(device_id id, size_t entry_size) const;

private:
    void ensure_shared_data();
    size_t capacity;
    size_t size;
    std::unique_ptr<char[]> shared_data;

    // These initial parameters must be stored so that resize() can reallocate.
    // There's no other use for them.
    vk::BufferUsageFlags flags;

    struct buffer_data
    {
        vkm<vk::Buffer> buffer;
        vkm<vk::Buffer> staging[MAX_FRAMES_IN_FLIGHT];
    };
    per_device<buffer_data> buffers;
};

template<typename T, typename F>
void gpu_buffer::foreach(uint32_t frame_index, size_t entries, F&& f)
{
    if(!*this) return;

    size_t alignment = sizeof(T);
    if(buffers.get_mask().size() == 1)
    {
        device& dev = *buffers.get_mask().begin();
        buffer_data& buf = buffers[dev.id];

        if(flags & vk::BufferUsageFlagBits::eUniformBuffer)
            alignment = calc_buffer_entry_alignment(dev.id, alignment);

        char* data = nullptr;
        vkm<vk::Buffer>& staging_buffer = buf.staging[frame_index];
        vmaMapMemory(dev.allocator, staging_buffer.get_allocation(), (void**)&data);

        for(size_t i = 0; i < entries; ++i)
        {
            T& entry = *reinterpret_cast<T*>(data + alignment * i);
            f(entry, i);
        }

        vmaUnmapMemory(dev.allocator, staging_buffer.get_allocation());
    }
    else
    {
        ensure_shared_data();
        for(size_t i = 0; i < entries; ++i)
        {
            T& entry = *reinterpret_cast<T*>(shared_data.get() + alignment * i);
            f(entry, i);
        }
        if(flags & vk::BufferUsageFlagBits::eUniformBuffer)
        {
            // Harder update since devices may have incompatible alignment
            // requirements :/
            for(auto[dev, buf]: buffers)
            {
                size_t local_alignment = calc_buffer_entry_alignment(dev.id, alignment);
                char* data = nullptr;
                vkm<vk::Buffer>& staging_buffer = buf.staging[frame_index];
                vmaMapMemory(dev.allocator, staging_buffer.get_allocation(), (void**)&data);

                for(size_t i = 0; i < entries; ++i)
                    memcpy(data + local_alignment * i, shared_data.get() + alignment * i, sizeof(T));

                vmaUnmapMemory(dev.allocator, staging_buffer.get_allocation());
            }
        }
        else update(frame_index, shared_data.get(), 0, size);
    }
}

template<typename T, typename F>
void gpu_buffer::map(uint32_t frame_index, F&& f)
{
    if(!*this) return;

    if(buffers.get_mask().size() == 1)
    {
        map_one<T, F>((*buffers.get_mask().begin()).id, frame_index, std::forward<F>(f));
    }
    else
    {
        ensure_shared_data();
        f(reinterpret_cast<T*>(shared_data.get()));
        update(frame_index, shared_data.get(), 0, size);
    }
}

template<typename T, typename F>
void gpu_buffer::map_one(device_id id, uint32_t frame_index, F&& f)
{
    if(!*this) return;

    device& dev = buffers.get_device(id);
    buffer_data& buf = buffers[dev.id];

    vkm<vk::Buffer>& staging_buffer = buf.staging[frame_index];
    T* data = nullptr;
    vmaMapMemory(dev.allocator, staging_buffer.get_allocation(), (void**)&data);
    f(data);
    vmaUnmapMemory(dev.allocator, staging_buffer.get_allocation());
}

}

#endif
