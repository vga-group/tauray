#ifndef TAURAY_VKM_HH
#define TAURAY_VKM_HH

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include "vk_mem_alloc.h"
#include <vulkan/vulkan_beta.h>
#include <vulkan/vulkan.hpp>
#include <functional>

namespace tr
{

struct device_data;

template<typename T>
class vkm_generic
{
public:
    vkm_generic();
    vkm_generic(device_data& dev, T resource);
    vkm_generic(const vkm_generic<T>& other) = delete;
    vkm_generic(vkm_generic<T>&& other);

    vkm_generic<T>& operator=(vkm_generic<T>&& other);
    operator T() const;
    operator const T*() const;
    operator T*();
    T operator*() const;
    const T* operator->() const;
    const T* get() const;
    T* get();

    // Immediately destroys the object instead of queuing destruction.
    // You should only use this with large temporary buffers which you know
    // cannot be used anymore.
    void destroy();

    // Queues destruction similar to the destructor, but allows you to set it
    // again to something else.
    void drop();

    // Makes the thing not be destroyed on destruction.
    void leak();

protected:
    vk::Device get_device();
    void queue_destroy();

    device_data* dev;
    T resource;
};

template<typename T>
class vkm: public vkm_generic<T>
{
public:
    vkm() = default;
    vkm(device_data& dev, T resource);
    vkm(vkm<T>&& other) = default;
    ~vkm();

    vkm<T>& operator=(vkm<T>&& other) = default;

private:
    friend class vkm_generic<T>;
    bool destroy_func(std::function<void()>& func);
};

template<>
class vkm<vk::Image>: public vkm_generic<vk::Image>
{
public:
    vkm() = default;
    vkm(device_data& dev, vk::Image img, VmaAllocation alloc = VK_NULL_HANDLE);
    vkm(vkm<vk::Image>&& other) = default;
    ~vkm();

    vkm<vk::Image>& operator=(vkm<vk::Image>&& other) = default;

private:
    friend class vkm_generic<vk::Image>;
    bool destroy_func(std::function<void()>& func);
    VmaAllocation alloc;
};

template<>
class vkm<vk::CommandBuffer>: public vkm_generic<vk::CommandBuffer>
{
public:
    vkm() = default;
    vkm(device_data& dev, vk::CommandBuffer cmd, vk::CommandPool pool);
    vkm(vkm<vk::CommandBuffer>&& other) = default;
    ~vkm();

    vkm<vk::CommandBuffer>& operator=(vkm<vk::CommandBuffer>&& other) = default;
    vk::CommandPool get_pool() const;

private:
    friend class vkm_generic<vk::CommandBuffer>;
    bool destroy_func(std::function<void()>& func);
    vk::CommandPool pool;
};

template<>
class vkm<vk::Buffer>: public vkm_generic<vk::Buffer>
{
public:
    vkm() = default;
    vkm(device_data& dev, vk::Buffer buf, VmaAllocation alloc = VK_NULL_HANDLE);
    vkm(vkm<vk::Buffer>&& other) = default;
    ~vkm();

    vkm<vk::Buffer>& operator=(vkm<vk::Buffer>&& other) = default;
    VmaAllocation get_allocation() const;
    vk::DeviceAddress get_address() const;

private:
    friend class vkm_generic<vk::Buffer>;
    bool destroy_func(std::function<void()>& func);
    VmaAllocation alloc;
};

template<typename T>
void unwrap_vkm_vector(const std::vector<tr::vkm<T>>& vec, std::vector<T>& to)
{
    std::vector<T> res;
    to.clear();
    to.reserve(vec.size());
    for(const tr::vkm<T>& v: vec)
        to.push_back(*v);
}

}

#endif
