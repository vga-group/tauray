#include "vkm.hh"
#include "context.hh"
#include <utility>
#include <type_traits>

namespace tr
{

template<typename T>
vkm_generic<T>::vkm_generic()
: dev(nullptr), resource(VK_NULL_HANDLE)
{
}

template<typename T>
vkm_generic<T>::vkm_generic(device& dev, T resource)
: dev(&dev), resource(resource)
{
}

template<typename T>
vkm_generic<T>::vkm_generic(vkm_generic<T>&& other)
: dev(nullptr), resource(VK_NULL_HANDLE)
{
    operator=(std::move(other));
}

template<typename T>
vkm_generic<T>& vkm_generic<T>::operator=(vkm_generic<T>&& other)
{
    queue_destroy();
    dev = other.dev;
    resource = other.resource;
    other.dev = nullptr;
    other.resource = VK_NULL_HANDLE;
    return *this;
}

template<typename T>
vkm_generic<T>::operator T() const
{
    return resource;
}

template<typename T>
vkm_generic<T>::operator const T*() const
{
    return get();
}

template<typename T>
vkm_generic<T>::operator T*()
{
    return get();
}

template<typename T>
T vkm_generic<T>::operator*() const
{
    return resource;
}

template<typename T>
const T* vkm_generic<T>::operator->() const
{
    return get();
}

template<typename T>
const T* vkm_generic<T>::get() const
{
    return &resource;
}

template<typename T>
T* vkm_generic<T>::get()
{
    return &resource;
}

template<typename T>
void vkm_generic<T>::destroy()
{
    std::function<void()> func;
    if(dev && static_cast<vkm<T>*>(this)->destroy_func(func))
    {
        func();
        dev = nullptr;
        resource = VK_NULL_HANDLE;
    }
}

template<typename T>
void vkm_generic<T>::drop()
{
    queue_destroy();
    dev = nullptr;
    resource = VK_NULL_HANDLE;
}

template<typename T>
void vkm_generic<T>::leak()
{
    dev = nullptr;
}

template<typename T>
vk::Device vkm_generic<T>::get_device()
{
    return dev->dev;
}

template<typename T>
void vkm_generic<T>::queue_destroy()
{
    std::function<void()> func;
    if(dev && static_cast<vkm<T>*>(this)->destroy_func(func))
    {
        dev->ctx->queue_frame_finish_callback(std::move(func));
        dev = nullptr;
        resource = VK_NULL_HANDLE;
    }
}

template<typename T>
vkm<T>::vkm(device& dev, T resource)
: vkm_generic<T>(dev, resource) {}

template<typename T>
vkm<T>::~vkm() { vkm_generic<T>::queue_destroy(); }

template<typename T>
bool vkm<T>::destroy_func(std::function<void()>& func)
{
    vk::Device dev = vkm_generic<T>::get_device();
    T resource = vkm_generic<T>::resource;
    if(!resource)
        return false;

    if constexpr(std::is_same_v<vk::ImageView, T>)
    {
        func = [res = resource, dev = dev](){ dev.destroyImageView(res); };
        return true;
    }
    else if constexpr(std::is_same_v<vk::DescriptorPool, T>)
    {
        func = [res = resource, dev = dev](){ dev.destroyDescriptorPool(res); };
        return true;
    }
    else if constexpr(std::is_same_v<vk::AccelerationStructureKHR, T>)
    {
        func = [res = resource, dev = dev](){ dev.destroyAccelerationStructureKHR(res); };
        return true;
    }
    else if constexpr(std::is_same_v<vk::Sampler, T>)
    {
        func = [res = resource, dev = dev](){ dev.destroySampler(res); };
        return true;
    }
    else if constexpr(std::is_same_v<vk::ShaderModule, T>)
    {
        func = [res = resource, dev = dev](){ dev.destroyShaderModule(res); };
        return true;
    }
    else if constexpr(std::is_same_v<vk::Pipeline, T>)
    {
        func = [res = resource, dev = dev](){ dev.destroyPipeline(res); };
        return true;
    }
    else if constexpr(std::is_same_v<vk::PipelineLayout, T>)
    {
        func = [res = resource, dev = dev](){ dev.destroyPipelineLayout(res); };
        return true;
    }
    else if constexpr(std::is_same_v<vk::DescriptorSetLayout, T>)
    {
        func = [res = resource, dev = dev](){ dev.destroyDescriptorSetLayout(res); };
        return true;
    }
    else if constexpr(std::is_same_v<vk::RenderPass, T>)
    {
        func = [res = resource, dev = dev](){ dev.destroyRenderPass(res); };
        return true;
    }
    else if constexpr(std::is_same_v<vk::Semaphore, T>)
    {
        func = [res = resource, dev = dev](){ dev.destroySemaphore(res); };
        return true;
    }
    else if constexpr(std::is_same_v<vk::Framebuffer, T>)
    {
        func = [res = resource, dev = dev](){ dev.destroyFramebuffer(res); };
        return true;
    }
    else if constexpr(std::is_same_v<vk::QueryPool, T>)
    {
        func = [res = resource, dev = dev](){ dev.destroyQueryPool(res); };
        return true;
    }
    else if constexpr(std::is_same_v<vk::Fence, T>)
    {
        func = [res = resource, dev = dev](){ dev.destroyFence(res); };
        return true;
    }
    return false;
}

vkm<vk::Image>::vkm(device& dev, vk::Image img, VmaAllocation alloc)
: vkm_generic<vk::Image>(dev, img), alloc(alloc)
{}

vkm<vk::Image>::~vkm() { vkm_generic<vk::Image>::queue_destroy(); }

bool vkm<vk::Image>::destroy_func(std::function<void()>& func)
{
    if(resource && alloc)
    {
        func = [img=resource, allocator=dev->allocator, alloc=alloc](){
            vmaDestroyImage(allocator, img, alloc);
        };
        return true;
    }
    return false;
}

vkm<vk::CommandBuffer>::vkm(device& dev, vk::CommandBuffer cmd, vk::CommandPool pool)
: vkm_generic<vk::CommandBuffer>(dev, cmd), pool(pool)
{}

vkm<vk::CommandBuffer>::~vkm() { vkm_generic<vk::CommandBuffer>::queue_destroy(); }

vk::CommandPool vkm<vk::CommandBuffer>::get_pool() const
{
    return pool;
}

bool vkm<vk::CommandBuffer>::destroy_func(std::function<void()>& func)
{
    if(resource && pool)
    {
        func = [cmd=resource, dev=dev->dev, pool=pool](){
            dev.freeCommandBuffers(pool, cmd);
        };
        return true;
    }
    return false;
}

vkm<vk::Buffer>::vkm(device& dev, vk::Buffer buf, VmaAllocation alloc)
: vkm_generic<vk::Buffer>(dev, buf), alloc(alloc)
{}

vkm<vk::Buffer>::~vkm() { vkm_generic<vk::Buffer>::queue_destroy(); }

VmaAllocation vkm<vk::Buffer>::get_allocation() const
{
    return alloc;
}

vk::DeviceAddress vkm<vk::Buffer>::get_address() const
{
    return dev->dev.getBufferAddress({resource});
}

bool vkm<vk::Buffer>::destroy_func(std::function<void()>& func)
{
    if(resource && alloc)
    {
        func = [buf=resource, allocator=dev->allocator, alloc=alloc](){
            vmaDestroyBuffer(allocator, buf, alloc);
        };
        return true;
    }
    return false;
}

template class vkm_generic<vk::Image>;
template class vkm_generic<vk::CommandBuffer>;
template class vkm_generic<vk::Buffer>;
template class vkm_generic<vk::ImageView>;
template class vkm_generic<vk::DescriptorPool>;
template class vkm_generic<vk::AccelerationStructureKHR>;
template class vkm_generic<vk::Sampler>;
template class vkm_generic<vk::ShaderModule>;
template class vkm_generic<vk::Pipeline>;
template class vkm_generic<vk::PipelineLayout>;
template class vkm_generic<vk::DescriptorSetLayout>;
template class vkm_generic<vk::RenderPass>;
template class vkm_generic<vk::Semaphore>;
template class vkm_generic<vk::Framebuffer>;
template class vkm_generic<vk::QueryPool>;
template class vkm_generic<vk::Fence>;
template class vkm<vk::ImageView>;
template class vkm<vk::DescriptorPool>;
template class vkm<vk::AccelerationStructureKHR>;
template class vkm<vk::Sampler>;
template class vkm<vk::ShaderModule>;
template class vkm<vk::Pipeline>;
template class vkm<vk::PipelineLayout>;
template class vkm<vk::DescriptorSetLayout>;
template class vkm<vk::RenderPass>;
template class vkm<vk::Semaphore>;
template class vkm<vk::Framebuffer>;
template class vkm<vk::QueryPool>;
template class vkm<vk::Fence>;

}
