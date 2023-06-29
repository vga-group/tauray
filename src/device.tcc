#ifndef TAURAY_DEVICE_TCC
#define TAURAY_DEVICE_TCC
#include "device.hh"

namespace tr
{

template<typename T>
per_device<T>::per_device()
{
}

template<typename T>
per_device<T>::per_device(device_mask mask)
{
    active_mask = mask;
    for(device& dev: mask)
        devices.emplace(
            std::piecewise_construct,
            std::make_tuple(dev.index),
            std::forward_as_tuple()
        );
}

template<typename T>
template<typename... Args>
void per_device<T>::emplace(device_mask mask, Args&&... args)
{
    active_mask = mask;
    for(device& dev: mask)
    {
        devices.emplace(
            std::piecewise_construct,
            std::make_tuple(dev.index),
            std::forward_as_tuple(dev, std::forward<Args>(args)...)
        );
    }
}

template<typename T>
template<typename F>
void per_device<T>::init(device_mask mask, F&& create_callback)
{
    active_mask = mask;
    for(device& dev: mask)
        devices.emplace(dev.index, create_callback(dev));
}

template<typename T>
T& per_device<T>::operator[](device_id id)
{
    return devices.at(id);
}

template<typename T>
const T& per_device<T>::operator[](device_id id) const
{
    return devices.at(id);
}

template<typename T>
template<typename F>
void per_device<T>::operator()(F&& callback)
{
    for(device& dev: active_mask)
        callback(dev, devices.at(dev.index));
}

template<typename T>
device_mask per_device<T>::get_mask() const
{
    return active_mask;
}

template<typename T>
context* per_device<T>::get_context() const
{
    return active_mask.get_context();
}

template<typename T>
void per_device<T>::clear()
{
    active_mask.clear();
    devices.clear();
}

template<typename T>
device& per_device<T>::get_device(device_id id) const
{
    return active_mask.get_device(id);
}

}

#endif
