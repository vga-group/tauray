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
            std::make_tuple(dev.id),
            std::forward_as_tuple()
        );
}

template<typename T>
template<typename... Args>
per_device<T>::per_device(device_mask mask, Args&&... args)
{
    active_mask = mask;
    for(device& dev: mask)
    {
        devices.emplace(
            std::piecewise_construct,
            std::make_tuple(dev.id),
            std::forward_as_tuple(dev, std::forward<Args>(args)...)
        );
    }
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
            std::make_tuple(dev.id),
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
        devices.emplace(dev.id, create_callback(dev));
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

template<typename T>
typename per_device<T>::iterator& per_device<T>::iterator::operator++()
{
    ++it;
    return *this;
}

template<typename T>
std::tuple<device&, T&> per_device<T>::iterator::operator*() const
{
    return std::forward_as_tuple(active_mask.get_device(it->first), it->second);
}

template<typename T>
bool per_device<T>::iterator::operator==(const iterator& other) const
{
    return it == other.it;
}

template<typename T>
bool per_device<T>::iterator::operator!=(const iterator& other) const
{
    return it != other.it;
}

template<typename T>
typename per_device<T>::iterator per_device<T>::begin()
{
    return {active_mask, devices.begin()};
}

template<typename T>
typename per_device<T>::iterator per_device<T>::end()
{
    return {active_mask, devices.end()};
}

}

#endif
