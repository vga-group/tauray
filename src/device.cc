#include "device.hh"
#include "context.hh"
#include "math.hh"

namespace tr
{

device_mask::device_mask(): ctx(nullptr), bitmask(0) {}

device_mask::device_mask(device& dev)
: ctx(dev.ctx), bitmask(1llu << uint64_t(dev.index))
{
}

device_mask device_mask::all(context& ctx)
{
    device_mask dm;
    dm.ctx = &ctx;
    dm.bitmask = (1llu<<ctx.get_devices().size())-1;
    return dm;
}

device_mask device_mask::none(context& ctx)
{
    device_mask dm;
    dm.ctx = &ctx;
    dm.bitmask = 0;
    return dm;
}

bool device_mask::contains(device_id id) const
{
    return (bitmask >> (uint64_t)id)&1;
}

void device_mask::erase(device_id id)
{
    bitmask &= ~(1llu << (uint64_t)id);
}

void device_mask::insert(device_id id)
{
    bitmask |= 1llu << (uint64_t)id;
}

device_mask::iterator& device_mask::iterator::operator++()
{
    bitmask ^= (1llu << findLSB(bitmask));
    return *this;
}

device& device_mask::iterator::operator*() const
{
    return ctx->get_devices()[findLSB(bitmask)];
}

bool device_mask::iterator::operator==(const iterator& other) const
{
    return bitmask == other.bitmask;
}

bool device_mask::iterator::operator!=(const iterator& other) const
{
    return bitmask != other.bitmask;
}

device_mask::iterator device_mask::begin() const
{
    return {ctx, bitmask};
}

device_mask::iterator device_mask::end() const
{
    return {ctx, 0};
}

device_mask::iterator device_mask::cbegin() const
{
    return {ctx, bitmask};
}

device_mask::iterator device_mask::cend() const
{
    return {ctx, 0};
}

void device_mask::clear()
{
    bitmask = 0;
}

std::size_t device_mask::size() const
{
    return bitCount(bitmask);
}

context* device_mask::get_context() const
{
    return ctx;
}

device& device_mask::get_device(device_id id) const
{
    return ctx->get_devices()[id];
}

device_mask device_mask::operator-(device_mask other) const
{
    device_mask res;
    res.ctx = ctx;
    res.bitmask = bitmask & ~other.bitmask;
    return res;
}

device_mask device_mask::operator|(device_mask other) const
{
    device_mask res;
    res.ctx = ctx;
    res.bitmask = bitmask | other.bitmask;
    return res;
}

device_mask device_mask::operator&(device_mask other) const
{
    device_mask res;
    res.ctx = ctx;
    res.bitmask = bitmask & other.bitmask;
    return res;
}

device_mask device_mask::operator^(device_mask other) const
{
    device_mask res;
    res.ctx = ctx;
    res.bitmask = bitmask ^ other.bitmask;
    return res;
}

device_mask device_mask::operator~() const
{
    device_mask res;
    res.ctx = ctx;
    res.bitmask = (~bitmask) & ((1llu<<ctx->get_devices().size()) - 1llu);
    return res;
}

device_mask& device_mask::operator-=(device_mask other)
{
    return *this = *this - other;
}

device_mask& device_mask::operator|=(device_mask other)
{
    return *this = *this | other;
}

device_mask& device_mask::operator&=(device_mask other)
{
    return *this = *this & other;
}

device_mask& device_mask::operator^=(device_mask other)
{
    return *this = *this ^ other;
}

}
