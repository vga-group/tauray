#include "timer.hh"

namespace tr
{

timer::timer()
: dev(nullptr), timer_id(-1)
{
}

timer::timer(device_data& dev, const std::string& name)
: dev(&dev)
{
    timer_id = dev.ctx->register_timer(dev.index, name);
}

timer::timer(timer&& other)
: dev(other.dev), timer_id(other.timer_id)
{
    other.dev = nullptr;
}

timer::~timer()
{
    if(dev) dev->ctx->unregister_timer(dev->index, timer_id);
}

timer& timer::operator=(timer&& other)
{
    if(dev) dev->ctx->unregister_timer(dev->index, timer_id);
    dev = other.dev;
    timer_id = other.timer_id;
    other.dev = nullptr;
    return *this;
}

void timer::begin(
    vk::CommandBuffer cb, uint32_t frame_index, vk::PipelineStageFlagBits stage
){
    if(timer_id < 0 || !dev) return;
    uint32_t query_id = timer_id * 2u;
    vk::QueryPool pool = dev->ctx->get_timestamp_pool(dev->index, frame_index);
    cb.resetQueryPool(pool, query_id, 2);
    cb.writeTimestamp(stage, pool, query_id);
}

void timer::end(
    vk::CommandBuffer cb, uint32_t frame_index, vk::PipelineStageFlagBits stage
){
    if(timer_id < 0 || !dev) return;
    uint32_t query_id = timer_id * 2u;
    vk::QueryPool pool = dev->ctx->get_timestamp_pool(dev->index, frame_index);
    cb.writeTimestamp(stage, pool, query_id+1);
}

}
