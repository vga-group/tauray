#include "timer.hh"

namespace tr
{

timer::timer(){}

timer::timer(device_mask dev, const std::string& name)
{
    timer_id.init(dev,
        [&](device& d){
            return d.ctx->get_timing().register_timer(d.index, name);
        }
    );
}

timer::timer(timer&& other)
: timer_id(std::move(other.timer_id))
{
    other.timer_id.clear();
}

timer::~timer()
{
    for(auto[d, timer_id]: timer_id)
        d.ctx->get_timing().unregister_timer(d.index, timer_id);
}

timer& timer::operator=(timer&& other)
{
    for(auto[d, timer_id]: timer_id)
        d.ctx->get_timing().unregister_timer(d.index, timer_id);
    timer_id = std::move(other.timer_id);
    other.timer_id.clear();
    return *this;
}

void timer::begin(
    vk::CommandBuffer cb, device_id id, uint32_t frame_index, vk::PipelineStageFlagBits stage
){
    int tid = timer_id[id];
    if(tid < 0) return;
    uint32_t query_id = tid * 2u;
    vk::QueryPool pool = timer_id.get_context()->get_timing().get_timestamp_pool(id, frame_index);
    cb.resetQueryPool(pool, query_id, 2);
    cb.writeTimestamp(stage, pool, query_id);
}

void timer::end(
    vk::CommandBuffer cb, device_id id, uint32_t frame_index, vk::PipelineStageFlagBits stage
){
    int tid = timer_id[id];
    if(tid < 0) return;
    uint32_t query_id = tid * 2u;
    vk::QueryPool pool = timer_id.get_context()->get_timing().get_timestamp_pool(id, frame_index);
    cb.writeTimestamp(stage, pool, query_id+1);
}

}
