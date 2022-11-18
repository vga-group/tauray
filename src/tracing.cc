#include "tracing.hh"
#include "context.hh"
#include "json.hpp"
#include <iostream>

namespace tr
{

enum trace_format
{
    TAURAY,
    TRACE_EVENT_FORMAT
};

tracing_record::tracing_record(context* ctx)
: ctx(ctx), frame_counter(0), host_finished_frame_counter(0), device_finished_frame_counter(0)
{
}

void tracing_record::init(unsigned max_timestamps)
{
    this->max_timestamps = max_timestamps;
    if(max_timestamps)
    {
        auto& devices = ctx->get_devices();
        timing_resources.resize(devices.size());
        for(size_t i = 0; i < devices.size(); ++i)
        {
            timing_data& t = timing_resources[i];
            t.timestamp_pools.resize(MAX_FRAMES_IN_FLIGHT);
            for(size_t j = 0; j < MAX_FRAMES_IN_FLIGHT; ++j)
                t.timestamp_pools[j] = vkm(
                    devices[i],
                    devices[i].dev.createQueryPool({
                        {}, vk::QueryType::eTimestamp, max_timestamps * 2u
                    })
                );
            for(unsigned j = 0; j < max_timestamps; ++j)
                t.available_queries.insert(j);
        }

        host_reference_ns = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now().time_since_epoch()).count() * 1e9;
        for(size_t i = 0; i < devices.size(); ++i)
        {
           vk::CalibratedTimestampInfoEXT info;
           info.timeDomain = vk::TimeDomainEXT::eDevice;
           timing_resources[i].device_reference_ns = double(devices[i].dev.getCalibratedTimestampEXT(info).first)*double(devices[i].props.limits.timestampPeriod);
        }
    }
}

void tracing_record::deinit()
{
    timing_resources.clear();
}

void tracing_record::begin_frame()
{
    finish_host_frame();

    if(times.size() != 0 && times.front().frame_number + 1 < device_finished_frame_counter)
        times.pop_front();

    times.push_back({frame_counter, {}, {}});
    frame_counter++;
}

void tracing_record::host_wait()
{
    if(times.size() == 0) return;
    wait_start_time = std::chrono::steady_clock::now();
    timing_result& res = times.back();
    res.host_traces.push_back({
        std::chrono::duration_cast<std::chrono::duration<double>>(frame_start_time.time_since_epoch()).count() * 1e9 - host_reference_ns,
        std::chrono::duration_cast<std::chrono::duration<double>>(wait_start_time - frame_start_time).count() * 1e9,
        "CPU working"
    });
}

void tracing_record::device_finish_frame()
{
    if(max_timestamps == 0) return;

    timing_result* res = nullptr;
    for(auto it = times.begin(); it != times.end(); ++it)
    {
        if(it->frame_number == device_finished_frame_counter)
            res = &*it;
    }
    if(!res) return;

    uint32_t findex = res->frame_number%MAX_FRAMES_IN_FLIGHT;

    const auto& devices = ctx->get_devices();
    res->device_traces.resize(devices.size());
    for(size_t i = 0; i < res->device_traces.size(); ++i)
    {
        timing_data& t = timing_resources[i];
        std::vector<uint64_t> results(max_timestamps*2u);
        (void)devices[i].dev.getQueryPoolResults(
            t.timestamp_pools[findex],
            0,
            max_timestamps*2u,
            results.size()*sizeof(uint64_t),
            results.data(),
            0,
            vk::QueryResultFlagBits::e64
        );
        for(const auto& pair: t.reserved_queries)
        {
            res->device_traces[i].push_back(trace_event{
                double(results[pair.first*2])*double(devices[i].props.limits.timestampPeriod) - t.device_reference_ns,
                double(results[pair.first*2+1]-results[pair.first*2])*double(devices[i].props.limits.timestampPeriod),
                pair.second
            });
        }
        std::sort(
            res->device_traces[i].begin(),
            res->device_traces[i].end(),
            [](const trace_event& a, const trace_event& b){
                return a.start_ns < b.start_ns;
            }
        );
    }

    device_finished_frame_counter++;
}

void tracing_record::wait_all_frames(bool print_traces, trace_format format)
{
    if(max_timestamps == 0) return;
    ctx->sync();
    finish_host_frame();
    while(device_finished_frame_counter < frame_counter)
    {
        device_finish_frame();
        if(print_traces)
            print_last_trace(format);
    }
}

int tracing_record::register_timer(size_t device_index, const std::string& name)
{
    if(max_timestamps == 0) return -1;
    timing_data& t = timing_resources[device_index];
    if(t.available_queries.size() == 0)
        throw std::runtime_error(
            "Not enough timer queries in pool! Increase max_timestamps!"
        );
    auto it = t.available_queries.begin();
    int ret = *it;
    t.reserved_queries[ret] = name;
    t.available_queries.erase(it);
    return ret;
}

void tracing_record::unregister_timer(size_t device_index, int timer_id)
{
    if(max_timestamps == 0) return;
    timing_data& t = timing_resources[device_index];
    t.reserved_queries.erase(timer_id);
    t.available_queries.insert(timer_id);
}

vk::QueryPool tracing_record::get_timestamp_pool(size_t device_index, uint32_t frame_index)
{
    if(max_timestamps == 0) return {};
    return timing_resources[device_index].timestamp_pools[frame_index];
}

float tracing_record::get_duration(size_t device_index, const std::string& name) const
{
    const timing_result* res = find_latest_finished_frame();
    if(!res) return 0.0f;
    float total_time = 0.0f;
    for(const trace_event& ti: res->device_traces[device_index])
    {
        if(ti.name.compare(0, name.length(), name) == 0)
            total_time += ti.duration_ns;
    }
    return total_time;
}

void tracing_record::print_last_trace(trace_format format)
{
    const timing_result* res = find_latest_finished_frame();
    if(!res) return;

    switch(format)
    {
    default:
    case SIMPLE:
        print_simple_trace(*res);
        break;
    case TRACE_EVENT_FORMAT:
        print_tef_trace(*res);
        break;
    }
}

void tracing_record::finish_host_frame()
{
    auto time_now = std::chrono::steady_clock::now();
    if(frame_counter > host_finished_frame_counter)
    {
        timing_result& res = times.back();
        host_finished_frame_counter++;

        res.host_traces.push_back({
            std::chrono::duration_cast<std::chrono::duration<double>>(wait_start_time.time_since_epoch()).count() * 1e9 - host_reference_ns,
            std::chrono::duration_cast<std::chrono::duration<double>>(time_now - wait_start_time).count() * 1e9,
            res.host_traces.size() == 0 ? "CPU working" : "CPU waiting"
        });
    }
    frame_start_time = time_now;
    wait_start_time = time_now;
}

const tracing_record::timing_result* tracing_record::find_latest_finished_frame() const
{
    for(auto it = times.rbegin(); it != times.rend(); ++it)
    {
        if(it->frame_number < host_finished_frame_counter && it->frame_number < device_finished_frame_counter)
            return &*it;
    }
    return nullptr;
}

void tracing_record::print_simple_trace(const timing_result& res)
{
    std::cout << "FRAME " << res.frame_number << ":\n";
    for(size_t i = 0; i < res.device_traces.size(); ++i)
    {
        const std::vector<trace_event>& times = res.device_traces[i];

        std::cout << "\tDEVICE " << i << ": ";
        if(times.size() == 0)
            std::cout << "\n";
        else
        {
            double delta_ns = times.back().start_ns + times.back().duration_ns - times.front().start_ns;
            std::cout << delta_ns/1e6 << " ms\n";
        }

        for(const trace_event& t: times)
        {
            std::cout << "\t\t[" << t.name << "] " << t.duration_ns/1e6 << " ms"
                << "\n";
        }
    }

    std::cout << "\tHOST: ";
    if(res.host_traces.size() == 0)
        std::cout << "\n";
    else
    {
        double delta_ns = res.host_traces.back().start_ns + res.host_traces.back().duration_ns - res.host_traces.front().start_ns;
        std::cout << delta_ns/1e6 << " ms\n";
    }
    for(const trace_event& t: res.host_traces)
    {
        std::cout << "\t\t[" << t.name << "] " << t.duration_ns/1e6 << " ms"
            << "\n";
    }
}

void tracing_record::print_tef_trace(const timing_result& res)
{
    static bool first_call = true;
    if(first_call)
    {
        std::cout << "[";
        first_call = false;
    }
    for(const trace_event& t: res.host_traces)
    {
        nlohmann::json output = {
            {"pid",0},
            {"tid",0},
            {"ts",int64_t(t.start_ns*1e-3)},
            {"dur",int64_t(t.duration_ns*1e-3)},
            {"ph", "X"},
            {"name", t.name},
            {"args", {"ms", t.duration_ns*1e-6}}
        };
        std::cout << output.dump(-1, '\t') << ",\n";
    }
    for(size_t i = 0; i < res.device_traces.size(); ++i)
    {
        for(const trace_event& t: res.device_traces[i])
        {
            nlohmann::json output = {
                {"pid",1},
                {"tid",i},
                {"ts",int64_t(t.start_ns*1e-3)},
                {"dur",int64_t(t.duration_ns*1e-3)},
                {"ph", "X"},
                {"name", t.name},
                {"args", {"ms", t.duration_ns*1e-6}}
            };
            std::cout << output.dump(-1, '\t') << ",\n";
        }
    }
}

}

