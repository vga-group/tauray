#ifndef TAURAY_TRACING_HH
#define TAURAY_TRACING_HH

#include "vkm.hh"
#include <set>
#include <map>
#include <deque>
#include <chrono>

namespace tr
{

struct trace_event
{
    double start_ns;
    double duration_ns;
    std::string name;
};

class context;

class tracing_record
{
public:
    enum trace_format
    {
        SIMPLE,
        TRACE_EVENT_FORMAT
    };

    tracing_record(context* ctx);
    void init(unsigned max_timestamps);
    void deinit();

    void begin_frame();
    void host_wait();
    void device_finish_frame();
    void wait_all_frames(bool print_traces, trace_format format = SIMPLE);

    int register_timer(size_t device_index, const std::string& name);
    void unregister_timer(size_t device_index, int timer_id);
    vk::QueryPool get_timestamp_pool(size_t device_index, uint32_t frame_index);

    float get_duration(size_t device_index, const std::string& name) const;
    void print_last_trace(trace_format format = SIMPLE);

private:
    struct timing_result
    {
        uint32_t frame_number = 0;
        std::vector<trace_event> host_traces;
        std::vector<std::vector<trace_event>> device_traces;
    };

    void finish_host_frame();
    const timing_result* find_latest_finished_frame() const;
    void print_simple_trace(const timing_result& res);
    void print_tef_trace(const timing_result& res);

    context* ctx;
    unsigned max_timestamps;
    uint32_t frame_counter;
    uint32_t host_finished_frame_counter;
    uint32_t device_finished_frame_counter;
    std::deque<timing_result> times;

    struct timing_data
    {
        timing_data() = default;
        timing_data(timing_data&& other) = default;
        timing_data(const timing_data& other) = delete;

        std::vector<vkm<vk::QueryPool>> timestamp_pools;

        std::set<int> available_queries;
        std::map<int, std::string> reserved_queries;
        std::vector<uint64_t> last_results;

        double device_reference_ns;
    };
    std::vector<timing_data> timing_resources;
    std::chrono::steady_clock::time_point frame_start_time;
    std::chrono::steady_clock::time_point wait_start_time;

    double host_reference_ns;
};

}

#endif

