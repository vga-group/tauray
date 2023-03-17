#include "log.hh"

namespace tr
{

bool enabled_log_types[4] = {true, true, true, true};
std::ostream* log_output_streams[4] = {
    &std::cout,
    &std::cerr,
    &std::cout,
    &std::cout
};

std::chrono::system_clock::time_point get_initial_time()
{
    static std::chrono::system_clock::time_point initial =
        std::chrono::system_clock::now();
    return initial;
}

}
