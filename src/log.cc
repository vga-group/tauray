#include "log.hh"

namespace tr
{

bool enabled_log_types[5] = {true, true, true, true, true};
std::ostream* log_output_streams[5] = {
    &std::cout,
    &std::cerr,
    &std::cerr,
    &std::cout,
    &std::cout
};

void apply_color(log_type type, std::ostream& os)
{
#ifdef __unix__
    if(&os != &std::cout && &os != &std::cerr)
        return;

    switch(type)
    {
    case log_type::GENERAL:
        os << "\x1b[0;39m";
        break;
    case log_type::ERROR:
        os << "\x1b[0;31m";
        break;
    case log_type::WARNING:
        os << "\x1b[0;33m";
        break;
    case log_type::DEBUG:
        os << "\x1b[0;32m";
        break;
    case log_type::TIMING:
        os << "\x1b[0;94m";
        break;
    }
#endif
}

std::chrono::system_clock::time_point get_initial_time()
{
    static std::chrono::system_clock::time_point initial =
        std::chrono::system_clock::now();
    return initial;
}

}
