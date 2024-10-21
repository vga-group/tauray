// Adapted from CC0 code from https://gist.github.com/juliusikkala/fc9c082d33488bdd3b03285463b998f3
#ifndef LOG_HH
#define LOG_HH
#include "math.hh"
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>

// Thanks microsoft...
#undef ERROR

#ifndef TAURAY_PROJECT_ROOT_PATH_SIZE
#define TAURAY_PROJECT_ROOT_PATH_SIZE 0
#endif

#define __FILENAME__ ((const char*)((uintptr_t)__FILE__ + TAURAY_PROJECT_ROOT_PATH_SIZE))

#ifdef PROJECT_DEBUG
#define TR_DBG(...) tr::log_message(tr::log_type::DEBUG __LINE__, __FILENAME__, __VA_ARGS__)
#else
#define TR_DBG(...)
#endif

#define TR_LOG(...) tr::log_message(tr::log_type::GENERAL, __LINE__, __FILENAME__, __VA_ARGS__)
#define TR_ERR(...) tr::log_message(tr::log_type::ERROR, __LINE__, __FILENAME__, __VA_ARGS__)
#define TR_WARN(...) tr::log_message(tr::log_type::WARNING, __LINE__, __FILENAME__, __VA_ARGS__)
#define TR_TIME(...) tr::log_message(tr::log_type::TIMING, __LINE__, __FILENAME__, __VA_ARGS__)

namespace tr
{

std::chrono::system_clock::time_point get_initial_time();

template<typename T>
std::string to_string(const T& t) {
    if constexpr(std::is_pointer_v<T>)
    {
        std::stringstream stream;
        stream << typeid(std::decay_t<std::remove_pointer_t<T>>).name() << "*(" << std::hex << (uintptr_t)t << ")";
        std::string result(stream.str());
        return stream.str();
    }
    else return std::to_string(t);
}
inline std::string to_string(const char* t) { return t; }
inline std::string to_string(const std::string& t) { return t; }
inline std::string to_string(const std::string_view& t) { return std::string(t.begin(), t.end()); }

template<glm::length_t L, typename T, glm::qualifier Q>
inline std::string to_string(const glm::vec<L, T, Q>& t) { return glm::to_string(t); }

template<glm::length_t C, glm::length_t R, typename T, glm::qualifier Q>
inline std::string to_string(const glm::mat<C, R, T, Q>& t) { return glm::to_string(t); }

template<typename T>
std::string make_string(const T& t) { return to_string(t); }

template<typename T, typename... Args>
std::string make_string(
    const T& t,
    const Args&... rest
){ return to_string(t) + make_string(rest...); }

enum class log_type: uint32_t
{
    GENERAL = 0,
    ERROR,
    WARNING,
    DEBUG,
    TIMING
};

extern bool enabled_log_types[5];
extern std::ostream* log_output_streams[5];

void apply_color(log_type type, std::ostream& os);

template<typename... Args>
void log_message(
    log_type type,
    int line,
    const char* file,
    const Args&... rest
){
    if(enabled_log_types[(uint32_t)type])
    {
        std::chrono::system_clock::time_point now =
            std::chrono::system_clock::now();
        std::chrono::system_clock::duration d = now - get_initial_time();

        std::ostream& o = *log_output_streams[(uint32_t)type];

        apply_color(type, o);
        if(type != log_type::TIMING)
        {
            o << "[" << std::fixed << std::setprecision(3) <<
                std::chrono::duration_cast<
                    std::chrono::milliseconds
                >(d).count()/1000.0
                << "](" << file << ":" << line << ") ";
        }
        o << make_string(rest...) << std::endl;
        apply_color(log_type::GENERAL, o);
    }
}

}

#endif
