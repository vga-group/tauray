#ifndef TAURAY_ANIMATION_TCC
#define TAURAY_ANIMATION_TCC
#include "animation.hh"
#include <algorithm>

namespace tr
{

template<typename T>
struct numeric_mixer
{
    T operator()(const T& begin, const T& end, double t) const
    {
        return begin * (1.0 - t) + end * t;
    }
};

template<glm::length_t L, typename T, glm::qualifier Q>
struct numeric_mixer<glm::vec<L, T, Q>>
{
    using V = glm::vec<L, T, Q>;
    V operator()(const V& begin, const V& end, double t) const
    {
        return begin * T(1.0 - t) + end * T(t);
    }
};

template<typename T, glm::qualifier Q>
struct numeric_mixer<glm::qua<T, Q>>
{
    using V = glm::qua<T, Q>;
    V operator()(const V& begin, const V& end, double t) const
    {
        return glm::slerp(begin, end, T(t));
    }
};

template<typename T>
T animation::interpolate(
    time_ticks time,
    const std::vector<sample<T>>& data,
    interpolation interp
) const
{
    auto it = std::upper_bound(
        data.begin(), data.end(), time,
        [](time_ticks time, const sample<T>& s){ return time < s.timestamp; }
    );
    if(it == data.end()) return data.back().data;
    if(it == data.begin()) return data.front().data;

    auto prev = it-1;
    float frame_ticks = it->timestamp-prev->timestamp;
    float ratio = (time-prev->timestamp)/frame_ticks;
    switch(interp)
    {
    default:
    case LINEAR:
        return numeric_mixer<T>()(prev->data, it->data, ratio);
    case STEP:
        return prev->data;
    case CUBICSPLINE:
        {
            // Scale factor has to use seconds unfortunately.
            float scale = frame_ticks * 0.000001f;
            return cubic_spline(
                prev->data,
                prev->out_tangent*scale,
                it->data,
                it->in_tangent*scale,
                ratio
            );
        }
    }
}

template<typename Derived>
animation_controller<Derived>::animation_controller()
: timer(0), loop_time(0), paused(false)
{
}

template<typename Derived>
animation_controller<Derived>& animation_controller<Derived>::queue(
    const std::string& name, bool loop
){
    animation_queue.push_back({name, loop});
    if(animation_queue.size() == 1)
    {
        timer = 0;
        loop_time = static_cast<Derived*>(this)->set_animation(name, false);
    }
    return *this;
}

template<typename Derived>
void animation_controller<Derived>::play(
    const std::string& name,
    bool loop,
    bool use_fallback
){
    timer = 0;
    loop_time = static_cast<Derived*>(this)->set_animation(name, use_fallback);

    if(loop_time)
    {
        animation_queue.clear();
        animation_queue.push_back({name, loop});
    }
}

template<typename Derived>
void animation_controller<Derived>::pause(bool paused)
{
    this->paused = paused;
}

template<typename Derived>
void animation_controller<Derived>::restart()
{
    timer = 0;
}

template<typename Derived>
bool animation_controller<Derived>::is_playing() const
{
    return !animation_queue.empty() && !paused;
}

template<typename Derived>
bool animation_controller<Derived>::is_paused() const
{
    return this->paused;
}

template<typename Derived>
void animation_controller<Derived>::finish()
{
    if(animation_queue.size() != 0)
    {
        animation_queue.resize(1);
        animation_queue.front().loop = false;
    }
}

template<typename Derived>
void animation_controller<Derived>::stop()
{
    animation_queue.clear();
    timer = 0;
    loop_time = 0;
}

template<typename Derived>
const std::string&
animation_controller<Derived>::get_playing_animation_name() const
{
    static const std::string empty_dummy("");
    if(animation_queue.size() == 0) return empty_dummy;
    return animation_queue.front().name;
}

template<typename Derived>
time_ticks animation_controller<Derived>::get_animation_time() const
{
    return timer;
}

template<typename Derived>
void animation_controller<Derived>::update(time_ticks dt)
{
    if(!is_playing()) return;

    timer += dt;

    animation_step& cur_step = animation_queue.front();

    // If we have a waiting animation, check if the timer rolled over the
    // looping point.
    if(animation_queue.size() > 1)
    {
        if(timer >= loop_time)
        {
            timer -= loop_time;
            animation_queue.erase(animation_queue.begin());
            loop_time = static_cast<Derived*>(this)->set_animation(
                animation_queue.front().name, false
            );
        }
    }
    // If there's nothing waiting, then keep looping.
    else if(cur_step.loop)
        timer %= loop_time;
    // If we're past the end of a non-looping animation, stop animating.
    else if(timer >= loop_time)
    {
        animation_queue.erase(animation_queue.begin());
        loop_time = 0;
        timer = 0;
        return;
    }

    static_cast<Derived*>(this)->apply_animation(timer);
}

}

#endif
