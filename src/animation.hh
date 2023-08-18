#ifndef TAURAY_ANIMATION_HH
#define TAURAY_ANIMATION_HH
#include "transformable.hh"
#include <map>

namespace tr
{

using time_ticks = int64_t;

class animation
{
public:
    template<typename T>
    struct sample
    {
        time_ticks timestamp;
        T data;
        // These are only used if the interpolation is CUBICSPLINE.
        T in_tangent;
        T out_tangent;
    };

    enum interpolation
    {
        LINEAR = 0,
        STEP,
        CUBICSPLINE
    };

    animation();

    void set_position(
        interpolation position_interpolation,
        std::vector<sample<vec3>>&& position
    );

    void set_scaling(
        interpolation scaling_interpolation,
        std::vector<sample<vec3>>&& scaling
    );

    void set_orientation(
        interpolation orientation_interpolation,
        std::vector<sample<quat>>&& orientation
    );

    void apply(transformable& node, time_ticks time) const;
    time_ticks get_loop_time() const;

private:
    void determine_loop_time();

    template<typename T>
    T interpolate(
        time_ticks time,
        const std::vector<sample<T>>& data,
        interpolation interp
    ) const;

    time_ticks loop_time;
    interpolation position_interpolation;
    std::vector<sample<vec3>> position;
    interpolation scaling_interpolation;
    std::vector<sample<vec3>> scaling;
    interpolation orientation_interpolation;
    std::vector<sample<quat>> orientation;
};

// std::map used for alphabetical order.
using animation_pool = std::map<std::string /*name*/, animation>;

// You can use this to provide the animation functions to your class, as long as
// you implement the following member functions:
//   time_ticks set_animation(const std::string& name);
//   void apply_animation(time_ticks time);
template<typename Derived>
class animation_controller
{
public:
    animation_controller();

    // Starts playing the queued animation at the next loop point, or
    // immediately if there are no playing animations. Returns a reference to
    // this object for chaining purposes.
    animation_controller& queue(const std::string& name, bool loop = false);
    void play(
        const std::string& name,
        bool loop = false,
        bool use_fallback = false
    );
    void pause(bool paused = true);
    // Only restarts the current animation step in the queue!
    void restart();
    // The animation can be unpaused and still not play simply when there is
    // no animation in the queue left to play.
    bool is_playing() const;
    bool is_paused() const;
    // Drops queued animations and ends the looping of the current animation.
    void finish();
    // Drops queued animations and instantly stops current animation as well.
    void stop();
    const std::string& get_playing_animation_name() const;
    time_ticks get_animation_time() const;

    void update(time_ticks dt);

private:
    struct animation_step
    {
        std::string name;
        bool loop;
    };
    std::vector<animation_step> animation_queue;
    time_ticks timer;
    time_ticks loop_time;
    bool paused;
};

class animated_node
: public animation_controller<animated_node>, public transformable
{
friend class animation_controller<animated_node>;
public:
    animated_node(
        transformable* parent = nullptr,
        const animation_pool* pool = nullptr
    );

    void set_animation_pool(const animation_pool* pool);
    const animation_pool* get_animation_pool() const;

protected:
    time_ticks set_animation(const std::string& name, bool use_fallback);
    void apply_animation(time_ticks time);

private:
    const animation_pool* pool;
    const animation* cur_anim;
};

}

#include "animation.tcc"
#endif
