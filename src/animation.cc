#include "animation.hh"

namespace tr
{

animation::animation()
:   loop_time(0), position_interpolation(LINEAR),
    scaling_interpolation(LINEAR), orientation_interpolation(LINEAR)
{
}

void animation::set_position(
    interpolation position_interpolation,
    std::vector<sample<vec3>>&& position
){
    this->position_interpolation = position_interpolation;
    this->position = std::move(position);
    determine_loop_time();
}

void animation::set_scaling(
    interpolation scaling_interpolation,
    std::vector<sample<vec3>>&& scaling
){
    this->scaling_interpolation = scaling_interpolation;
    this->scaling = std::move(scaling);
    determine_loop_time();
}

void animation::set_orientation(
    interpolation orientation_interpolation,
    std::vector<sample<quat>>&& orientation
){
    this->orientation_interpolation = orientation_interpolation;
    this->orientation = std::move(orientation);
    determine_loop_time();
}

void animation::apply(transformable_node& node, time_ticks time) const
{
    if(position.size())
        node.set_position(interpolate(time, position, position_interpolation));
    if(scaling.size())
        node.set_scaling(interpolate(time, scaling, scaling_interpolation));
    if(orientation.size())
    {
        quat o = interpolate(time, orientation, orientation_interpolation);
        if(orientation_interpolation == CUBICSPLINE)
            o = normalize(o);
        node.set_orientation(o);
    }
}

time_ticks animation::get_loop_time() const
{
    return loop_time;
}

void animation::determine_loop_time()
{
    loop_time = 0;
    if(position.size())
        loop_time = std::max(position.back().timestamp, loop_time);
    if(scaling.size())
        loop_time = std::max(scaling.back().timestamp, loop_time);
    if(orientation.size())
        loop_time = std::max(orientation.back().timestamp, loop_time);
}

animated_node::animated_node(
    transformable_node* parent,
    const animation_pool* pool
): transformable_node(parent), pool(pool), cur_anim(nullptr)
{
}

void animated_node::set_animation_pool(const animation_pool* pool)
{
    if(pool != this->pool)
    {
        this->pool = pool;
        cur_anim = nullptr;
    }
}

const animation_pool* animated_node::get_animation_pool() const
{
    return pool;
}

time_ticks animated_node::set_animation(
    const std::string& name,
    bool use_fallback
){
    if(pool)
    {
        auto it = pool->find(name);
        if(it != pool->end())
        {
            cur_anim = &it->second;
            return cur_anim->get_loop_time();
        }
        else if(use_fallback && pool->size() != 0)
        {
            cur_anim = &pool->begin()->second;
            return cur_anim->get_loop_time();
        }
    }
    cur_anim = nullptr;
    return 0;
}

void animated_node::apply_animation(time_ticks time)
{
    if(cur_anim) cur_anim->apply(*this, time);
}

}
