#ifndef TAURAY_TRANSFORMABLE_HH
#define TAURAY_TRANSFORMABLE_HH
#include "math.hh"

// Transform caching doubles the size of transformable, but can make
// get_global_transform() significantly faster in some cases.
#define TR_TRANSFORM_CACHING

namespace tr
{

class transformable
{
public:
    transformable(transformable* parent = nullptr);

    void rotate(float angle, vec3 axis, vec3 local_origin = vec3(0));
    void rotate(vec3 axis_magnitude, vec3 local_origin = vec3(0));
    void rotate(float angle, vec2 local_origin = vec2(0));
    void rotate_local(float angle, vec3 axis, vec3 local_origin = vec3(0));

    void rotate(quat rotation);
    void set_orientation(float angle);
    void set_orientation(float angle, vec3 axis);
    void set_orientation(float pitch, float yaw, float roll = 0);
    void set_orientation(quat orientation = quat());
    quat get_orientation() const;
    vec3 get_orientation_euler() const;

    void translate(vec2 offset);
    void translate(vec3 offset);
    void translate_local(vec2 offset);
    void translate_local(vec3 offset);
    void set_position(vec2 position);
    void set_position(vec3 position = vec3(0));
    void set_depth(float depth = 0);
    vec3 get_position() const;

    void scale(float scale);
    void scale(vec2 scale);
    void scale(vec3 scale);
    void set_scaling(vec2 size);
    void set_scaling(vec3 size = vec3(1));
    vec2 get_size() const;
    vec3 get_scaling() const;

    void set_transform(const mat4& transform = mat4());
    mat4 get_transform() const;

    void set_direction(vec3 direction, vec3 forward = vec3(0,0,-1));
    vec3 get_direction(vec3 forward = vec3(0,0,-1)) const;

#ifdef TR_TRANSFORM_CACHING
    const mat4& get_global_transform() const;
    const mat4& get_global_inverse_transpose_transform() const;
#else
    mat4 get_global_transform() const;
    mat4 get_global_inverse_transpose_transform() const;
#endif

    vec3 get_global_position() const;
    quat get_global_orientation() const;
    vec3 get_global_orientation_euler() const;
    vec3 get_global_scaling() const;

    void set_global_orientation(float angle, vec3 axis);
    void set_global_orientation(float pitch, float yaw, float roll = 0);
    void set_global_orientation(vec3 euler_angles);
    void set_global_orientation(quat orientation = quat());
    void set_global_position(vec3 pos = vec3(0));
    void set_global_scaling(vec3 size = vec3(1));

    void set_parent(
        transformable* parent = nullptr,
        bool keep_transform = false
    );
    transformable* get_parent() const;

    // Once marked static, a transformable should no longer move in any way.
    // This includes its parents! So make sure that children of dynamic objects
    // are not marked as static.
    void set_static(bool s);
    bool is_static() const;

    void lookat(
        vec3 pos,
        vec3 up = vec3(0,1,0),
        vec3 forward = vec3(0,0,-1),
        float angle_limit = -1,
        vec3 lock_axis = vec3(0)
    );
    void lookat(
        const transformable* other,
        vec3 up = vec3(0,1,0),
        vec3 forward = vec3(0,0,-1),
        float angle_limit = -1,
        vec3 lock_axis = vec3(0)
    );

    void set_global_direction(vec3 direction, vec3 forward = vec3(0,0,-1));
    vec3 get_global_direction(vec3 forward = vec3(0,0,-1)) const;

    void align_to_view(
        vec3 global_view_dir,
        vec3 global_view_up_dir,
        vec3 up = vec3(0,1,0),
        vec3 lock_axis = vec3(0)
    );

#ifdef TR_TRANSFORM_CACHING
    uint16_t update_cached_transform() const;
#endif

private:
#ifdef TR_TRANSFORM_CACHING
    mutable uint16_t cached_revision;
    mutable uint16_t revision;
    mutable mat4 cached_transform;
    mutable mat4 cached_inverse_transpose_transform;
    mutable uint16_t cached_parent_revision;
#endif

    transformable* parent;
    quat orientation;
    vec3 position, scaling;
    bool static_locked;
};

}

#endif
