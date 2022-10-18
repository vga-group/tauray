#ifndef TAURAY_CAMERA_HH
#define TAURAY_CAMERA_HH
#include "transformable.hh"
#include "animation.hh"

class json;
namespace tr
{

class camera: public animated_node
{
public:
    enum projection_type
    {
        PERSPECTIVE = 0,
        ORTHOGRAPHIC = 1,
        EQUIRECTANGULAR = 2
    };

    explicit camera(transformable_node* parent = nullptr);

    void copy_projection(const camera& other);

    void perspective(float fov, float aspect, float near, float far = INFINITY);
    void ortho(float left, float right, float bottom, float top, float near, float far);
    void equirectangular(float fov_x, float fov_y);

    projection_type get_projection_type() const;
    mat4 get_projection_matrix() const;

    vec3 get_global_view_direction(vec3 local_view = vec3(0,0,-1)) const;

    void set_near(float near);
    void set_far(float far);
    float get_near() const;
    float get_far() const;
    vec2 get_range() const;

    void set_focus(
        float f_stop,
        float focus_distance,
        int aperture_sides, // 0 == circular, 1 and 2 are invalid, rest are regular polygons
        float aperture_angle,
        float sensor_size
    );

    void set_aspect(float aspect);
    void set_fov(float fov);
    // This also sets the aspect ratio and an asymmetric image-space offset
    void set_fov(float fov_left, float fov_right, float fov_up, float fov_down);
    void set_pan(vec2 offset);

    float get_vfov() const;
    float get_hfov() const;

    ray get_view_ray(vec2 uv, float near_mul = 1.0f) const;
    ray get_global_view_ray(vec2 uv = vec2(0.5)) const;

    // This function may throw, if the current projection does not have a matrix
    // representation!
    mat4 get_view_projection() const;

    // These two should only be used in rasterization-related things and as
    // such, they expect the projection type to be matrix-based.
    vec3 get_clip_info() const;
    vec2 get_projection_info() const;

    static size_t get_projection_type_uniform_buffer_size(projection_type type);
    void write_uniform_buffer(void* data) const;

    json serialize_projection() const;

    void set_jitter(const std::vector<vec2>& jitter_sequence);
    void step_jitter();
    vec2 get_jitter() const;

private:
    void refresh();

    projection_type type;
    union
    {
        struct
        {
            mat4 projection;
            vec2 fov_offset;
            float fov, aspect, near, far;
            vec4 focus;
        } perspective;

        struct
        {
            mat4 projection;
            float left, right, bottom, top, near, far;
        } orthographic;

        struct
        {
            vec2 fov;
        } equirectangular;
    } pd;
    std::vector<vec2> jitter_sequence;
    unsigned jitter_index;
};

class camera_log
{
public:
    camera_log(camera* cam);
    ~camera_log();

    void frame(time_ticks dt);
    void write(const std::string& path);

private:
    camera* cam;
    struct frame_data
    {
        time_ticks dt;
        mat4 view;
    };
    std::vector<frame_data> frames;
};

}

#endif
