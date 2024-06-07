#include "camera.hh"
#include "json.hpp"
#include <fstream>

namespace
{
using namespace tr;

nlohmann::json matrix_to_json(const mat4& m)
{
    return {
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3]
    };
}

}

namespace tr
{

camera::camera()
: jitter_index(0)
{
    perspective(90.0f, 1.0f, 0.1f, 100.0f);
}

void camera::copy_projection(const camera& other)
{
    type = other.type;
    pd = other.pd;
}

void camera::perspective(float fov, float aspect, float near, float far)
{
    type = PERSPECTIVE;
    pd.perspective = {{}, vec2(0), fov, aspect, near, far, vec4(1,0,0,0)};
    refresh();
}

void camera::ortho(
    float left, float right, float bottom, float top,
    float near, float far
){
    type = ORTHOGRAPHIC;
    pd.orthographic = {{}, left, right, bottom, top, near, far};
    refresh();
}

void camera::equirectangular(float fov_x, float fov_y)
{
    type = EQUIRECTANGULAR;
    pd.equirectangular.fov = vec2(fov_x, fov_y);
    refresh();
}

camera::projection_type camera::get_projection_type() const
{
    return type;
}

mat4 camera::get_projection_matrix() const
{
    switch(type)
    {
    case PERSPECTIVE:
        return pd.perspective.projection;
    case ORTHOGRAPHIC:
        return pd.orthographic.projection;
    default:
        throw std::runtime_error(
            "This camera projection does not support matrix representation. It "
            "cannot be used in rasterization-based pipelines."
        );
    }
}

void camera::set_near(float near)
{
    switch(type)
    {
    case PERSPECTIVE:
        pd.perspective.near = near;
        break;
    case ORTHOGRAPHIC:
        pd.orthographic.near = near;
        break;
    default:
        break;
    }
    refresh();
}

void camera::set_far(float far)
{
    switch(type)
    {
    case PERSPECTIVE:
        pd.perspective.far = far;
        break;
    case ORTHOGRAPHIC:
        pd.orthographic.far = far;
        break;
    default:
        break;
    }
    refresh();
}

float camera::get_near() const
{
    switch(type)
    {
    case PERSPECTIVE:
        return pd.perspective.near;
    case ORTHOGRAPHIC:
        return pd.orthographic.near;
    default:
        return 0.0f;
    }
}

float camera::get_far() const
{
    switch(type)
    {
    case PERSPECTIVE:
        return pd.perspective.far;
    case ORTHOGRAPHIC:
        return pd.orthographic.far;
    default:
        return INFINITY;
    }
}

vec2 camera::get_range() const
{
    return vec2(get_near(), get_far());
}

void camera::set_focus(
    float f_stop,
    float focus_distance,
    int aperture_sides,
    float aperture_angle,
    float sensor_size
){
    switch(type)
    {
    case PERSPECTIVE:
        pd.perspective.focus = vec4(
            focus_distance,
            f_stop == 0.0f ? 0.0f : sensor_size / f_stop,
            glm::radians(aperture_angle),
            aperture_sides
        );
        break;
    default:
        // No DoF for other projection types yet!
        break;
    }
}

void camera::set_aspect(float aspect)
{
    switch(type)
    {
    case PERSPECTIVE:
        pd.perspective.aspect = aspect;
        break;
    case ORTHOGRAPHIC:
        {
            float x_range = pd.orthographic.right - pd.orthographic.left;
            float y_range = x_range / aspect;
            float y_center = (pd.orthographic.bottom + pd.orthographic.top)*0.5f;
            pd.orthographic.bottom = y_center - y_range*0.5f;
            pd.orthographic.top = y_center + y_range*0.5f;
        }
        break;
    default:
        break;
    }
    refresh();
}

void camera::set_fov(float fov)
{
    switch(type)
    {
    case PERSPECTIVE:
        pd.perspective.fov = fov;
        break;
    case EQUIRECTANGULAR:
        pd.equirectangular.fov.x = fov;
        break;
    default:
        break;
    }
    refresh();
}

void camera::set_fov(
    float fov_left, float fov_right,
    float fov_up, float fov_down
){
    if(type == PERSPECTIVE)
    {
        float right = tan(glm::radians(fov_right));
        float left = tan(glm::radians(fov_left));
        float up = tan(glm::radians(fov_up));
        float down = tan(glm::radians(fov_down));

        float w = right - left;
        float h = up - down;

        pd.perspective.fov_offset = vec2(
            (right + left) / w,
            (up + down) / h
        );

        pd.perspective.fov = fov_up-fov_down;
        pd.perspective.aspect = w/h;
    }
    refresh();
}

void camera::set_pan(vec2 offset)
{
    if(type == PERSPECTIVE)
    {
        pd.perspective.fov_offset = offset;
    }
    refresh();
}

float camera::get_vfov() const
{
    switch(type)
    {
    case PERSPECTIVE:
        return pd.perspective.fov;
    case EQUIRECTANGULAR:
        return pd.equirectangular.fov.y;
    default:
        return 0;
    }
}

float camera::get_hfov() const
{
    switch(type)
    {
    case PERSPECTIVE:
        return 2.0f * glm::degrees(atan(pd.perspective.aspect * tan(glm::radians(pd.perspective.fov * 0.5f))));
    case EQUIRECTANGULAR:
        return pd.equirectangular.fov.x;
    default:
        return 0;
    }
}

ray camera::get_view_ray(vec2 uv, float near_mul) const
{
    ray r;
    switch(type)
    {
    case PERSPECTIVE:
        {
            vec2 projection_info = get_projection_info();
            vec3 dir = vec3((0.5f-uv) * projection_info, 1.0f);
            float use_near = -near_mul * pd.perspective.near;
            r.o = dir * -use_near;
            if(pd.perspective.far == INFINITY)
                r.dir = normalize(dir);
            else
                r.dir = (dir * -pd.perspective.far) - r.o;
        }
        break;
    case ORTHOGRAPHIC:
        {
            auto [_, left, right, bottom, top, near, far] =
                pd.orthographic;
            float use_near = -near_mul * near;
            r.o = vec3(
                mix(left, right, uv.x),
                mix(bottom, top, uv.y),
                -use_near
            );
            r.dir = vec3(0, 0, use_near-far);
        }
        break;
    case EQUIRECTANGULAR:
        {
            r.o = vec3(0);
            vec2 local_uv = (uv * 2.0f - 1.0f) *
                glm::radians(pd.equirectangular.fov) * 0.5f;
            vec2 c = cos(local_uv);
            vec2 s = sin(local_uv);
            r.dir = normalize(vec3(s.x*c.y, s.y, -c.x*c.y));
        }
        break;
    }
    return r;
}

ray camera::get_global_view_ray(transformable& t, vec2 uv) const
{
    return t.get_global_transform() * get_view_ray(uv);
}

mat4 camera::get_view_projection(transformable& t) const
{
    return get_projection_matrix() * glm::inverse(t.get_global_transform());
}

mat4 camera::get_view_projection(const mat4& global_transform) const
{
    return get_projection_matrix() * glm::inverse(global_transform);
}

vec4 camera::get_projection_info() const
{
    switch(type)
    {
    case PERSPECTIVE:
        {
            float rad_fov = glm::radians(pd.perspective.fov);
            vec4 proj_info;
            proj_info.w = 2*tan(rad_fov/2.0f);
            proj_info.z = proj_info.w * pd.perspective.aspect;
            if(pd.perspective.far == INFINITY)
            {
                proj_info.x = -pd.perspective.near;
                proj_info.y = -1;
            }
            else
            {

                float near = pd.perspective.near;
                float far = pd.perspective.far;

                proj_info.x = near * far / (near-far);
                proj_info.y = (near + far) / (near-far);
            }
            return proj_info;
        }
    case ORTHOGRAPHIC:
        {
            return vec4(
                pd.orthographic.far - pd.orthographic.near,
                -pd.orthographic.far,
                pd.orthographic.right - pd.orthographic.left,
                pd.orthographic.top - pd.orthographic.bottom
            );
        }
    default:
        return vec4(0);
    }
}

void camera::refresh()
{
    switch(type)
    {
    case PERSPECTIVE:
        {
            auto& [projection, fov_offset, fov, aspect, near, far, focus] = pd.perspective;
            float rad_fov = glm::radians(fov);
            if(far == INFINITY)
                projection = glm::infinitePerspective(rad_fov, aspect, near);
            else
                projection = glm::perspective(rad_fov, aspect, near, far);
            projection[2][0] = fov_offset.x;
            projection[2][1] = fov_offset.y;
            if(jitter_sequence.size())
            {
                projection[2][0] += jitter_sequence[jitter_index].x;
                projection[2][1] += jitter_sequence[jitter_index].y;
            }
        }
        break;
    case ORTHOGRAPHIC:
        {
            auto& [projection, left, right, bottom, top, near, far] =
                pd.orthographic;
            projection = glm::ortho(left, right, bottom, top, near, far);
        }
        break;
    default:
        break;
    }
}

// These structs must match the camera_data_buffers in shader/camera.glsl
struct matrix_camera_data_buffer
{
    pmat4 view;
    pmat4 view_inverse;
    pmat4 view_proj;
    pmat4 proj_inverse;
    pvec4 origin;
    pvec4 dof_params;
    pvec4 projection_info;
    pvec4 pan;
};

struct equirectangular_camera_data_buffer
{
    pmat4 view;
    pmat4 view_inverse;
    pvec4 origin;
    pvec2 fov;
};

size_t camera::get_projection_type_uniform_buffer_size(projection_type type)
{
    switch(type)
    {
    case PERSPECTIVE:
    case ORTHOGRAPHIC:
        return sizeof(matrix_camera_data_buffer);
    case EQUIRECTANGULAR:
        return sizeof(equirectangular_camera_data_buffer);
    }
    assert(false);
    return 0;
}

void camera::write_uniform_buffer(transformable& self, void* data) const
{
    switch(type)
    {
    case PERSPECTIVE:
    case ORTHOGRAPHIC:
        {
            auto& buf = *static_cast<matrix_camera_data_buffer*>(data);
            mat4 inv_view = self.get_global_transform();
            mat4 view = inverse(inv_view);
            vec4 origin = inv_view * vec4(0,0,0,1);
            mat4 projection = get_projection_matrix();
            mat4 inv_projection = inverse(projection);

            buf.view = view;
            buf.view_inverse = inv_view;
            buf.view_proj = projection * view;
            buf.proj_inverse = inv_projection;
            buf.origin = origin;
            buf.dof_params = type == PERSPECTIVE ? pd.perspective.focus : vec4(0);
            buf.projection_info = get_projection_info();
            buf.pan = type == PERSPECTIVE ? vec4(pd.perspective.fov_offset, 0, 0) : vec4(0);

            if(jitter_sequence.size())
            {
                vec2 cur_jitter = jitter_sequence[jitter_index];
                buf.pan.x += cur_jitter.x;
                buf.pan.y += cur_jitter.y;
                buf.pan.z += cur_jitter.x;
                buf.pan.w += cur_jitter.y;
            }
        }
        break;
    case EQUIRECTANGULAR:
        {
            auto& buf = *static_cast<equirectangular_camera_data_buffer*>(data);
            mat4 inv_view = self.get_global_transform();
            mat4 view = inverse(inv_view);
            vec4 origin = inv_view * vec4(0,0,0,1);

            buf.view = view;
            buf.view_inverse = inv_view;
            buf.origin = origin;
            buf.fov = glm::radians(pd.equirectangular.fov) * 0.5f;
        }
        break;
    }
}

void camera::set_jitter(const std::vector<vec2>& jitter_sequence)
{
    this->jitter_sequence = jitter_sequence;
    jitter_index = 0;
}

void camera::step_jitter()
{
    if(jitter_sequence.size())
    {
        jitter_index = (jitter_index+1)%jitter_sequence.size();
        refresh();
    }
}

vec2 camera::get_jitter() const
{
    return jitter_sequence.size() == 0 ? vec2(0) : jitter_sequence[jitter_index];
}

camera_log::camera_log(transformable* cam_transform, camera* cam)
: cam_transform(cam_transform), cam(cam)
{
}

camera_log::~camera_log()
{
}

void camera_log::frame(time_ticks dt)
{
    frames.push_back({dt, inverse(cam_transform->get_global_transform())});
}

void camera_log::write(const std::string& path)
{
    nlohmann::json out = nlohmann::json::object();

    switch(cam->type)
    {
    case camera::PERSPECTIVE:
    case camera::ORTHOGRAPHIC:
        out["projection"] = matrix_to_json(cam->get_projection_matrix());
        break;
    case camera::EQUIRECTANGULAR:
        out["projection"] = {
            cam->pd.equirectangular.fov.x,
            cam->pd.equirectangular.fov.y
        };
        break;
    default:
        break;
    }

    for(frame_data fd: frames)
    {
        out["frames"].push_back({
            {"delta", fd.dt/1000000.0},
            {"view", matrix_to_json(fd.view)}
        });
    }

    std::ofstream out_file(path, std::ios::trunc);
    out_file << out.dump(2);
    out_file.close();
}

}
