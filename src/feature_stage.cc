#include "feature_stage.hh"
#include "scene.hh"
#include "environment_map.hh"

namespace
{
using namespace tr;

namespace feature
{
    shader_sources load_sources(feature_stage::options opt)
    {
        std::string feature;
        switch(opt.feat)
        {
        default:
        case feature_stage::ALBEDO:
            feature = "mat.albedo";
            break;
        case feature_stage::WORLD_NORMAL:
            feature = "vec4(v.mapped_normal, 1)";
            break;
        case feature_stage::VIEW_NORMAL:
            feature = "vec4((cam.view * vec4(v.mapped_normal, 0)).xyz, 1)";
            break;
        case feature_stage::WORLD_POS:
            feature = "vec4(v.pos, 1)";
            break;
        case feature_stage::VIEW_POS:
            feature = "cam.view * vec4(v.pos, 1)";
            break;
        case feature_stage::DISTANCE:
            feature = "vec4(vec3(gl_HitTEXT), 1)";
            break;
        case feature_stage::WORLD_MOTION:
            feature = "vec4(v.pos-v.prev_pos, 1)";
            break;
        case feature_stage::VIEW_MOTION:
            feature = "vec4((cam.view * vec4(v.pos, 1) - prev_cam.view * vec4(v.prev_pos, 1)).xyz, 1)";
            break;
        case feature_stage::SCREEN_MOTION:
            feature = "vec4(get_camera_projection(prev_cam, v.prev_pos), 0, 1)";
            break;
        case feature_stage::INSTANCE_ID:
            feature = "vec4(gl_InstanceID, gl_PrimitiveID, 0, 1)";
            break;
        }
        std::map<std::string, std::string> defines;
        rt_camera_stage::get_common_defines(defines, opt);
        defines["FEATURE"] = feature;

        shader_sources fsrc = {
            {}, {},
            {"shader/rt_feature.rgen", defines},
            {
                {
                    vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                    {"shader/rt_feature.rchit", defines},
                    {"shader/rt_feature.rahit"}
                }
            },
            {{"shader/rt_feature.rmiss"}}
        };
        return fsrc;
    }

    struct push_constant_buffer
    {
        vec4 default_value;
        float min_ray_dist;
    };

    static_assert(sizeof(push_constant_buffer) <= 128);
}

}

namespace tr
{

feature_stage::feature_stage(
    device_data& dev,
    uvec2 ray_count,
    const gbuffer_target& output_target,
    const options& opt
):  rt_camera_stage(
        dev, output_target,
        rt_stage::get_common_state(
            ray_count, uvec4(0, 0, output_target.get_size()),
            ::feature::load_sources(opt), opt
        ),
        opt
    ),
    opt(opt)
{
}

void feature_stage::record_command_buffer_push_constants(
    vk::CommandBuffer cb,
    uint32_t /*frame_index*/,
    uint32_t /*pass_index*/
){
    ::feature::push_constant_buffer control;
    control.default_value = opt.default_value;
    control.min_ray_dist = opt.min_ray_dist;

    gfx.push_constants(cb, control);
}

}
