#include "feature_stage.hh"
#include "scene_stage.hh"
#include "environment_map.hh"

namespace
{
using namespace tr;

struct push_constant_buffer
{
    vec4 default_value;
    float min_ray_dist;
};

static_assert(sizeof(push_constant_buffer) <= 128);

}

namespace tr
{

feature_stage::feature_stage(
    device& dev,
    scene_stage& ss,
    const gbuffer_target& output_target,
    const options& opt
):  rt_camera_stage(dev, ss, output_target, opt),
    desc(dev),
    gfx(dev),
    opt(opt)
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
        feature = "vec4(get_camera_projection(prev_cam, v.prev_pos), 1)";
        break;
    case feature_stage::INSTANCE_ID:
        feature = "vec4(gl_InstanceID, gl_PrimitiveID, 0, 1)";
        break;
    case feature_stage::METALLNESS:
        feature = "vec4(vec3(mat.metallic), 1.0f)";
        break;
    case feature_stage::ROUGHNESS:
        feature = "vec4(vec3(mat.roughness), 1.0f)";
        break;
    }
    std::map<std::string, std::string> defines;
    rt_camera_stage::get_common_defines(defines);
    defines["FEATURE"] = feature;

    rt_shader_sources src = {
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
    desc.add(src);
    gfx.init(src, {&desc, &ss.get_descriptors()});
}

void feature_stage::record_command_buffer_pass(
    vk::CommandBuffer cb,
    uint32_t /*frame_index*/,
    uint32_t /*pass_index*/,
    uvec3 expected_dispatch_size,
    bool
){
    gfx.bind(cb);
    get_descriptors(desc);
    gfx.push_descriptors(cb, desc, 0);
    gfx.set_descriptors(cb, ss->get_descriptors(), 0, 1);

    push_constant_buffer control;
    control.default_value = opt.default_value;
    control.min_ray_dist = opt.min_ray_dist;

    gfx.push_constants(cb, control);
    gfx.trace_rays(cb, expected_dispatch_size);
}

}
