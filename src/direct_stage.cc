#include "direct_stage.hh"
#include "scene_stage.hh"
#include "misc.hh"
#include "environment_map.hh"

namespace
{
using namespace tr;

namespace direct
{
    rt_shader_sources load_sources(
        direct_stage::options opt,
        const gbuffer_target& gbuf
    ){
        shader_source pl_rint("shader/rt_common_point_light.rint");
        shader_source shadow_chit("shader/rt_common_shadow.rchit");
        std::map<std::string, std::string> defines;
        defines["MAX_BOUNCES"] = std::to_string(opt.max_ray_depth);
        defines["SAMPLES_PER_PASS"] = std::to_string(opt.samples_per_pass);

        if(opt.transparent_background)
            defines["USE_TRANSPARENT_BACKGROUND"];

        add_defines(opt.sampling_weights, defines);

#define TR_GBUFFER_ENTRY(name, ...)\
        if(gbuf.name) defines["USE_"+to_uppercase(#name)+"_TARGET"];
        TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY

        add_defines(opt.film, defines);
        add_defines(opt.bounce_mode, defines);
        add_defines(opt.tri_light_mode, defines);

        rt_camera_stage::get_common_defines(defines, opt);

        return {
            {"shader/direct.rgen", defines},
            {
                {
                    vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                    {"shader/rt_common.rchit", defines},
                    {"shader/rt_common.rahit", defines}
                },
                {
                    vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                    shadow_chit,
                    {"shader/rt_common_shadow.rahit", defines}
                },
                {
                    vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup,
                    {"shader/rt_common_point_light.rchit", defines},
                    {},
                    pl_rint
                },
                {
                    vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup,
                    shadow_chit,
                    {},
                    pl_rint
                }
            },
            {
                {"shader/rt_common.rmiss", defines},
                {"shader/rt_common_shadow.rmiss", defines}
            }
        };
    }

    struct push_constant_buffer
    {
        uint32_t samples;
        uint32_t previous_samples;
        float min_ray_dist;
        float indirect_clamping;
        float film_radius;
        int antialiasing;
    };

    // The minimum maximum size for push constant buffers is 128 bytes in vulkan.
    static_assert(sizeof(push_constant_buffer) <= 128);
}

}

namespace tr
{

direct_stage::direct_stage(
    device& dev,
    scene_stage& ss,
    const gbuffer_target& output_target,
    const options& opt
):  rt_camera_stage(
        dev, ss, output_target, opt, "direct light",
        opt.samples_per_pixel / opt.samples_per_pass
    ),
    desc(dev),
    gfx(dev),
    opt(opt)
{
    rt_shader_sources src = direct::load_sources(opt, output_target);
    desc.add(src);
    gfx.init(src, {&desc, &ss.get_descriptors()});
}

void direct_stage::record_command_buffer_pass(
    vk::CommandBuffer cb,
    uint32_t frame_index,
    uint32_t pass_index,
    uvec3 expected_dispatch_size,
    bool first_in_command_buffer
){
    if(first_in_command_buffer)
    {
        gfx.bind(cb, frame_index);
        get_descriptors(desc);
        gfx.push_descriptors(cb, desc, 0);
        gfx.set_descriptors(cb, ss->get_descriptors(), 0, 1);
    }

    direct::push_constant_buffer control;

    control.film_radius = opt.film_radius;
    control.min_ray_dist = opt.min_ray_dist;

    control.previous_samples = pass_index * opt.samples_per_pass;
    control.samples = opt.samples_per_pass;
    control.antialiasing = opt.film != film_filter::POINT ? 1 : 0;

    gfx.push_constants(cb, control);
    gfx.trace_rays(cb, expected_dispatch_size);
}

}

