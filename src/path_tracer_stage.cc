#include "path_tracer_stage.hh"
#include "scene_stage.hh"
#include "misc.hh"
#include "environment_map.hh"
#include "log.hh"

namespace
{
using namespace tr;

struct push_constant_buffer
{
    uint32_t samples;
    uint32_t previous_samples;
    float min_ray_dist;
    float indirect_clamping;
    float film_radius;
    float russian_roulette_delta;
    int antialiasing;
    float regularization_gamma;
};

// The minimum maximum size for push constant buffers is 128 bytes in vulkan.
static_assert(sizeof(push_constant_buffer) <= 128);

}

namespace tr
{

path_tracer_stage::path_tracer_stage(
    device& dev,
    scene_stage& ss,
    const gbuffer_target& output_target,
    const options& opt
):  rt_camera_stage(
        dev, ss, output_target, opt, "path tracing",
        opt.samples_per_pixel / opt.samples_per_pass
    ),
    desc(dev),
    gfx(dev),
    opt(opt)
{
    shader_source pl_rint("shader/rt_common_point_light.rint");
    shader_source shadow_chit("shader/rt_common_shadow.rchit");
    std::map<std::string, std::string> defines;
    defines["MAX_BOUNCES"] = std::to_string(opt.max_ray_depth);
    defines["SAMPLES_PER_PASS"] = std::to_string(opt.samples_per_pass);

    if(opt.russian_roulette_delta > 0)
        defines["USE_RUSSIAN_ROULETTE"];

    if(opt.use_shadow_terminator_fix)
        defines["USE_SHADOW_TERMINATOR_FIX"];

    if(opt.use_white_albedo_on_first_bounce)
        defines["USE_WHITE_ALBEDO_ON_FIRST_BOUNCE"];

    if(opt.hide_lights)
        defines["HIDE_LIGHTS"];

    if(opt.transparent_background)
        defines["USE_TRANSPARENT_BACKGROUND"];

    if(opt.regularization_gamma != 0.0f)
        defines["PATH_SPACE_REGULARIZATION"];

    if(opt.depth_of_field)
        defines["USE_DEPTH_OF_FIELD"];

#if 0 
    if(opt.boda == bd::BOUNCE_COUNT)
        defines["BD_BOUNCE_COUNT"];

    if(opt.boda == bd::CONTRIBUTION)
        defines["BD_CONTRIBUTION"];

    if(opt.boda == bd::MATERIAL_ID)
        defines["BD_MATERIAL_ID"];

    if(opt.boda == bd::BSDF_SUM)
        defines["BD_BSDF_SUM"];

    if(opt.boda == bd::BSDF_VAR)
        defines["BD_BSDF_VAR"];

    if(opt.boda == bd::PDF_CONTRIBUTION)
        defines["BD_PDF_CONTRIBUTION"];

    if(opt.boda == bd::FULL_PDF_CONTRIBUTION)
        defines["BD_FULL_PDF_CONTRIBUTION"];

    if(opt.dt == denoiser_type::BMFR)
        defines["BD_BMFR"];
#endif

    if(opt.boda != bd::OFF)
        defines[bd_str_map.at(opt.boda)];

    if(opt.dt == denoiser_type::BMFR)
        defines["BD_BMFR"];

    int id = 0;
    for(auto i = 0; i < (int)opt.bd_vec.size(); i++)
    {
        auto option = opt.bd_vec.at(i);
        if(option != bd::POSITION && option != bd::POSITION_2 && option != bd::NORMAL)
        {
            defines[bd_str_map.at(option) + " " + std::to_string(id)];
            TR_LOG(bd_str_map.at(option) + " " + std::to_string(id));
            ++id;
        }
    }

#define TR_GBUFFER_ENTRY(name, ...)\
    if(output_target.name) defines["USE_"+to_uppercase(#name)+"_TARGET"];
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY

    add_defines(opt.sampling_weights, defines);
    add_defines(opt.film, defines);
    add_defines(opt.mis_mode, defines);
    add_defines(opt.bounce_mode, defines);
    add_defines(opt.tri_light_mode, defines);

    get_common_defines(defines);

    rt_shader_sources src = {
        {"shader/path_tracer.rgen", defines},
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
    desc.add(src);
    gfx.init(src, {&desc, &ss.get_descriptors()});
}

void path_tracer_stage::record_command_buffer_pass(
    vk::CommandBuffer cb,
    uint32_t,
    uint32_t pass_index,
    uvec3 expected_dispatch_size,
    bool first_in_command_buffer
){
    if(first_in_command_buffer)
    {
        gfx.bind(cb);
        get_descriptors(desc);
        gfx.push_descriptors(cb, desc, 0);
        gfx.set_descriptors(cb, ss->get_descriptors(), 0, 1);
    }

    push_constant_buffer control;

    control.film_radius = opt.film_radius;
    control.russian_roulette_delta = opt.russian_roulette_delta;
    control.min_ray_dist = opt.min_ray_dist;
    control.indirect_clamping = opt.indirect_clamping;
    control.regularization_gamma = opt.regularization_gamma;

    control.previous_samples = pass_index * opt.samples_per_pass;
    control.samples = opt.samples_per_pass;
    control.antialiasing = opt.film != film_filter::POINT ? 1 : 0;

    gfx.push_constants(cb, control);
    gfx.trace_rays(cb, expected_dispatch_size);
}

}
