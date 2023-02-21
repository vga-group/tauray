#include "path_tracer_stage.hh"
#include "scene.hh"
#include "misc.hh"
#include "environment_map.hh"

namespace
{
using namespace tr;

namespace path_tracer
{
    shader_sources load_sources(
        path_tracer_stage::options opt,
        const gbuffer_target& gbuf
    ){
        shader_source pl_rint("shader/path_tracer_point_light.rint");
        shader_source shadow_chit("shader/path_tracer_shadow.rchit");
        std::map<std::string, std::string> defines;
        defines["MAX_BOUNCES"] = std::to_string(opt.max_ray_depth);

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

        if(opt.sample_point_lights > 0)
            defines["NEE_SAMPLE_POINT_LIGHTS"] = std::to_string(opt.sample_point_lights);
        if(opt.sample_directional_lights > 0)
            defines["NEE_SAMPLE_DIRECTIONAL_LIGHTS"] = std::to_string(opt.sample_directional_lights);
        if(opt.sample_envmap > 0)
            defines["NEE_SAMPLE_ENVMAP"] = std::to_string(opt.sample_envmap);
        if(opt.sample_emissive_triangles > 0)
            defines["NEE_SAMPLE_EMISSIVE_TRIANGLES"] = std::to_string(opt.sample_emissive_triangles);

#define TR_GBUFFER_ENTRY(name, ...)\
        if(gbuf.name) defines["USE_"+to_uppercase(#name)+"_TARGET"];
        TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY

        switch(opt.film)
        {
        case film::POINT:
            defines["USE_POINT_FILTER"];
            break;
        case film::BOX:
            defines["USE_BOX_FILTER"];
            break;
        case film::BLACKMAN_HARRIS:
            defines["USE_BLACKMAN_HARRIS_FILTER"];
            break;
        }

        switch(opt.mis_mode)
        {
        case multiple_importance_sampling_mode::MIS_DISABLED:
            break;
        case multiple_importance_sampling_mode::MIS_BALANCE_HEURISTIC:
            defines["MIS_BALANCE_HEURISTIC"];
            break;
        case multiple_importance_sampling_mode::MIS_POWER_HEURISTIC:
            defines["MIS_POWER_HEURISTIC"];
            break;
        }

        switch(opt.bounce_mode)
        {
        case bounce_sampling_mode::HEMISPHERE:
            defines["BOUNCE_HEMISPHERE"];
            break;
        case bounce_sampling_mode::COSINE_HEMISPHERE:
            defines["BOUNCE_COSINE_HEMISPHERE"];
            break;
        case bounce_sampling_mode::MATERIAL:
            defines["BOUNCE_MATERIAL"];
            break;
        }

        switch(opt.tri_light_mode)
        {
        case tri_light_sampling_mode::AREA:
            defines["TRI_LIGHT_SAMPLE_AREA"];
            break;
        case tri_light_sampling_mode::SOLID_ANGLE:
            defines["TRI_LIGHT_SAMPLE_SOLID_ANGLE"];
            break;
        case tri_light_sampling_mode::HYBRID:
            defines["TRI_LIGHT_SAMPLE_HYBRID"];
            break;
        }

        rt_camera_stage::get_common_defines(defines, opt);

        return {
            {}, {},
            {"shader/path_tracer.rgen", defines},
            {
                {
                    vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                    {"shader/path_tracer.rchit", defines},
                    {"shader/path_tracer.rahit", defines}
                },
                {
                    vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                    shadow_chit,
                    {"shader/path_tracer_shadow.rahit", defines}
                },
                {
                    vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup,
                    {"shader/path_tracer_point_light.rchit", defines},
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
                {"shader/path_tracer.rmiss", defines},
                {"shader/path_tracer_shadow.rmiss", defines}
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
        float russian_roulette_delta;
        int antialiasing;
        // -1 for no environment map
        int environment_proj;
        pvec4 environment_factor;
        float regularization_gamma;
    };

    // The minimum maximum size for push constant buffers is 128 bytes in vulkan.
    static_assert(sizeof(push_constant_buffer) <= 128);
}

}

namespace tr
{

path_tracer_stage::path_tracer_stage(
    device_data& dev,
    uvec2 ray_count,
    const gbuffer_target& output_target,
    const options& opt
):  rt_camera_stage(
        dev, output_target,
        rt_stage::get_common_state(
            ray_count, uvec4(0,0,output_target.get_size()),
            path_tracer::load_sources(opt, output_target), opt
        ),
        opt,
        "path tracing",
        opt.samples_per_pixel
    ),
    opt(opt)
{
}

void path_tracer_stage::record_command_buffer_push_constants(
    vk::CommandBuffer cb,
    uint32_t /*frame_index*/,
    uint32_t pass_index
){
    scene* cur_scene = get_scene();
    path_tracer::push_constant_buffer control;

    environment_map* envmap = cur_scene->get_environment_map();
    if(envmap)
    {
        control.environment_factor = vec4(envmap->get_factor(), 1);
        control.environment_proj = (int)envmap->get_projection();
    }
    else
    {
        control.environment_factor = vec4(0);
        control.environment_proj = -1;
    }

    control.film_radius = opt.film_radius;
    control.russian_roulette_delta = opt.russian_roulette_delta;
    control.min_ray_dist = opt.min_ray_dist;
    control.indirect_clamping = opt.indirect_clamping;
    control.regularization_gamma = opt.regularization_gamma;

    control.previous_samples = pass_index;
    control.samples = min(
        opt.samples_per_pixel - (int)control.previous_samples,
        1
    );
    control.antialiasing = opt.film != film::POINT ? 1 : 0;

    gfx.push_constants(cb, control);
}

}
