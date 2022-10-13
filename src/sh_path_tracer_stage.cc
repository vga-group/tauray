#include "sh_path_tracer_stage.hh"
#include "scene.hh"
#include "sh_grid.hh"
#include "environment_map.hh"

namespace
{
using namespace tr;

struct grid_data_buffer
{
    mat4 transform;
    mat4 normal_transform;
    puvec3 grid_size;
    float rotation;
    pvec3 cell_scale;
    float mix_ratio;
};

namespace sh_path_tracer
{
    shader_sources load_sources(const sh_path_tracer_stage::options& opt)
    {
        shader_source pl_rint("shader/path_tracer_point_light.rint");
        shader_source shadow_chit("shader/path_tracer_shadow.rchit");
        std::map<std::string, std::string> defines;
        defines["MAX_BOUNCES"] = std::to_string(opt.max_ray_depth);

        if(opt.russian_roulette_delta > 0)
            defines["USE_RUSSIAN_ROULETTE"];

        if(opt.importance_sample_envmap)
            defines["IMPORTANCE_SAMPLE_ENVMAP"];

        if(opt.regularization_gamma != 0.0f)
            defines["PATH_SPACE_REGULARIZATION"];

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

        defines["SH_ORDER"] = std::to_string(opt.sh_order);
        defines["SH_COEF_COUNT"] = std::to_string(
            sh_grid::get_coef_count(opt.sh_order)
        );

        rt_stage::get_common_defines(defines, opt);

        return {
            {}, {},
            {"shader/sh_path_tracer.rgen", defines},
            {
                {
                    vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                    {"shader/path_tracer.rchit"},
                    {"shader/path_tracer.rahit"}
                },
                {
                    vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                    shadow_chit,
                    {"shader/path_tracer_shadow.rahit"}
                },
                {
                    vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup,
                    {"shader/path_tracer_point_light.rchit"},
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
                {"shader/path_tracer.rmiss"},
                {"shader/path_tracer_shadow.rmiss"}
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

sh_path_tracer_stage::sh_path_tracer_stage(
    device_data& dev,
    texture& output_grid,
    vk::ImageLayout output_layout,
    const options& opt
):  rt_stage(
        dev,
        rt_stage::get_common_state(
            uvec2(0), uvec4(0), sh_path_tracer::load_sources(opt), opt
        ),
        opt, "SH path tracing", 1
    ),
    opt(opt),
    output_grid(&output_grid),
    output_layout(output_layout),
    grid_data(dev, sizeof(grid_data_buffer), vk::BufferUsageFlagBits::eUniformBuffer)
{
    rt_stage::set_local_sampler_parameters(
        output_grid.get_dimensions(),
        opt.samples_per_probe
    );
}

void sh_path_tracer_stage::update(uint32_t frame_index)
{
    rt_stage::update(frame_index);
    sh_grid* grid = get_scene()->get_sh_grids()[opt.sh_grid_index];
    mat4 transform = grid->get_global_transform();

    uint32_t sampling_start_counter =
        dev->ctx->get_frame_counter() * opt.samples_per_probe;
    grid_data.map<grid_data_buffer>(
        frame_index,
        [&](grid_data_buffer* guni){
            guni->transform = transform;
            guni->normal_transform = mat4(get_matrix_orientation(transform));
            guni->grid_size = grid->get_resolution();
            guni->mix_ratio = max(1.0f/dev->ctx->get_frame_counter(), opt.temporal_ratio);
            guni->cell_scale = 0.5f*vec3(grid->get_resolution())/grid->get_scaling();
            guni->rotation = pcg(sampling_start_counter)/float(0xFFFFFFFFu);
        }
    );
}

void sh_path_tracer_stage::init_scene_resources()
{
    rt_stage::init_scene_resources();

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        get_scene()->bind(gfx, i, -1);
        gfx.update_descriptor_set({
            {"inout_data", {{}, output_grid->get_image_view(dev->index), vk::ImageLayout::eGeneral}},
            {"grid", {*grid_data, 0, VK_WHOLE_SIZE}}
        }, i);
    }
}

void sh_path_tracer_stage::record_command_buffer(
    vk::CommandBuffer cb, uint32_t frame_index, uint32_t pass_index
){
    grid_data.upload(frame_index, cb);

    sh_grid* grid = get_scene()->get_sh_grids()[opt.sh_grid_index];
    uvec3 dim = grid->get_resolution();

    vk::ImageMemoryBarrier img_barrier(
        {}, vk::AccessFlagBits::eShaderWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        output_grid->get_image(dev->index),
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
    );

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eRayTracingShaderKHR,
        {}, {}, {}, img_barrier
    );

    gfx.bind(cb, frame_index);

    record_command_buffer_push_constants(cb, frame_index, pass_index);

    gfx.trace_rays(
        cb,
        uvec3(dim.x, dim.y, dim.z * (opt.samples_per_probe/opt.samples_per_invocation))
    );

    img_barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    img_barrier.dstAccessMask = {};
    img_barrier.oldLayout = vk::ImageLayout::eGeneral;
    img_barrier.newLayout = output_layout;
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eRayTracingShaderKHR,
        vk::PipelineStageFlagBits::eBottomOfPipe,
        {}, {}, {}, img_barrier
    );
}

void sh_path_tracer_stage::record_command_buffer_push_constants(
    vk::CommandBuffer cb,
    uint32_t /*frame_index*/,
    uint32_t /*pass_index*/
){
    scene* cur_scene = get_scene();
    sh_path_tracer::push_constant_buffer control;

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

    // We re-use the "previous samples" to mark how many samples to do in one
    // shader invocation. The name can't change on the shader side, since it is
    // shared with the normal path tracer.
    control.previous_samples = opt.samples_per_invocation;
    control.samples = opt.samples_per_probe/opt.samples_per_invocation;
    control.antialiasing = false;

    gfx.push_constants(cb, control);
}

}
