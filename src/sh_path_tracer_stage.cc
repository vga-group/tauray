#include "sh_path_tracer_stage.hh"
#include "scene_stage.hh"
#include "sh_grid.hh"
#include "environment_map.hh"

namespace
{
using namespace tr;

struct grid_data_buffer
{
    pmat4 transform;
    pmat4 normal_transform;
    puvec3 grid_size;
    float mix_ratio;
    pvec3 cell_scale;
    float rotation_x;
    float rotation_y;
};

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

sh_path_tracer_stage::sh_path_tracer_stage(
    device& dev,
    scene_stage& ss,
    texture& output_grid,
    vk::ImageLayout output_layout,
    const options& opt
):  rt_stage(dev, ss, opt, "SH path tracing", 1),
    desc(dev),
    gfx(dev),
    opt(opt),
    output_grid(&output_grid),
    output_layout(output_layout),
    grid_data(dev, sizeof(grid_data_buffer), vk::BufferUsageFlagBits::eUniformBuffer),
    history_length(0)
{
    sample_count_multiplier = opt.samples_per_probe;

    shader_source pl_rint("shader/rt_common_point_light.rint");
    shader_source shadow_chit("shader/rt_common_shadow.rchit");
    std::map<std::string, std::string> defines;
    defines["MAX_BOUNCES"] = std::to_string(opt.max_ray_depth);

    if(opt.russian_roulette_delta > 0)
        defines["USE_RUSSIAN_ROULETTE"];

    add_defines(opt.sampling_weights, defines);
    add_defines(opt.film, defines);
    add_defines(opt.mis_mode, defines);

    if(opt.regularization_gamma != 0.0f)
        defines["PATH_SPACE_REGULARIZATION"];

    defines["SH_ORDER"] = std::to_string(opt.sh_order);
    defines["SH_COEF_COUNT"] = std::to_string(
        sh_grid::get_coef_count(opt.sh_order)
    );

    get_common_defines(defines);

    rt_shader_sources src = {
        {"shader/sh_path_tracer.rgen", defines},
        {
            {
                vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                {"shader/rt_common.rchit"},
                {"shader/rt_common.rahit"}
            },
            {
                vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                shadow_chit,
                {"shader/rt_common_shadow.rahit"}
            },
            {
                vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup,
                {"shader/rt_common_point_light.rchit"},
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
            {"shader/rt_common.rmiss"},
            {"shader/rt_common_shadow.rmiss", defines}
        }
    };
    desc.add(src, 0);
    gfx.init(src, {&desc, &ss.get_descriptors()});
}

void sh_path_tracer_stage::update(uint32_t frame_index)
{
    rt_stage::update(frame_index);
    sh_grid* grid = ss->get_scene()->get<sh_grid>(opt.sh_grid_id);
    transformable* grid_transform = ss->get_scene()->get<transformable>(opt.sh_grid_id);
    mat4 transform = grid_transform->get_global_transform();

    uint32_t sampling_start_counter =
        dev->ctx->get_frame_counter() * opt.samples_per_probe;
    history_length++;
    grid_data.map<grid_data_buffer>(
        frame_index,
        [&](grid_data_buffer* guni){
            guni->transform = transform;
            guni->normal_transform = mat4(get_matrix_orientation(transform));
            guni->grid_size = grid->get_resolution();
            guni->mix_ratio = max(1.0f/history_length, opt.temporal_ratio);
            guni->cell_scale = 0.5f*vec3(grid->get_resolution())/grid_transform->get_scaling();
            guni->rotation_x = pcg(sampling_start_counter)/float(0xFFFFFFFFu);
            guni->rotation_y = pcg(sampling_start_counter+1)/float(0xFFFFFFFFu);
        }
    );
}

void sh_path_tracer_stage::record_command_buffer(
    vk::CommandBuffer cb, uint32_t frame_index, uint32_t pass_index,
    bool /*first_in_command_buffer*/
){
    grid_data.upload(dev->id, frame_index, cb);

    sh_grid* grid = ss->get_scene()->get<sh_grid>(opt.sh_grid_id);
    uvec3 dim = grid->get_resolution();

    vk::ImageMemoryBarrier img_barrier(
        {}, vk::AccessFlagBits::eShaderWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        output_grid->get_image(dev->id),
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
    );

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eRayTracingShaderKHR,
        {}, {}, {}, img_barrier
    );

    gfx.bind(cb);
    desc.set_image("inout_data", *output_grid);
    desc.set_buffer("grid", grid_data);
    get_descriptors(desc);
    gfx.push_descriptors(cb, desc, 0);
    gfx.set_descriptors(cb, ss->get_descriptors(), 0, 1);

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
    push_constant_buffer control;

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
