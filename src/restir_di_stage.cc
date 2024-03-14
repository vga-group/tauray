#include "restir_di_stage.hh"
#include "scene.hh"
#include "scene_stage.hh"
#include "misc.hh"
#include "environment_map.hh"
#include "log.hh"

namespace
{
using namespace tr;

namespace restir_di
{
    rt_shader_sources load_sources(
        restir_di_stage::options opt,
        const gbuffer_target& gbuf
    ){
        shader_source pl_rint("shader/rt_common_point_light.rint");
        shader_source shadow_chit("shader/rt_common_shadow.rchit");
        std::map<std::string, std::string> defines;
        defines["RIS_SAMPLE_COUNT"] = std::to_string(opt.ris_sample_count);
        defines["MAX_BOUNCES"] = std::to_string(opt.max_ray_depth);

        if(opt.temporal_reuse)
            defines["TEMPORAL_REUSE"];
        if(opt.shared_visibility)
        {
            defines["SHARED_VISIBILITY"];
            if(opt.sample_visibility)
                defines["SAMPLE_VISIBILITY"];
        }

#define TR_GBUFFER_ENTRY(name, ...)\
        if(gbuf.name) defines["USE_"+to_uppercase(#name)+"_TARGET"];
        TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY

        add_defines(opt.sampling_weights, defines);
        add_defines(opt.tri_light_mode, defines);

        rt_camera_stage::get_common_defines(defines, opt);

        return {
            {"shader/restir_di_canonical_and_temporal.rgen", defines},
            {
                { // Regular ray, triangle meshes
                    vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                    {"shader/rt_common.rchit", defines},
                    {"shader/rt_common.rahit", defines}
                },
                { // Shadow ray, triangle meshes
                    vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                    shadow_chit,
                    {"shader/rt_common_shadow.rahit", defines}
                },
                { // Area light ray, sphere intersection
                    vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup,
                    {"shader/rt_common_point_light.rchit", defines},
                    {},
                    pl_rint
                },
                { // Area light shadow ray, sphere intersection
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

    rt_shader_sources load_spatial_reuse_resources(
        restir_di_stage::options opt,
        const gbuffer_target& gbuf
    ){
        shader_source pl_rint("shader/rt_common_point_light.rint");
        shader_source shadow_chit("shader/rt_common_shadow.rchit");
        std::map<std::string, std::string> defines;
        defines["SPATIAL_SAMPLE_COUNT"] = std::to_string(opt.spatial_sample_count);
        defines["MAX_BOUNCES"] = std::to_string(opt.max_ray_depth);

        if(opt.spatial_reuse)
            defines["SPATIAL_REUSE"];
        if(opt.shared_visibility)
        {
            defines["SHARED_VISIBILITY"];
            if(opt.sample_visibility)
                defines["SAMPLE_VISIBILITY"];
        }

#define TR_GBUFFER_ENTRY(name, ...)\
        if(gbuf.name) defines["USE_"+to_uppercase(#name)+"_TARGET"];
        TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY

        add_defines(opt.sampling_weights, defines);
        add_defines(opt.tri_light_mode, defines);

        rt_camera_stage::get_common_defines(defines, opt);

        return {
            {"shader/restir_di_spatial.rgen", defines},
            {
                { // Regular ray, triangle meshes
                    vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                    {"shader/rt_common.rchit", defines},
                    {"shader/rt_common.rahit", defines}
                },
                { // Shadow ray, triangle meshes
                    vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                    shadow_chit,
                    {"shader/rt_cmomon_shadow.rahit", defines}
                },
                { // Area light ray, sphere intersection
                    vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup,
                    {"shader/rt_common_point_light.rchit", defines},
                    {},
                    pl_rint
                },
                { // Area light shadow ray, sphere intersection
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
        float max_confidence;
        float search_radius;
    };

    static_assert(sizeof(push_constant_buffer) <= 128);
}
}

namespace tr
{

restir_di_stage::restir_di_stage(
    device& dev,
    scene_stage& ss,
    const gbuffer_target& output_target,
    const options& opt
):  rt_camera_stage(dev, ss, output_target, opt, "restir_di", 1),
    gfx(dev, rt_stage::get_common_options(ss,
        restir_di::load_sources(opt, output_target),
        opt
    )),
    spatial_reuse(dev, rt_stage::get_common_options(ss,
        restir_di::load_spatial_reuse_resources(opt, output_target),
        opt
    )),
    opt(opt),
    param_buffer(
        dev,
        sizeof(int),
        vk::BufferUsageFlagBits::eUniformBuffer
    ),
    reservoir_data(
        dev,
        output_target.get_size(),
        2,
        vk::Format::eR32G32B32A32Sfloat,
        0,
        nullptr,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage
    ),
    light_data(
        dev,
        output_target.get_size(),
        2,
        vk::Format::eR16G16Unorm,
        0,
        nullptr,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage
    ),
    previous_normal_data(
        dev,
        output_target.get_size(),
        2,
        vk::Format::eR16G16Snorm,
        0,
        nullptr,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage
    ),
    previous_pos_data(
        dev,
        output_target.get_size(),
        2,
        vk::Format::eR32G32B32A32Sfloat,
        0,
        nullptr,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage
    )
{
}

void restir_di_stage::update(uint32_t frame_index)
{
    rt_camera_stage::update(frame_index);

    int parity = dev->ctx->get_frame_counter() ?
        dev->ctx->get_frame_counter() & 1 : -1;

    param_buffer.update(frame_index, &parity);
}

void restir_di_stage::init_scene_resources()
{
    rt_camera_stage::init_descriptors(gfx);
    rt_camera_stage::init_descriptors(spatial_reuse);

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        gfx.update_descriptor_set({
            {"parity_data", {param_buffer[dev->id], 0, VK_WHOLE_SIZE}},
            {"reservoir_data", {VK_NULL_HANDLE, reservoir_data.get_array_image_view(dev->id),
                vk::ImageLayout::eGeneral}},
            {"light_data_uni",{VK_NULL_HANDLE, light_data.get_array_image_view(dev->id),
                vk::ImageLayout::eGeneral}},
            {"previous_normal_data",{VK_NULL_HANDLE, previous_normal_data.get_array_image_view(dev->id),
                vk::ImageLayout::eGeneral}},
            {"previous_pos_data",{VK_NULL_HANDLE, previous_pos_data.get_array_image_view(dev->id),
                vk::ImageLayout::eGeneral}}
        }, i);

        spatial_reuse.update_descriptor_set({
            {"parity_data", {param_buffer[dev->id], 0, VK_WHOLE_SIZE}},
            {"reservoir_data", {VK_NULL_HANDLE, reservoir_data.get_array_image_view(dev->id),
                vk::ImageLayout::eGeneral}},
            {"light_data_uni",{VK_NULL_HANDLE, light_data.get_array_image_view(dev->id),
                vk::ImageLayout::eGeneral}},
            {"previous_normal_data",{VK_NULL_HANDLE, previous_normal_data.get_array_image_view(dev->id),
                vk::ImageLayout::eGeneral}},
            {"previous_pos_data",{VK_NULL_HANDLE, previous_pos_data.get_array_image_view(dev->id),
                vk::ImageLayout::eGeneral}}
        }, i);
    }
}

void restir_di_stage::record_command_buffer_pass(
    vk::CommandBuffer cb,
    uint32_t frame_index,
    uint32_t pass_index,
    uvec3 expected_dispatch_size,
    bool
){
    restir_di::push_constant_buffer control;

    param_buffer.upload(dev->id, frame_index, cb);

    control.min_ray_dist = opt.min_ray_dist;
    control.max_confidence = opt.max_confidence;
    control.search_radius = opt.search_radius;

    control.previous_samples = pass_index * opt.samples_per_pass;
    control.samples = opt.samples_per_pass;

    //RIS + temporal reuse
    gfx.bind(cb, frame_index);
    gfx.set_descriptors(cb, ss->get_descriptors(), 0, 1);

    gfx.push_constants(cb, control);
    gfx.trace_rays(cb, expected_dispatch_size);

    vk::ImageMemoryBarrier barrier(
        vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite,
        vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        reservoir_data.get_image(dev->id),
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0,1,0,2)
    );

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eRayTracingShaderKHR,
        vk::PipelineStageFlagBits::eRayTracingShaderKHR,
        {}, {}, {}, {barrier}
    );

    //Spatial reuse
    spatial_reuse.bind(cb, frame_index);
    spatial_reuse.set_descriptors(cb, ss->get_descriptors(), 0, 1);
    spatial_reuse.push_constants(cb, control);
    spatial_reuse.trace_rays(cb, expected_dispatch_size);

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eRayTracingShaderKHR,
        vk::PipelineStageFlagBits::eRayTracingShaderKHR,
        {}, {}, {}, {barrier}
    );
}

}
