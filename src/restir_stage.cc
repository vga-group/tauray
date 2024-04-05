#include "restir_stage.hh"
#include "scene_stage.hh"
#include "gbuffer.hh"
#include "log.hh"
#include "misc.hh"

namespace
{
using namespace tr;

struct restir_config
{
    gpu_shadow_mapping_parameters sm_params;

    float min_ray_dist;
    float max_ray_dist;
    //float regularization_gamma;

    float reconnection_scale;
    float max_confidence;

    float min_spatial_radius;
    float max_spatial_radius;
};

struct canonical_push_constant_buffer
{
    restir_config config;
    uint32_t sample_index;
    uint32_t camera_index;
    uint32_t first_pass;
};

struct temporal_push_constant_buffer
{
    restir_config config;
    float search_base_radius;
    float search_widening;
    uint32_t sample_index;
    uint32_t camera_index;
    pvec2 jitter;
    int permutation;
};

struct selection_push_constant_buffer
{
    restir_config config;
    uint32_t sample_index;
    uint32_t camera_index;
    puvec2 display_size;
};

struct spatial_trace_push_constant_buffer
{
    restir_config config;
    uint32_t sample_index;
    uint32_t camera_index;
};

struct spatial_gather_push_constant_buffer
{
    restir_config config;
    puvec2 display_size;
    uint32_t sample_index;
    uint32_t camera_index;
    uint32_t accumulated_samples;
    uint32_t initialize_output;
    uint32_t accumulate_color;
    uint32_t update_sample_color;
};

struct spatial_single_push_constant_buffer
{
    restir_config config;
    uint32_t sample_index;
    uint32_t camera_index;
    uint32_t accumulated_samples;
    uint32_t initialize_output;
    uint32_t accumulate_color;
    uint32_t update_sample_color;
};

struct spatial_selection_data
{
    uint candidate_key; // 2 bits per candidate, max 16 candidates.
    float total_confidence;
};

struct spatial_candidate_data
{
    pvec4 color_half_jacobian;
};

struct spatial_canonical_mis_data
{
    float mis_part;
};

restir_config get_restir_config(restir_stage::options& opt, gbuffer_target& output, scene_stage& ss)
{
    uvec2 size = output.albedo.size;
    restir_config config;
    config.sm_params = create_shadow_mapping_parameters(opt.sm_filter, ss);
    config.max_ray_dist = opt.max_ray_dist;
    config.min_ray_dist = opt.min_ray_dist;
    config.reconnection_scale = opt.reconnection_scale * opt.max_spatial_search_radius / size.x;
    config.max_confidence = opt.max_confidence;
    config.min_spatial_radius = opt.min_spatial_search_radius / size.x;
    config.max_spatial_radius = opt.max_spatial_search_radius / size.x;
    return config;
}

}

namespace tr
{

#define USED_BUFFERS \
    X(depth) \
    X(pos) \
    X(normal) \
    X(flat_normal) \
    X(albedo) \
    X(emission) \
    X(material)

restir_stage::restir_stage(
    device& dev,
    scene_stage& ss,
    gbuffer_target& current_buffers,
    gbuffer_target& previous_buffers,
    const options& opt
):  single_device_stage(dev),
    scene_data(&ss),
    canonical(dev),
    canonical_set(dev),
    temporal(dev),
    temporal_set(dev),
    selection(dev),
    selection_set(dev),
    spatial_trace(dev),
    spatial_trace_set(dev),
    spatial_gather(dev),
    spatial_gather_set(dev),
    current_buffers(current_buffers),
    previous_buffers(previous_buffers),
    gbuf_sampler(
        dev, vk::Filter::eNearest, vk::Filter::eNearest,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest,
        0, true, false, false, 0
    ),
    opt(opt),
    stage_timer(dev, "restir")
{
    reservoir_data_parity = 0,
    accumulated_samples = 0;
    valid_history_frame = UINT64_MAX;

    gbuffer_target& c = current_buffers;
    gbuffer_target& p = previous_buffers;
    // There's an internal bitmask that won't deal with more than 16 spatial
    // samples, sorry.
    assert(opt.spatial_samples <= 16);
    assert((c.depth && p.depth) || (c.pos && p.pos));
    assert(c.normal && p.normal);
    assert(c.albedo);
    assert(c.material);
    assert(c.screen_motion);
    assert((c.emission && p.emission));

    if((c.albedo && !p.albedo) || (c.material && !p.material))
    {
        this->opt.assume_unchanged_material = true;
    }

    if(!c.color && !opt.demodulated_output)
    {
        throw std::runtime_error("Missing color output buffer!");
    }
    if((!c.diffuse || !c.reflection) && opt.demodulated_output)
    {
        throw std::runtime_error("Missing demodulated output buffers (diffuse, reflection)!");
    }

    //if(!scene_data->has_temporal_tlas() && !opt.assume_unchanged_acceleration_structures)
    {
        // TODO: Keep previous acceleration structures around.
        TR_LOG(
            "ReSTIR will now assume unchanged acceleration structures due to "
            "previous acceleration structures not being available. This feature "
            "has not been implemented yet in Tauray."
        );
        this->opt.assume_unchanged_acceleration_structures = true;
    }

    uvec2 size = current_buffers.color.size;
    cached_sample_color.emplace(
        dev,
        size,
        1,
        vk::Format::eR32G32B32A32Sfloat,
        0, nullptr,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage,
        vk::ImageLayout::eGeneral,
        vk::SampleCountFlagBits::e1
    );

    for(int i = 0; i < 2; ++i)
    {
        reservoir_textures& rtex = reservoir_data[i];
        rtex.ris_data.emplace(
            dev,
            size,
            1,
            vk::Format::eR32G32B32A32Uint,
            0, nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage,
            vk::ImageLayout::eGeneral,
            vk::SampleCountFlagBits::e1
        );
        if(opt.shift_map != RANDOM_REPLAY_SHIFT)
        {
            rtex.reconnection_data.emplace(
                dev,
                size,
                1,
                vk::Format::eR32G32B32A32Uint,
                0, nullptr,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eStorage,
                vk::ImageLayout::eGeneral,
                vk::SampleCountFlagBits::e1
            );
            rtex.reconnection_radiance.emplace(
                dev,
                size,
                1,
                vk::Format::eR32G32B32A32Sfloat,
                0, nullptr,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eStorage,
                vk::ImageLayout::eGeneral,
                vk::SampleCountFlagBits::e1
            );
        }
        if(opt.max_bounces != 1 || opt.shift_map != RECONNECTION_SHIFT)
        {
            rtex.rng_seeds.emplace(
                dev, size, 1,
                vk::Format::eR32G32B32A32Sfloat,
                0, nullptr,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eStorage,
                vk::ImageLayout::eGeneral,
                vk::SampleCountFlagBits::e1
            );
        }
    }

    std::map<std::string, std::string> defines;
    ss.get_defines(defines);

    uint32_t visibility_ray_mask = 0xFF ^ 0x02;
    uint32_t ray_mask = 0xFF;
    if(opt.shade_all_explicit_lights) ray_mask ^= 0x02;

    defines["VISIBILITY_RAY_MASK"] = std::to_string(visibility_ray_mask);
    defines["RAY_MASK"] = std::to_string(ray_mask);
    defines["MAX_BOUNCES"] = std::to_string(opt.max_bounces);
    defines["CANONICAL_SAMPLES"] = std::to_string(opt.canonical_samples);
    defines["TEMPORAL_REUSE_ATTEMPTS"] = std::to_string(opt.temporal_reuse_search_attempts);
    defines["MAX_SPATIAL_SAMPLES"] = std::to_string(opt.spatial_samples);
    if(opt.spatial_samples > 0)
    {
        selection_tile_size = max(next_power_of_two(sqrt(256/opt.spatial_samples)), 1u);
        defines["SELECTION_TILE_SIZE"] = std::to_string(selection_tile_size);
    }
    if(c.pos) defines["USE_POSITION"];
    if(c.flat_normal) defines["USE_FLAT_NORMAL"];
    if(!opt.opaque_only) defines["STOCHASTIC_ALPHA_BLENDING"];

    if(this->opt.assume_unchanged_material)
        defines["ASSUME_SAME_MATERIAL_IN_TEMPORAL"];
    if(this->opt.assume_unchanged_reconnection_radiance)
        defines["ASSUME_UNCHANGED_RECONNECTION_RADIANCE"];
    if(this->opt.assume_unchanged_temporal_visibility)
        defines["ASSUME_UNCHANGED_TEMPORAL_VISIBILITY"];
    if(this->opt.assume_unchanged_acceleration_structures)
        defines["ASSUME_UNCHANGED_ACCELERATION_STRUCTURES"];
    if(this->opt.spatial_sample_oriented_disk)
        defines["NEIGHBOR_SAMPLE_ORIENTED_DISKS"];
    if(this->opt.shade_all_explicit_lights)
        defines["SHADE_ALL_EXPLICIT_LIGHTS"];
    if(this->opt.shade_fake_indirect)
        defines["SHADE_FAKE_INDIRECT"];
    if(this->opt.demodulated_output)
        defines["DEMODULATE_OUTPUT"];

    switch(opt.shift_map)
    {
    case RECONNECTION_SHIFT:
        defines["USE_RECONNECTION_SHIFT"];
        break;
    case RANDOM_REPLAY_SHIFT:
        defines["USE_RANDOM_REPLAY_SHIFT"];
        break;
    case HYBRID_SHIFT:
        defines["USE_HYBRID_SHIFT"];
        break;
    }

    rt_shader_sources shader;
    shader_source pl_rint("shader/rt_common_point_light.rint");
    shader_source shadow_chit("shader/rt_common_shadow.rchit");
    shader.rmiss = {{
        {"shader/rt_common.rmiss", defines},
        {"shader/rt_common_shadow.rmiss", defines}
    }};
    shader.rhit = {
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
    };
#define SET_BINDING_PARAMS(set) \
    set.add(shader); \
    set.set_binding_params("envmap_alias_table", 1, vk::DescriptorBindingFlagBits::ePartiallyBound);\
    set.set_binding_params("emission_tex", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("flat_normal_tex", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("prev_emission_tex", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("prev_flat_normal_tex", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("in_reservoir_ris_data_tex", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("in_reservoir_reconnection_data_tex", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("in_reservoir_reconnection_radiance_tex", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("in_reservoir_rng_seeds_tex", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("out_reservoir_ris_data_tex", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("out_reservoir_reconnection_data_tex", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("out_reservoir_reconnection_radiance_tex", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("out_reservoir_rng_seeds_tex", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("out_diffuse", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("out_reflection", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("out_length", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("spatial_selection", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("spatial_candidates", 1, vk::DescriptorBindingFlagBits::ePartiallyBound); \
    set.set_binding_params("mis_data", 1, vk::DescriptorBindingFlagBits::ePartiallyBound);

    { // CANONICAL
        shader.rgen = {"shader/restir_canonical.rgen", defines};
        SET_BINDING_PARAMS(canonical_set);
        canonical.init(
            shader,
            {
                &canonical_set,
                &ss.get_descriptors(),
                &ss.get_raster_descriptors()
            }
        );
    }

    { // TEMPORAL
        shader.rgen = shader_source("shader/restir_temporal.rgen", defines);
        SET_BINDING_PARAMS(temporal_set);
        temporal.init(
            shader,
            {
                &temporal_set,
                &ss.get_descriptors()
                // TODO: Temporal mapping tables!
                //scene_data->get_temporal_tables_descriptor_set().get_layout()
            }
        );
    }

    if(opt.spatial_samples > 0)
    {
        selection_data.emplace(
            dev,
            size,
            1,
            vk::Format::eR32G32Uint,
            0, nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage,
            vk::ImageLayout::eGeneral,
            vk::SampleCountFlagBits::e1
        );
        spatial_mis_data.emplace(
            dev,
            size,
            max(opt.spatial_samples, 1u),
            vk::Format::eR32G32B32A32Sfloat,
            0, nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage,
            vk::ImageLayout::eGeneral,
            vk::SampleCountFlagBits::e1
        );
        spatial_candidate_color.emplace(
            dev,
            size,
            max(opt.spatial_samples, 1u),
            vk::Format::eR32G32B32A32Sfloat,
            0, nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage,
            vk::ImageLayout::eGeneral,
            vk::SampleCountFlagBits::e1
        );
    }

    if(opt.spatial_samples > 0)
    { // SPATIAL SELECTION
        shader_source shader = {"shader/restir_spatial_selection.comp", defines};
        SET_BINDING_PARAMS(selection_set);
        selection.init(
            shader,
            {
                &selection_set,
                &ss.get_descriptors(),
                &ss.get_raster_descriptors()
            }
        );
    }

    if(opt.spatial_samples > 0)
    { // SPATIAL TRACE
        shader.rgen = {"shader/restir_spatial_trace.rgen", defines};
        SET_BINDING_PARAMS(spatial_trace_set);
        spatial_trace.init(
            shader,
            {
                &spatial_trace_set,
                &ss.get_descriptors(),
                &ss.get_raster_descriptors()
            }
        );
    }

    { // SPATIAL GATHER
        shader_source shader = {"shader/restir_spatial_gather.comp", defines};
        SET_BINDING_PARAMS(spatial_gather_set);
        spatial_gather.init(
            shader,
            {
                &spatial_gather_set,
                &ss.get_descriptors(),
                &ss.get_raster_descriptors()
            }
        );
    }

    if(opt.demodulated_output)
    {
        current_buffers.diffuse.layout = vk::ImageLayout::eGeneral;
        current_buffers.reflection.layout = vk::ImageLayout::eGeneral;
    }
    else current_buffers.color.layout = vk::ImageLayout::eGeneral;

#define BUF(name) \
    if(c.name) c.name.layout = vk::ImageLayout::eShaderReadOnlyOptimal; \
    if(p.name) p.name.layout = vk::ImageLayout::eShaderReadOnlyOptimal;

    BUF(depth)
    BUF(pos)
    BUF(normal)
    BUF(flat_normal)
    BUF(albedo)
    BUF(emission)
    BUF(material)
#undef OPT_BUF
    c.screen_motion.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
}

void restir_stage::reset_accumulation()
{
    accumulated_samples = 0;
}

void restir_stage::update(uint32_t frame_index)
{
    clear_commands();

    vk::CommandBuffer cmd = begin_graphics(true);
    stage_timer.begin(cmd, dev->id, frame_index);

    if(opt.demodulated_output)
    {
        current_buffers.diffuse.transition_layout_temporary(cmd, vk::ImageLayout::eGeneral);
        current_buffers.reflection.transition_layout_temporary(cmd, vk::ImageLayout::eGeneral);
    }
    else current_buffers.color.transition_layout_temporary(cmd, vk::ImageLayout::eGeneral);

#define X(name) \
    if(current_buffers.name) \
        current_buffers.name.transition_layout_temporary(cmd, vk::ImageLayout::eShaderReadOnlyOptimal); \
    if(previous_buffers.name) \
        previous_buffers.name.transition_layout_temporary(cmd, vk::ImageLayout::eShaderReadOnlyOptimal); \

    USED_BUFFERS
#undef X
    current_buffers.screen_motion.transition_layout_temporary(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

#define X(name) \
    { \
        const char* bind_name = #name "_tex";\
        const char* prev_bind_name = "prev_" #name "_tex";\
        bool do_bind = current_buffers.name;\
        if(\
            &current_buffers.name == &current_buffers.depth ||\
            &current_buffers.name == &current_buffers.pos\
        ){\
            bind_name = "depth_or_position_tex";\
            prev_bind_name = "prev_depth_or_position_tex";\
            /* Don't bind depth if position is available.*/\
            if(\
                &current_buffers.name == &current_buffers.depth &&\
                current_buffers.pos\
            ) do_bind = false;\
        }\
        if(do_bind)\
        {\
            if(current_buffers.name) \
                set.set_image(dev->id, bind_name, {{gbuf_sampler.get_sampler(dev->id), current_buffers.name.view, vk::ImageLayout::eShaderReadOnlyOptimal}}); \
            if(previous_buffers.name) \
                set.set_image(dev->id, prev_bind_name, {{gbuf_sampler.get_sampler(dev->id), previous_buffers.name.view, vk::ImageLayout::eShaderReadOnlyOptimal}});\
        }\
    }

#define sample_color_barrier(happens_before, happens_after) \
    { \
        vk::ImageMemoryBarrier barriers[3]; \
        uint32_t barrier_count = 0; \
        if(!opt.demodulated_output) \
        { \
            barriers[barrier_count++] = vk::ImageMemoryBarrier{ \
                happens_before, \
                happens_after, \
                vk::ImageLayout::eGeneral, \
                vk::ImageLayout::eGeneral, \
                VK_QUEUE_FAMILY_IGNORED, \
                VK_QUEUE_FAMILY_IGNORED, \
                cached_sample_color->get_image(dev->id), \
                { \
                    vk::ImageAspectFlagBits::eColor, \
                    0, VK_REMAINING_MIP_LEVELS, \
                    0, VK_REMAINING_ARRAY_LAYERS \
                } \
            }; \
        } \
        else \
        { \
            barriers[barrier_count++] = vk::ImageMemoryBarrier{ \
                happens_before, \
                happens_after, \
                vk::ImageLayout::eGeneral, \
                vk::ImageLayout::eGeneral, \
                VK_QUEUE_FAMILY_IGNORED, \
                VK_QUEUE_FAMILY_IGNORED, \
                current_buffers.diffuse.image, \
                { \
                    vk::ImageAspectFlagBits::eColor, \
                    0, VK_REMAINING_MIP_LEVELS, \
                    0, VK_REMAINING_ARRAY_LAYERS \
                } \
            }; \
            barriers[barrier_count++] = vk::ImageMemoryBarrier{ \
                happens_before, \
                happens_after, \
                vk::ImageLayout::eGeneral, \
                vk::ImageLayout::eGeneral, \
                VK_QUEUE_FAMILY_IGNORED, \
                VK_QUEUE_FAMILY_IGNORED, \
                current_buffers.reflection.image, \
                { \
                    vk::ImageAspectFlagBits::eColor, \
                    0, VK_REMAINING_MIP_LEVELS, \
                    0, VK_REMAINING_ARRAY_LAYERS \
                } \
            }; \
        } \
        cmd.pipelineBarrier( \
            vk::PipelineStageFlagBits::eAllCommands, \
            vk::PipelineStageFlagBits::eAllCommands, \
            {}, {}, {}, \
            {barrier_count, barriers} \
        ); \
    }

#define reservoir_barrier(r, happens_before, happens_after) \
    { \
        vk::ImageMemoryBarrier barriers[4]; \
        uint32_t barrier_count = 0; \
        if(r.ris_data.has_value()) \
        { \
            barriers[barrier_count++] = vk::ImageMemoryBarrier{ \
                happens_before, \
                happens_after, \
                vk::ImageLayout::eGeneral, \
                vk::ImageLayout::eGeneral, \
                VK_QUEUE_FAMILY_IGNORED, \
                VK_QUEUE_FAMILY_IGNORED, \
                r.ris_data->get_image(dev->id), \
                { \
                    vk::ImageAspectFlagBits::eColor, \
                    0, VK_REMAINING_MIP_LEVELS, \
                    0, VK_REMAINING_ARRAY_LAYERS \
                } \
            }; \
        } \
        if(r.reconnection_data.has_value()) \
        { \
            barriers[barrier_count++] = vk::ImageMemoryBarrier{ \
                happens_before, \
                happens_after, \
                vk::ImageLayout::eGeneral, \
                vk::ImageLayout::eGeneral, \
                VK_QUEUE_FAMILY_IGNORED, \
                VK_QUEUE_FAMILY_IGNORED, \
                r.reconnection_data->get_image(dev->id), \
                { \
                    vk::ImageAspectFlagBits::eColor, \
                    0, VK_REMAINING_MIP_LEVELS, \
                    0, VK_REMAINING_ARRAY_LAYERS \
                } \
            }; \
        } \
        if(r.reconnection_radiance.has_value()) \
        { \
            barriers[barrier_count++] = vk::ImageMemoryBarrier{ \
                happens_before, \
                happens_after, \
                vk::ImageLayout::eGeneral, \
                vk::ImageLayout::eGeneral, \
                VK_QUEUE_FAMILY_IGNORED, \
                VK_QUEUE_FAMILY_IGNORED, \
                r.reconnection_radiance->get_image(dev->id), \
                { \
                    vk::ImageAspectFlagBits::eColor, \
                    0, VK_REMAINING_MIP_LEVELS, \
                    0, VK_REMAINING_ARRAY_LAYERS \
                } \
            }; \
        } \
        if(r.rng_seeds.has_value()) \
        { \
            barriers[barrier_count++] = vk::ImageMemoryBarrier{ \
                happens_before, \
                happens_after, \
                vk::ImageLayout::eGeneral, \
                vk::ImageLayout::eGeneral, \
                VK_QUEUE_FAMILY_IGNORED, \
                VK_QUEUE_FAMILY_IGNORED, \
                r.rng_seeds->get_image(dev->id), \
                { \
                    vk::ImageAspectFlagBits::eColor, \
                    0, VK_REMAINING_MIP_LEVELS, \
                    0, VK_REMAINING_ARRAY_LAYERS \
                } \
            }; \
        } \
        cmd.pipelineBarrier( \
            vk::PipelineStageFlagBits::eAllCommands, \
            vk::PipelineStageFlagBits::eAllCommands, \
            {}, {}, {}, \
            {barrier_count, barriers} \
        ); \
    }

#define image_barrier(cmd, tex, happens_before, happens_after) \
    { \
        vk::ImageMemoryBarrier barrier{ \
            happens_before, \
            happens_after, \
            vk::ImageLayout::eGeneral, \
            vk::ImageLayout::eGeneral, \
            VK_QUEUE_FAMILY_IGNORED, \
            VK_QUEUE_FAMILY_IGNORED, \
            tex->get_image(dev->id), \
            { \
                vk::ImageAspectFlagBits::eColor, \
                0, VK_REMAINING_MIP_LEVELS, \
                0, VK_REMAINING_ARRAY_LAYERS \
            } \
        }; \
        cmd.pipelineBarrier( \
            vk::PipelineStageFlagBits::eAllCommands, \
            vk::PipelineStageFlagBits::eAllCommands, \
            {}, {}, {}, \
            barrier \
        ); \
    }


    for(uint32_t i = 0; i < opt.passes; ++i)
    {
        if(i != 0)
        {
            reservoir_textures& in_reservoir_data = reservoir_data[reservoir_data_parity];
            reservoir_textures& out_reservoir_data = reservoir_data[1-reservoir_data_parity];
            reservoir_barrier(in_reservoir_data, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
            reservoir_barrier(out_reservoir_data, vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eShaderWrite);
        }

        if(i == 0 || opt.do_canonical_samples_for_passes)
        {
            record_canonical_pass(cmd, frame_index, i);
        }

        record_spatial_pass(cmd, frame_index, i == opt.passes-1, i);
    }

    stage_timer.end(cmd, dev->id, frame_index);
    end_graphics(cmd, frame_index);

    if(opt.accumulate)
        accumulated_samples++;
    valid_history_frame = dev->ctx->get_frame_counter();
}

#define BIND_RESERVOIRS \
    if(out_reservoir_data.ris_data.has_value()) \
        set.set_image("out_reservoir_ris_data_tex", *out_reservoir_data.ris_data); \
    if(out_reservoir_data.reconnection_data.has_value()) \
        set.set_image("out_reservoir_reconnection_data_tex", *out_reservoir_data.reconnection_data); \
    if(out_reservoir_data.reconnection_radiance.has_value()) \
        set.set_image("out_reservoir_reconnection_radiance_tex", *out_reservoir_data.reconnection_radiance); \
    if(out_reservoir_data.rng_seeds.has_value()) \
        set.set_image("out_reservoir_rng_seeds_tex", *out_reservoir_data.rng_seeds); \
    if(in_reservoir_data.ris_data.has_value()) \
        set.set_image("in_reservoir_ris_data_tex", *in_reservoir_data.ris_data); \
    if(in_reservoir_data.reconnection_data.has_value()) \
        set.set_image("in_reservoir_reconnection_data_tex", *in_reservoir_data.reconnection_data); \
    if(in_reservoir_data.reconnection_radiance.has_value()) \
        set.set_image("in_reservoir_reconnection_radiance_tex", *in_reservoir_data.reconnection_radiance); \
    if(in_reservoir_data.rng_seeds.has_value()) \
        set.set_image("in_reservoir_rng_seeds_tex", *in_reservoir_data.rng_seeds); \

void restir_stage::record_canonical_pass(vk::CommandBuffer cmd, uint32_t /*frame_index*/, int pass_index)
{
    restir_config config = get_restir_config(opt, current_buffers, *scene_data);
    reservoir_textures& in_reservoir_data = reservoir_data[reservoir_data_parity];
    reservoir_textures& out_reservoir_data = reservoir_data[1-reservoir_data_parity];

    uvec2 size = current_buffers.albedo.size;

    { // CANONICAL
        canonical_set.set_image("out_color", *cached_sample_color);
        auto& set = canonical_set;
        BIND_RESERVOIRS
        USED_BUFFERS

        canonical.bind(cmd);
        canonical.push_descriptors(cmd, canonical_set, 0);
        canonical.set_descriptors(cmd, scene_data->get_descriptors(), 0, 1);

        canonical_push_constant_buffer pc;
        pc.config = config;
        pc.camera_index = 0;
        pc.sample_index = dev->ctx->get_frame_counter() * opt.passes + pass_index;
        pc.first_pass = pass_index == 0;

        canonical.push_constants(cmd, pc);
        canonical.trace_rays(cmd, uvec3(size, 1));

        reservoir_barrier(
            out_reservoir_data,
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite
        );
        sample_color_barrier(
            vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite
        );
    }

    uint64_t frame_counter = dev->ctx->get_frame_counter();
    if(pass_index == 0 && opt.temporal_reuse && valid_history_frame+1 == frame_counter ? 1 : 0)
    { // TEMPORAL
        temporal_set.set_image(dev->id, "motion_tex", {{gbuf_sampler.get_sampler(dev->id), current_buffers.screen_motion.view, vk::ImageLayout::eShaderReadOnlyOptimal}});
        temporal_set.set_image("out_color", *cached_sample_color);

        auto& set = temporal_set;
        BIND_RESERVOIRS
        USED_BUFFERS

        temporal.bind(cmd);
        temporal.push_descriptors(cmd, temporal_set, 0);
        temporal.set_descriptors(cmd, scene_data->get_descriptors(), 0, 1);
        // TODO: Temporal tables!
        //temporal.set_descriptors(cmd, scene_data->get_temporal_tables_descriptor_set(), 0, 2);

        temporal_push_constant_buffer pc;
        pc.config = config;
        pc.search_base_radius = opt.temporal_search_base_radius;
        pc.search_widening = opt.temporal_search_widening;
        pc.camera_index = 0;
        pc.sample_index = frame_counter * opt.passes + pass_index;

        uint32_t tmp_seed = pc.sample_index;
        pc.jitter = clamp(vec2(
            generate_uniform_random(tmp_seed),
            generate_uniform_random(tmp_seed)
        ), vec2(0.05f), vec2(0.95f))-0.5f;
        pc.permutation = opt.temporal_permutation;

        temporal.push_constants(cmd, pc);
        temporal.trace_rays(cmd, uvec3(size, 1));

        reservoir_barrier(
            out_reservoir_data,
            vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite
        );
        sample_color_barrier(
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite
        );
    }

    reservoir_data_parity = 1-reservoir_data_parity;
}

void restir_stage::record_spatial_pass(vk::CommandBuffer cmd, uint32_t /*frame_index*/, bool /*final_pass*/, int pass_index)
{
    restir_config config = get_restir_config(opt, current_buffers, *scene_data);
    reservoir_textures& in_reservoir_data = reservoir_data[reservoir_data_parity];
    reservoir_textures& out_reservoir_data = reservoir_data[1-reservoir_data_parity];

    uvec2 size = current_buffers.albedo.size;

    if(opt.spatial_samples > 0)
    {
        selection_set.set_image("spatial_selection", *selection_data);

        auto& set = selection_set;
        BIND_RESERVOIRS
        USED_BUFFERS

        selection.bind(cmd);
        selection.push_descriptors(cmd, selection_set, 0);
        selection.set_descriptors(cmd, scene_data->get_descriptors(), 0, 1);

        selection_push_constant_buffer pc;
        pc.config = config;
        pc.camera_index = 0;
        pc.sample_index = dev->ctx->get_frame_counter() * opt.passes + pass_index;
        pc.display_size = size;

        selection.push_constants(cmd, pc);
        uvec3 wg = uvec3((ivec2(pc.display_size) + selection_tile_size-1) / selection_tile_size, 1);
        cmd.dispatch(wg.x, wg.y, wg.z);
        image_barrier(
            cmd, selection_data,
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead
        );
    }

    if(opt.spatial_samples > 0)
    {
        spatial_trace_set.set_image("spatial_selection", *selection_data);
        spatial_trace_set.set_image("spatial_candidates", *spatial_candidate_color);
        spatial_trace_set.set_image("mis_data", *spatial_mis_data);

        auto& set = spatial_trace_set;
        BIND_RESERVOIRS
        USED_BUFFERS

        spatial_trace.bind(cmd);
        spatial_trace.push_descriptors(cmd, spatial_trace_set, 0);
        spatial_trace.set_descriptors(cmd, scene_data->get_descriptors(), 0, 1);

        spatial_trace_push_constant_buffer pc;
        pc.config = config;
        pc.camera_index = 0;
        pc.sample_index = dev->ctx->get_frame_counter() * opt.passes + pass_index;

        spatial_trace.push_constants(cmd, pc);
        spatial_trace.trace_rays(cmd, uvec3(size, opt.spatial_samples));
    }

    if(opt.spatial_samples > 0)
    {
        image_barrier(
            cmd, spatial_mis_data,
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead
        );
        image_barrier(
            cmd, spatial_candidate_color,
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead
        );
    }

    {
        if(opt.spatial_samples > 0)
        {
            spatial_gather_set.set_image("spatial_selection", *selection_data);
            spatial_gather_set.set_image("spatial_candidates", *spatial_candidate_color);
            spatial_gather_set.set_image("mis_data", *spatial_mis_data);
        }

        if(opt.demodulated_output)
        {
            spatial_gather_set.set_image(dev->id, "out_diffuse", {{{}, current_buffers.diffuse.view, vk::ImageLayout::eGeneral}});
            spatial_gather_set.set_image(dev->id, "out_reflection", {{{}, current_buffers.reflection.view, vk::ImageLayout::eGeneral}});
        }
        else
        {
            spatial_gather_set.set_image("out_diffuse", *cached_sample_color);
            spatial_gather_set.set_image(dev->id, "out_reflection", {{{}, current_buffers.color.view, vk::ImageLayout::eGeneral}});
        }

        auto& set = spatial_gather_set;
        BIND_RESERVOIRS
        USED_BUFFERS

        spatial_gather.bind(cmd);
        spatial_gather.push_descriptors(cmd, spatial_gather_set, 0);
        spatial_gather.set_descriptors(cmd, scene_data->get_descriptors(), 0, 1);

        spatial_gather_push_constant_buffer pc;
        pc.config = config;
        pc.display_size = size;
        pc.camera_index = 0;
        pc.sample_index = dev->ctx->get_frame_counter() * opt.passes + pass_index;
        pc.accumulated_samples = accumulated_samples;
        pc.initialize_output = (opt.accumulate && pass_index == 0) || !opt.accumulate ? 1 : 0;
        pc.accumulate_color = opt.accumulate || pass_index == int(opt.passes)-1 ? 1 : 0;
        pc.update_sample_color = pass_index != int(opt.passes)-1 ? 1 : 0;

        spatial_gather.push_constants(cmd, pc);
        uvec3 wg = uvec3((size+15u)/16u, 1);
        cmd.dispatch(wg.x, wg.y, wg.z);
    }

    reservoir_data_parity = 1-reservoir_data_parity;
}

}
