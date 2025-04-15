#include "radiance_cascades_stage.hh"
#include "shadow_map.hh"
#include "misc.hh"

namespace
{
using namespace tr;

struct trace_push_constant_buffer
{
    pvec4 base_offset;
    pvec4 xyz_step;
    pvec2 jitter;
    int cascade;
    int cascade_count;
    float interval_start;
    float interval_end;
    int c0_angular_resolution;
    int cascade_size;
    int has_history;
};

struct gather_push_constant_buffer
{
    int cascade;
    int cascade_count;
    int c0_angular_resolution;
    float blend_ratio;
    int cascade_size;
    int carry_from_previous;
};

struct live_counter_push_constant_buffer
{
    pvec4 base_offset;
    pvec4 xyz_step;
    int cascade;
    int cascade_count;
    float interval_end;
    int cascade_size;
    uint index_buffer_size;
};

struct live_dispatcher_push_constant_buffer
{
    int cascade;
    int cascade_count;
    int c0_angular_resolution;
};

struct cascade_metadata_buffer
{
    pvec4 aabb_min;
    pvec4 aabb_max;
    // size.x = c0 width (x)
    // size.y = c0 height (y)
    // size.z = c0 depth (z)
    // size.w = cascade count
    pivec4 size;
    int c0_angular_resolution;
};

}

namespace tr
{

radiance_cascades_stage::radiance_cascades_stage(
    device& dev,
    scene_stage& ss,
    const options& opt
):  single_device_stage(dev),
    ss(&ss),
    trace_desc(dev),
    gather_desc(dev),
    live_counter_desc(dev),
    live_dispatcher_desc(dev),
    cascade_descriptors(dev),
    //cascade_sampler(dev, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerMipmapMode::eNearest, 0, false, false, false, 0.0f),
    cascade_sampler(dev, vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerMipmapMode::eNearest, 0, false, false, false, 0.0f),
    trace(dev),
    gather(dev),
    live_counter(dev),
    live_dispatcher(dev),
    opt(opt),
    prev_cascades_valid(false),
    stage_timer(dev, "radiance cascade update"),
    trace_timer(dev, "radiance cascade trace"),
    gather_timer(dev, "radiance cascade gather"),
    live_counter_timer(dev, "radiance cascade live count"),
    history_frames(0),
    cascades_metadata(dev, sizeof(cascade_metadata_buffer), vk::BufferUsageFlagBits::eUniformBuffer)
{
    bool has_prev_cascades =
        opt.recursive || (opt.jitter_rays && opt.temporal_ratio < 1.0f);
    descriptor_set& scene_ds = ss.get_descriptors();
    descriptor_set& raster_scene_ds = ss.get_raster_descriptors();

    cascade_descriptors.add("radiance_cascades", {0, vk::DescriptorType::eCombinedImageSampler, 16, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    cascade_descriptors.add("radiance_cascades_visibility", {1, vk::DescriptorType::eCombinedImageSampler, 16, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    cascade_descriptors.add("radiance_cascade_metadata", {2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});

    vec3 extent = opt.volume.max - opt.volume.min;
    float diagonal_range = length(extent);

    size_t total_probes = 0;
    for(uint32_t cascade = 0; cascade <= opt.log2_resolution; ++cascade)
    {
        size_t cascade_size = 1<<(opt.log2_resolution-cascade);
        size_t resolution = opt.c0_probe_resolution << cascade;

        vec2 interval = get_cascade_interval(cascade);
        // Cascade starting distance is out of cascade volume
        // => no point in allocating or rendering the rest of the layers.
        if(interval[0] > diagonal_range)
            break;

        total_probes += cascade_size * cascade_size * cascade_size;

        cascades.emplace_back(
            device_mask(dev),
            //uvec3(cascade_size*resolution, cascade_size*resolution, cascade_size),
            uvec2(cascade_size*resolution),
            cascade_size,
            //vk::Format::eR16Sfloat,
            vk::Format::eR32Sfloat,
            0,
            nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage,
            vk::ImageLayout::eGeneral
        );
        cascades_visibility.emplace_back(
            device_mask(dev),
            //uvec3(cascade_size*resolution, cascade_size*resolution, cascade_size),
            uvec2(cascade_size*resolution),
            cascade_size,
            vk::Format::eR8Unorm,
            0,
            nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage,
            vk::ImageLayout::eGeneral
        );
        if(has_prev_cascades)
        {
            alt_cascades.emplace_back(
                device_mask(dev),
                //uvec3(cascade_size*resolution, cascade_size*resolution, cascade_size),
                uvec2(cascade_size*resolution),
                cascade_size,
                //vk::Format::eR16Sfloat,
                vk::Format::eR32Sfloat,
                0,
                nullptr,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage,
                vk::ImageLayout::eGeneral
            );
            alt_cascades_visibility.emplace_back(
                device_mask(dev),
                //uvec3(cascade_size*resolution, cascade_size*resolution, cascade_size),
                uvec2(cascade_size*resolution),
                cascade_size,
                vk::Format::eR8Unorm,
                0,
                nullptr,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage,
                vk::ImageLayout::eGeneral
            );
        }
    }

    vk::BufferCreateInfo bufferInfo;
    bufferInfo.size = 16 * sizeof(uint) + total_probes * sizeof(puvec3);
    bufferInfo.usage =
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst;
    dispatch_info_buffer = create_buffer(dev, bufferInfo, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    bufferInfo.size = 32 * sizeof(uvec4);
    bufferInfo.usage |= vk::BufferUsageFlagBits::eIndirectBuffer;
    dispatch_size_buffer = create_buffer(dev, bufferInfo, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    dispatch_index_max_size = total_probes;

    std::map<std::string, std::string> defines;
    add_defines(defines);

    {
        shader_source src("shader/radiance_cascades_trace.comp", defines);
        trace_desc.add(src);
        trace.init(src, {&trace_desc, &scene_ds, &raster_scene_ds, &cascade_descriptors});
    }

    {
        shader_source src("shader/radiance_cascades_gather.comp", defines);
        gather_desc.add(src);
        gather.init(src, {&gather_desc});
    }

    {
        shader_source src("shader/radiance_cascades_live_counter.comp", defines);
        live_counter_desc.add(src);
        live_counter.init(src, {&live_counter_desc});
    }

    {
        shader_source src("shader/radiance_cascades_live_dispatcher.comp", defines);
        live_dispatcher_desc.add(src);
        live_dispatcher.init(src, {&live_dispatcher_desc});
    }
}

descriptor_set& radiance_cascades_stage::get_descriptors()
{
    return cascade_descriptors;
}

size_t radiance_cascades_stage::get_cascade_count() const
{
    return cascades.size();
}

float radiance_cascades_stage::get_cascade_t0(int cascade) const
{
    vec3 extent = opt.volume.max - opt.volume.min;
    float h0 = max(extent.x, max(extent.y, extent.z))/float(1<<opt.log2_resolution);

    const vec2 octahedral_theta_table[] = {
        vec2(7.853982e-01, 1.570796e+00),
        vec2(3.217506e-01, 9.553166e-01),
        vec2(1.418971e-01, 5.148060e-01),
        vec2(6.656816e-02, 2.631283e-01),
        vec2(3.224688e-02, 1.323247e-01),
        vec2(1.587168e-02, 6.625893e-02),
        vec2(7.873853e-03, 3.314159e-02),
        vec2(3.921549e-03, 1.657231e-02),
        vec2(1.956945e-03, 8.286344e-03),
        vec2(9.775168e-04, 4.143196e-03),
        vec2(4.885197e-04, 2.071601e-03),
        vec2(2.442002e-04, 1.035801e-03),
        vec2(1.220852e-04, 5.179005e-04),
        vec2(6.103888e-05, 2.589502e-04)
    };

    return (1<<cascade) * h0 / sin(octahedral_theta_table[cascade].x);

    /*
    if(cascade == 0) return 0.5f * h0 * sqrt(3.0f);

    // Relative error: (looks low-res but more consistent in motion)
    //float spatial_resolution = max(extent.x, max(extent.y, extent.z))/float(1<<(opt.log2_resolution-cascade));

    // Constant error: (original radiance cascades paper?)
    //float spatial_resolution = max(extent.x, max(extent.y, extent.z))/float(1<<opt.log2_resolution);
    //size_t resolution = opt.c0_probe_resolution << cascade;
    //return (spatial_resolution * resolution) / M_PI;
    return (1<<((cascade-1)*2)) * h0 * opt.c0_probe_resolution * 2.0f * M_PI;
    */
}

vec2 radiance_cascades_stage::get_cascade_interval(int cascade) const
{
    return vec2(get_cascade_t0(cascade), get_cascade_t0(cascade+1));
}

uvec3 radiance_cascades_stage::get_cascade_size(int cascade) const
{
    size_t cascade_size = 1<<(opt.log2_resolution-cascade);
    size_t resolution = opt.c0_probe_resolution << cascade;
    return uvec3(cascade_size*resolution, cascade_size*resolution, cascade_size);
}

void radiance_cascades_stage::add_defines(std::map<std::string, std::string>& defines) const
{
    defines["RC_C0_ANGULAR_RESOLUTION"] = std::to_string(opt.c0_probe_resolution);
    defines["RC_C0_SPATIAL_RESOLUTION"] = std::to_string(1<<opt.log2_resolution);
    defines["RC_CASCADE_COUNT"] = std::to_string(get_cascade_count());
}

void radiance_cascades_stage::update(uint32_t frame_index)
{
    clear_commands();

    cascades_metadata.map<cascade_metadata_buffer>(
        frame_index, [&](cascade_metadata_buffer* data){
            data->aabb_min = vec4(opt.volume.min, 0);
            data->aabb_max = vec4(opt.volume.max, 0);
            data->size = pivec4(
                1<<opt.log2_resolution,
                1<<opt.log2_resolution,
                1<<opt.log2_resolution,
                get_cascade_count()
            );
            data->c0_angular_resolution = opt.c0_probe_resolution;
        }
    );

    vk::CommandBuffer cb = begin_compute();
    stage_timer.begin(cb, dev->id, frame_index);

    cascades_metadata.upload(dev->id, frame_index, cb);

    cb.fillBuffer(dispatch_info_buffer, 0, 16 * sizeof(uint), 0);
    cb.fillBuffer(dispatch_size_buffer, 0, VK_WHOLE_SIZE, 0);

    std::vector<texture>* next_cascades = &cascades;
    std::vector<texture>* prev_cascades = nullptr;
    std::vector<texture>* next_cascades_visibility = &cascades_visibility;
    std::vector<texture>* prev_cascades_visibility = nullptr;

    if(opt.recursive || (opt.jitter_rays && opt.temporal_ratio < 1.0f))
    {
        next_cascades = (frame_index&1) ? &cascades : &alt_cascades;
        prev_cascades = (frame_index&1) ? &alt_cascades : &cascades;
        next_cascades_visibility = (frame_index&1) ? &cascades_visibility : &alt_cascades_visibility;
        prev_cascades_visibility = (frame_index&1) ? &alt_cascades_visibility : &cascades_visibility;
    }

    std::vector<vk::ImageMemoryBarrier> image_barriers;
    std::vector<vk::BufferMemoryBarrier> buffer_barriers;
    std::vector<vk::DescriptorImageInfo> dii;
    std::vector<vk::DescriptorImageInfo> dii_visibility;
    for(size_t i = 0; i < next_cascades->size(); ++i)
    {
        image_barriers.push_back(vk::ImageMemoryBarrier(
            {}, vk::AccessFlagBits::eShaderWrite,
            vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            (*next_cascades)[i].get_image(dev->id),
            {vk::ImageAspectFlagBits::eColor, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}
        ));
        image_barriers.push_back(vk::ImageMemoryBarrier(
            {}, vk::AccessFlagBits::eShaderWrite,
            vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            (*next_cascades_visibility)[i].get_image(dev->id),
            {vk::ImageAspectFlagBits::eColor, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}
        ));

        buffer_barriers.push_back(vk::BufferMemoryBarrier(
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
            0, 0,
            *dispatch_info_buffer,
            0, VK_WHOLE_SIZE
        ));
        buffer_barriers.push_back(vk::BufferMemoryBarrier(
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
            0, 0,
            *dispatch_size_buffer,
            0, VK_WHOLE_SIZE
        ));

        dii.push_back(vk::DescriptorImageInfo{
            cascade_sampler.get_sampler(dev->id),
            (*next_cascades)[i].get_array_image_view(dev->id),
            vk::ImageLayout::eGeneral
        });
        dii_visibility.push_back(vk::DescriptorImageInfo{
            cascade_sampler.get_sampler(dev->id),
            (*next_cascades_visibility)[i].get_array_image_view(dev->id),
            vk::ImageLayout::eGeneral
        });
    }

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eAllCommands,
        {}, {}, buffer_barriers, image_barriers
    );
    if(history_frames == 0)
    {
        cascade_descriptors.reset(cascade_descriptors.get_mask(), 1);
        cascade_descriptors.set_image(dev->id, 0, "radiance_cascades", std::move(dii));
        cascade_descriptors.set_image(dev->id, 0, "radiance_cascades_visibility", std::move(dii_visibility));
        cascade_descriptors.set_buffer(0, "radiance_cascade_metadata", cascades_metadata);
    }

    //==========================================================================
    // Live counting & dispatch generation
    //==========================================================================
    // Counts the number of live probes on each cascade, and generates the
    // necessary data for indirect dispatching the computation for those probes.
    live_counter_timer.begin(cb, dev->id, frame_index);
    for(int cascade = get_cascade_count()-1; cascade >= 0; --cascade)
    {
        {
            live_counter.bind(cb);
            live_counter_desc.set_image(dev->id, "distance_field", {{
                {},
                opt.distance_field->get_mip_image_view(dev->id, cascade),
                vk::ImageLayout::eGeneral
            }});
            live_counter_desc.set_buffer(dev->id, "dispatch_info", {{*dispatch_info_buffer, 0, VK_WHOLE_SIZE}});
            live_counter.push_descriptors(cb, live_counter_desc, 0);

            size_t cascade_size = 1<<(opt.log2_resolution-cascade);

            live_counter_push_constant_buffer pc;

            vec3 half_step = float(1<<cascade) * (opt.volume.max-opt.volume.min)/float(2<<opt.log2_resolution);
            pc.base_offset = pvec4(opt.volume.min + half_step, 0);
            pc.xyz_step = pvec4(half_step * 2.0f, 0.0f);
            pc.cascade = cascade;

            vec2 interval = get_cascade_interval(cascade);
            pc.interval_end = cascade+1 == get_cascade_count() ? 1e9 : interval[1];
            pc.cascade_size = cascade_size;
            pc.cascade_count = get_cascade_count();
            pc.index_buffer_size = dispatch_index_max_size;

            live_counter.push_constants(cb, pc);

            if(cascade == get_cascade_count() - 1)
            {
                size_t parent_cascade_size = max(cascade_size >> 1, size_t(1));
                cb.dispatch(((parent_cascade_size*parent_cascade_size*parent_cascade_size)+7u)/8u, 1,1);
            }
            else
            {
                cb.dispatchIndirect(*dispatch_size_buffer, sizeof(uvec4) * (cascade+1));
            }
        }

        vk::BufferMemoryBarrier barriers[2] = {
            {
                vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eIndirectCommandRead,
                vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eIndirectCommandRead,
                0, 0,
                *dispatch_info_buffer,
                0, VK_WHOLE_SIZE
            },
            {
                vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eIndirectCommandRead,
                vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eIndirectCommandRead,
                0, 0,
                *dispatch_size_buffer,
                0, VK_WHOLE_SIZE
            }
        };
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eAllCommands,
            vk::PipelineStageFlagBits::eAllCommands,
            {}, {}, barriers, {}
        );

        {
            live_dispatcher.bind(cb);
            live_dispatcher_desc.set_buffer(dev->id, "dispatch_size", {{*dispatch_size_buffer, 0, VK_WHOLE_SIZE}});
            live_dispatcher_desc.set_buffer(dev->id, "dispatch_info", {{*dispatch_info_buffer, 0, VK_WHOLE_SIZE}});
            live_dispatcher.push_descriptors(cb, live_dispatcher_desc, 0);
            live_dispatcher_push_constant_buffer pc;
            pc.cascade = cascade;
            pc.cascade_count = get_cascade_count();
            pc.c0_angular_resolution = opt.c0_probe_resolution;
            live_dispatcher.push_constants(cb, pc);
            cb.dispatch(1,1,1);
        }

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eAllCommands,
            vk::PipelineStageFlagBits::eAllCommands,
            {}, {}, barriers, {}
        );
    }
    live_counter_timer.end(cb, dev->id, frame_index);

    //==========================================================================
    // Trace pass - traces rays for radiance intervals
    //==========================================================================
    trace_timer.begin(cb, dev->id, frame_index);
    trace.bind(cb);
    trace.set_descriptors(cb, ss->get_descriptors(), 0, 1);
    trace.set_descriptors(cb, ss->get_raster_descriptors(), 0, 2);
    trace.set_descriptors(cb, cascade_descriptors, 0, 3);

    trace_push_constant_buffer pc;
    pc.jitter = opt.jitter_rays ? r2_noise(vec2(dev->ctx->get_frame_counter())) : vec2(0.5f);
    pc.interval_start = 0;
    pc.interval_end = 0;
    pc.c0_angular_resolution = opt.c0_probe_resolution;
    pc.has_history = history_frames != 0;

    for(uint32_t cascade = 0; cascade < get_cascade_count(); ++cascade)
    {
        texture& target = (*next_cascades)[cascade];
        texture& target_visibility = (*next_cascades_visibility)[cascade];
        trace_desc.set_image(dev->id, "cascade_target", {{{}, target.get_array_image_view(dev->id), vk::ImageLayout::eGeneral}});
        trace_desc.set_image(dev->id, "cascade_target_visibility", {{{}, target_visibility.get_array_image_view(dev->id), vk::ImageLayout::eGeneral}});
        //trace_desc.set_image(dev->id, "distance_field", {{{}, opt.distance_field->get_image_view(dev->id), vk::ImageLayout::eGeneral}});
        trace_desc.set_buffer(dev->id, "dispatch_info", {{*dispatch_info_buffer, 0, VK_WHOLE_SIZE}});
        trace.push_descriptors(cb, trace_desc, 0);

        size_t cascade_size = 1<<(opt.log2_resolution-cascade);

        pc.cascade = cascade;
        pc.cascade_count = get_cascade_count();

        vec2 interval = get_cascade_interval(cascade);
        pc.interval_start = interval[0];
        // Last iteration gets to cover the entire world.
        pc.interval_end = cascade+1 == get_cascade_count() ? 1e9 : interval[1];
        vec3 half_step = float(1<<cascade) * (opt.volume.max-opt.volume.min)/float(2<<opt.log2_resolution);
        pc.base_offset = pvec4(opt.volume.min + half_step, 0);
        pc.xyz_step = pvec4(half_step * 2.0f, 0.0f);
        pc.cascade_size = cascade_size;

        trace.push_constants(cb, pc);

        cb.dispatchIndirect(*dispatch_size_buffer, sizeof(uvec4) * (16+cascade));
    }

    if(history_frames != 0)
    {
        cascade_descriptors.reset(cascade_descriptors.get_mask(), 1);
        cascade_descriptors.set_image(dev->id, 0, "radiance_cascades", std::move(dii));
        cascade_descriptors.set_image(dev->id, 0, "radiance_cascades_visibility", std::move(dii_visibility));
        cascade_descriptors.set_buffer(0, "radiance_cascade_metadata", cascades_metadata);
    }
    trace_timer.end(cb, dev->id, frame_index);

    //==========================================================================
    // Gather pass - fills importance values for missed rays & implements
    // temporal accumulation.
    //==========================================================================
    gather_timer.begin(cb, dev->id, frame_index);
    for(auto& b: image_barriers)
    {
        b.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        b.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
        b.oldLayout = vk::ImageLayout::eGeneral;
    }

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eAllCommands,
        {}, {}, {}, image_barriers
    );

    gather.bind(cb);

    for(uint32_t i = 0; i < get_cascade_count(); ++i)
    {
        uint32_t cur_cascade = get_cascade_count()-1-i;
        uint32_t upper_cascade = cur_cascade+1;
        texture& cur_target = (*next_cascades)[cur_cascade];
        texture& cur_target_visibility = (*next_cascades_visibility)[cur_cascade];
        texture& upper_target = i == 0 ? cur_target : (*next_cascades)[upper_cascade];
        texture& prev_target = prev_cascades ?
            (*prev_cascades)[cur_cascade] : cur_target;
        texture& prev_target_visibility = prev_cascades_visibility ?
            (*prev_cascades_visibility)[cur_cascade] : cur_target_visibility;
        gather_desc.set_image(dev->id, "upper_target", {{{}, upper_target.get_array_image_view(dev->id), vk::ImageLayout::eGeneral}});
        gather_desc.set_image(dev->id, "lower_target", {{{}, cur_target.get_array_image_view(dev->id), vk::ImageLayout::eGeneral}});
        gather_desc.set_image(dev->id, "lower_target_visibility", {{{}, cur_target_visibility.get_array_image_view(dev->id), vk::ImageLayout::eGeneral}});
        gather_desc.set_image(dev->id, "prev_lower_target", {{{}, prev_target.get_array_image_view(dev->id), vk::ImageLayout::eGeneral}});
        gather_desc.set_image(dev->id, "prev_lower_target_visibility", {{{}, prev_target_visibility.get_array_image_view(dev->id), vk::ImageLayout::eGeneral}});
        gather_desc.set_buffer(dev->id, "dispatch_info", {{*dispatch_info_buffer, 0, VK_WHOLE_SIZE}});
        gather.push_descriptors(cb, gather_desc, 0);

        size_t cascade_size = 1<<(opt.log2_resolution-cur_cascade);

        gather_push_constant_buffer pc;
        pc.cascade = cur_cascade;
        pc.cascade_count = get_cascade_count();
        pc.c0_angular_resolution = opt.c0_probe_resolution;
        pc.blend_ratio = history_frames == 0 ? 1.0f : max(1.0f/history_frames, opt.temporal_ratio);
        pc.cascade_size = cascade_size;
        pc.carry_from_previous = i != 0 ? 1 : 0;
        gather.push_constants(cb, pc);

        cb.dispatchIndirect(*dispatch_size_buffer, sizeof(uvec4) * (16+cur_cascade));
    }

    // Change layout to sampleable
    for(auto& b: image_barriers)
    {
        b.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
        b.dstAccessMask = {};
        b.newLayout = vk::ImageLayout::eGeneral;
    }

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eAllCommands,
        {}, {}, {}, image_barriers
    );

    gather_timer.end(cb, dev->id, frame_index);
    stage_timer.end(cb, dev->id, frame_index);
    end_compute(cb, frame_index);
    prev_cascades_valid = true;
    history_frames++;
}

}
