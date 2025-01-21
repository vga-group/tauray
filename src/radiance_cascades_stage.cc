#include "radiance_cascades_stage.hh"
#include "misc.hh"

namespace
{
using namespace tr;

struct trace_push_constant_buffer
{
    pvec4 base_offset;
    pvec4 xyz_step;
    int cascade;
    int cascade_count;
    float interval_start;
    float interval_end;
    int c0_angular_resolution;
    int cascade_size;
};

struct gather_push_constant_buffer
{
    int cascade;
    int c0_angular_resolution;
    float blend_ratio;
    int cascade_size;
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
    cascade_descriptors(dev),
    cascade_sampler(dev, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerMipmapMode::eNearest, 0, false, false, false, 0.0f),
    trace(dev),
    gather(dev),
    opt(opt),
    prev_cascades_valid(false),
    stage_timer(dev, "radiance cascade update"),
    cascades_metadata(dev, sizeof(cascade_metadata_buffer), vk::BufferUsageFlagBits::eUniformBuffer)
{
    bool has_prev_cascades =
        opt.recursive || (opt.jitter_rays && opt.temporal_ratio < 1.0f);
    descriptor_set& scene_ds = ss.get_descriptors();
    descriptor_set& raster_scene_ds = ss.get_raster_descriptors();

    {
        shader_source src("shader/radiance_cascades_trace.comp");
        trace_desc.add(src);
        trace.init(src, {&trace_desc, &scene_ds, &raster_scene_ds});
    }

    {
        shader_source src("shader/radiance_cascades_gather.comp");
        gather_desc.add(src);
        gather.init(src, {&gather_desc});
    }

    vec3 extent = opt.volume.max - opt.volume.min;
    float diagonal_range = length(extent);

    for(uint32_t cascade = 0; cascade <= opt.log2_resolution; ++cascade)
    {
        size_t cascade_size = 1<<(opt.log2_resolution-cascade);
        size_t resolution = opt.c0_probe_resolution << cascade;

        vec2 interval = get_cascade_interval(cascade);
        // Cascade starting distance is out of cascade volume
        // => no point in allocating or rendering the rest of the layers.
        if(interval[0] > diagonal_range)
            break;

        cascades.emplace_back(
            device_mask(dev),
            uvec3(cascade_size*resolution, cascade_size*resolution, cascade_size),
            vk::Format::eR16G16Sfloat,
            0,
            nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage,
            vk::ImageLayout::eShaderReadOnlyOptimal
        );
        if(has_prev_cascades)
        {
            alt_cascades.emplace_back(
                device_mask(dev),
                uvec3(cascade_size*resolution, cascade_size*resolution, cascade_size),
                vk::Format::eR16G16Sfloat,
                0,
                nullptr,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage,
                vk::ImageLayout::eShaderReadOnlyOptimal
            );
        }
    }

    cascade_descriptors.add("radiance_cascades", {0, vk::DescriptorType::eCombinedImageSampler, 16, vk::ShaderStageFlagBits::eAll, nullptr}, vk::DescriptorBindingFlagBits::ePartiallyBound);
    cascade_descriptors.add("radiance_cascade_metadata", {1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
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
    if(cascade == 0) return 0;

    vec3 extent = opt.volume.max - opt.volume.min;

    float h0 = max(extent.x, max(extent.y, extent.z))/float(1<<opt.log2_resolution);
    return (1<<((cascade-1)*2)) * h0 * opt.c0_probe_resolution * 2.0f * M_PI;

    // Relative error: (looks low-res but more consistent in motion)
    //float spatial_resolution = max(extent.x, max(extent.y, extent.z))/float(1<<(opt.log2_resolution-cascade));

    // Constant error: (original radiance cascades paper?)
    //float spatial_resolution = max(extent.x, max(extent.y, extent.z))/float(1<<opt.log2_resolution);
    //size_t resolution = opt.c0_probe_resolution << cascade;
    //return (spatial_resolution * resolution) / M_PI;
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

    std::vector<texture>* next_cascades = &cascades;
    std::vector<texture>* prev_cascades = nullptr;

    if(opt.recursive || (opt.jitter_rays && opt.temporal_ratio < 1.0f))
    {
        next_cascades = (frame_index&1) ? &cascades : &alt_cascades;
        prev_cascades = (frame_index&1) ? &alt_cascades : &cascades;
    }

    std::vector<vk::ImageMemoryBarrier> barriers;
    std::vector<vk::DescriptorImageInfo> dii;
    for(size_t i = 0; i < next_cascades->size(); ++i)
    {
        barriers.push_back(vk::ImageMemoryBarrier(
            {}, vk::AccessFlagBits::eShaderWrite,
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            (*next_cascades)[i].get_image(dev->id),
            {vk::ImageAspectFlagBits::eColor, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}
        ));

        dii.push_back(vk::DescriptorImageInfo{
            cascade_sampler.get_sampler(dev->id),
            (*next_cascades)[i].get_image_view(dev->id),
            vk::ImageLayout::eShaderReadOnlyOptimal
        });
    }

    cascade_descriptors.reset(cascade_descriptors.get_mask(), 1);
    cascade_descriptors.set_image(dev->id, 0, "radiance_cascades", std::move(dii));
    cascade_descriptors.set_buffer(0, "radiance_cascade_metadata", cascades_metadata);

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, {}, {}, barriers
    );

    //==========================================================================
    // Trace pass - traces rays for radiance intervals
    //==========================================================================
    trace.bind(cb);
    trace.set_descriptors(cb, ss->get_descriptors(), 0, 1);
    trace.set_descriptors(cb, ss->get_raster_descriptors(), 0, 2);

    trace_push_constant_buffer pc;
    pc.xyz_step = pvec4((opt.volume.max-opt.volume.min)/float(2<<opt.log2_resolution), 0);
    pc.interval_start = 0;
    pc.interval_end = 0;
    pc.c0_angular_resolution = opt.c0_probe_resolution;

    for(uint32_t cascade = 0; cascade < get_cascade_count(); ++cascade)
    {
        texture& target = (*next_cascades)[cascade];
        trace_desc.set_image(dev->id, "cascade_target", {{{}, target.get_image_view(dev->id), vk::ImageLayout::eGeneral}});
        trace_desc.set_image(dev->id, "distance_field", {{{}, opt.distance_field->get_image_view(dev->id), vk::ImageLayout::eGeneral}});
        trace.push_descriptors(cb, trace_desc, 0);

        size_t cascade_size = 1<<(opt.log2_resolution-cascade);
        size_t resolution = opt.c0_probe_resolution<<cascade;

        pc.cascade = cascade;
        pc.cascade_count = get_cascade_count();

        vec2 interval = get_cascade_interval(cascade);
        pc.interval_start = interval[0];
        // Last iteration gets to cover the entire world.
        pc.interval_end = cascade+1 == get_cascade_count() ? 1e9 : interval[1];
        pc.base_offset = pvec4(opt.volume.min + vec3(pc.xyz_step), 0);
        pc.xyz_step *= 2.0f;
        pc.cascade_size = cascade_size;

        trace.push_constants(cb, pc);

        uvec3 wg = uvec3(uvec2(cascade_size * resolution+7u)/8u, cascade_size);
        cb.dispatch(wg.x, wg.y, wg.z);
    }

    //==========================================================================
    // Gather pass - fills importance values for missed rays & implements
    // temporal accumulation.
    //==========================================================================
    for(size_t i = 0; i < next_cascades->size(); ++i)
    {
        barriers[i].srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        barriers[i].dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
        barriers[i].oldLayout = vk::ImageLayout::eGeneral;
    }

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, {}, {}, barriers
    );

    gather.bind(cb);

    for(uint32_t i = 1; i < get_cascade_count(); ++i)
    {
        uint32_t cur_cascade = get_cascade_count()-1-i;
        uint32_t upper_cascade = cur_cascade+1;
        texture& upper_target = (*next_cascades)[upper_cascade];
        texture& cur_target = (*next_cascades)[cur_cascade];
        texture& prev_target = prev_cascades ?
            (*prev_cascades)[cur_cascade] : cur_target;
        gather_desc.set_image(dev->id, "upper_target", {{{}, upper_target.get_image_view(dev->id), vk::ImageLayout::eGeneral}});
        gather_desc.set_image(dev->id, "lower_target", {{{}, cur_target.get_image_view(dev->id), vk::ImageLayout::eGeneral}});
        gather_desc.set_image(dev->id, "prev_lower_target", {{{}, prev_target.get_image_view(dev->id), vk::ImageLayout::eGeneral}});
        gather.push_descriptors(cb, gather_desc, 0);

        size_t cascade_size = 1<<(opt.log2_resolution-cur_cascade);
        size_t resolution = opt.c0_probe_resolution<<cur_cascade;

        gather_push_constant_buffer pc;
        pc.cascade = cur_cascade;
        pc.c0_angular_resolution = opt.c0_probe_resolution;
        pc.blend_ratio = 0.0f;
        pc.cascade_size = cascade_size;
        gather.push_constants(cb, pc);

        uvec3 wg = uvec3(uvec2(cascade_size * resolution+7u)/8u, cascade_size);
        cb.dispatch(wg.x, wg.y, wg.z);
    }

    // Change layout to sampleable
    for(size_t i = 0; i < next_cascades->size(); ++i)
    {
        barriers[i].srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
        barriers[i].dstAccessMask = {};
        barriers[i].newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eBottomOfPipe,
        {}, {}, {}, barriers
    );

    stage_timer.end(cb, dev->id, frame_index);
    end_compute(cb, frame_index);
    prev_cascades_valid = true;
}

}
