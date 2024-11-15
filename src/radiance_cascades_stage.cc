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
    cascade_sampler(dev, vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerMipmapMode::eNearest, 0, false, false, false, 0.0f),
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

    if(this->opt.t0 < 0)
    {
        vec3 extent = opt.volume.max - opt.volume.min;
        this->opt.t0 = length(extent) / ((1<<(opt.log2_resolution+1))-1);
    }

    for(uint32_t cascade = 0; cascade <= opt.log2_resolution; ++cascade)
    {
        size_t cascade_size = 1<<(opt.log2_resolution-cascade);
        size_t resolution = 1<<(cascade+1);

        cascades.emplace_back(
            device_mask(dev),
            uvec2(cascade_size*resolution),
            (unsigned)cascade_size,
            //vk::Format::eR16Sfloat,
            vk::Format::eR16G16B16A16Sfloat,
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
                uvec2(cascade_size*resolution),
                (unsigned)cascade_size,
                //vk::Format::eR16Sfloat,
                vk::Format::eR16G16B16A16Sfloat,
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
    return opt.log2_resolution+1;
}

uvec3 radiance_cascades_stage::get_cascade_size(int cascade) const
{
    size_t cascade_size = 1<<(opt.log2_resolution-cascade);
    size_t resolution = 1<<(cascade+1);
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
                opt.log2_resolution+1
            );
        }
    );

    vk::CommandBuffer cb = begin_compute();
    stage_timer.begin(cb, dev->id, frame_index);

    cascades_metadata.upload(dev->id, frame_index, cb);

    std::vector<texture>* next_cascades = &cascades;
    //std::vector<texture>* prev_cascades = nullptr;

    if(opt.recursive || (opt.jitter_rays && opt.temporal_ratio < 1.0f))
    {
        next_cascades = (frame_index&1) ? &cascades : &alt_cascades;
        //prev_cascades = (frame_index&1) ? &alt_cascades : &cascades;
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
            (*next_cascades)[i].get_array_image_view(dev->id),
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

    float length = opt.t0;
    for(uint32_t cascade = 0; cascade <= opt.log2_resolution; ++cascade)
    {
        texture& target = (*next_cascades)[cascade];
        trace_desc.set_image(dev->id, "cascade_target", {{{}, target.get_array_image_view(dev->id), vk::ImageLayout::eGeneral}});
        trace.push_descriptors(cb, trace_desc, 0);

        pc.cascade = cascade;
        pc.cascade_count = get_cascade_count();
        pc.interval_start = pc.interval_end;
        // Last iteration gets to cover the entire world.
        pc.interval_end = cascade == opt.log2_resolution ? 1e9 : pc.interval_start + length;
        pc.base_offset = pvec4(opt.volume.min + vec3(pc.xyz_step), 0);
        pc.xyz_step *= 2.0f;
        length *= 2.0f;

        trace.push_constants(cb, pc);

        size_t cascade_size = 1<<(opt.log2_resolution-cascade);
        size_t resolution = 1<<(cascade+1);
        uvec3 wg = uvec3(uvec2(cascade_size * resolution+7u)/8u, cascade_size);
        cb.dispatch(wg.x, wg.y, wg.z);
    }

    //==========================================================================
    // Gather pass - fills importance values for missed rays.
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

    for(uint32_t i = 1; i <= opt.log2_resolution; ++i)
    {
        uint32_t prev_cascade = opt.log2_resolution-i+1;
        uint32_t cur_cascade = opt.log2_resolution-i;
        texture& prev_target = (*next_cascades)[prev_cascade];
        texture& cur_target = (*next_cascades)[cur_cascade];
        gather_desc.set_image(dev->id, "prev_cascade", {{{}, prev_target.get_array_image_view(dev->id), vk::ImageLayout::eGeneral}});
        gather_desc.set_image(dev->id, "cur_cascade", {{{}, cur_target.get_array_image_view(dev->id), vk::ImageLayout::eGeneral}});
        gather.push_descriptors(cb, gather_desc, 0);

        size_t cascade_size = 1<<(opt.log2_resolution-cur_cascade);
        size_t resolution = 1<<(cur_cascade+1);
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
}

}
