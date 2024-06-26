#include "sh_compact_stage.hh"
#include "misc.hh"

namespace
{

struct push_constant_buffer
{
    int samples;
    int samples_per_work_item;
};

}

namespace tr
{

sh_compact_stage::sh_compact_stage(
    device& dev,
    texture& inflated_source,
    texture& compacted_output
):  single_device_stage(dev),
    desc(dev),
    comp(dev),
    compact_timer(dev, "SH compact")
{
    shader_source src("shader/sh_compact.comp");
    desc.add(src);
    comp.init(src, {&desc});

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        // Record command buffer
        vk::CommandBuffer cb = begin_compute();
        compact_timer.begin(cb, dev.id, i);

        vk::ImageMemoryBarrier img_barrier(
            {}, vk::AccessFlagBits::eShaderWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            compacted_output.get_image(dev.id),
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
        );

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, {}, img_barrier
        );

        comp.bind(cb);

        desc.set_image(dev.id, "input_sh", {{{}, inflated_source.get_image_view(dev.id), vk::ImageLayout::eGeneral}});
        desc.set_image(dev.id, "output_sh", {{{}, compacted_output.get_image_view(dev.id), vk::ImageLayout::eGeneral}});
        comp.push_descriptors(cb, desc, 0);

        push_constant_buffer control;
        uvec3 src_dim = inflated_source.get_dimensions();
        uvec3 dst_dim = compacted_output.get_dimensions();
        int samples = src_dim.z / dst_dim.z;
        control.samples = samples;
        control.samples_per_work_item = (samples+255)/256;

        comp.push_constants(cb, control);

        cb.dispatch(dst_dim.x, dst_dim.y, dst_dim.z);

        img_barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        img_barrier.dstAccessMask = {};
        img_barrier.oldLayout = vk::ImageLayout::eGeneral;
        img_barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eBottomOfPipe,
            {}, {}, {}, img_barrier
        );

        compact_timer.end(cb, dev.id, i);
        end_compute(cb, i);
    }
}

}
