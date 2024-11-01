#include "rt_camera_stage.hh"
#include "mesh.hh"
#include "shader_source.hh"
#include "placeholders.hh"
#include "scene_stage.hh"
#include "camera.hh"
#include "environment_map.hh"
#include "misc.hh"
#include "texture.hh"
#include "sampler.hh"

namespace
{
using namespace tr;

// This must match the distribution_data_buffer in shader/rt.glsl
struct distribution_data_buffer
{
    uvec2 size;
    unsigned index;
    unsigned count;
    unsigned primary;
    unsigned samples_accumulated;
};

}

namespace tr
{

void rt_camera_stage::get_common_defines(std::map<std::string, std::string>& defines)
{
    rt_stage::get_common_defines(defines);
    defines["CAMERA_PROJECTION_TYPE"] = std::to_string((int)opt.projection);
    defines["DISTRIBUTION_STRATEGY"] = std::to_string((int)opt.distribution.strategy);
}

rt_camera_stage::rt_camera_stage(
    device& dev,
    scene_stage& ss,
    const gbuffer_target& output_target,
    const options& opt,
    const std::string& timer_name,
    unsigned pass_count
):  rt_stage(
        dev, ss, opt,
        timer_name + " ("+ std::to_string(opt.active_viewport_count) +" viewports)",
        pass_count
    ),
    distribution_data(dev, sizeof(distribution_data_buffer), vk::BufferUsageFlagBits::eUniformBuffer),
    opt(opt),
    target(output_target),
    accumulated_samples(0)
{
    sample_count_multiplier = opt.samples_per_pixel;
}

void rt_camera_stage::reset_accumulated_samples()
{
    accumulated_samples = 0;
}

int rt_camera_stage::get_accumulated_samples() const
{
    return accumulated_samples;
}

void rt_camera_stage::reset_distribution_params(distribution_params distribution)
{
    opt.distribution = distribution;
    force_command_buffer_refresh();
}

void rt_camera_stage::update(uint32_t frame_index)
{
    rt_stage::update(frame_index);

    distribution_data.map<distribution_data_buffer>(
        frame_index,
        [&](distribution_data_buffer* duni){
            duni->index = opt.distribution.index;
            duni->size = opt.distribution.size;
            duni->count = opt.distribution.strategy == DISTRIBUTION_SHUFFLED_STRIPS ?
                calculate_shuffled_strips_b(opt.distribution.size) : opt.distribution.count;
            duni->primary = opt.distribution.primary ? 1 : 0;
            duni->samples_accumulated = accumulated_samples;
        }
    );

    scene* cur_scene = ss->get_scene();

    cur_scene->foreach([&](camera& cam, camera_metadata& md){
        if(md.enabled && cam.get_projection_type() != opt.projection)
            throw std::runtime_error(
                "Camera projection type does not match what this pipeline is "
                "configured for"
            );
    });

    accumulated_samples += opt.samples_per_pixel;
}

void rt_camera_stage::get_descriptors(push_descriptor_set& desc)
{
    rt_stage::get_descriptors(desc);
    desc.set_buffer("distribution", distribution_data);

#define TR_GBUFFER_ENTRY(name, ...)\
    desc.set_image(dev->id, #name "_target", {{ \
        {}, target.name ? target.name.view : VK_NULL_HANDLE, vk::ImageLayout::eGeneral\
    }});
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY
}

void rt_camera_stage::record_command_buffer(
    vk::CommandBuffer cb, uint32_t frame_index, uint32_t pass_index,
    bool first_in_command_buffer
){
    std::vector<vk::ImageMemoryBarrier> in_barriers;
    std::vector<vk::ImageMemoryBarrier> out_barriers;

    target.visit([&](const render_target& target) {
        vk::ImageMemoryBarrier barrier(
            {}, vk::AccessFlagBits::eShaderWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            target.image,
            target.get_range()
        );
        if(get_pass_count() > 0)
            barrier.dstAccessMask |= vk::AccessFlagBits::eShaderRead;
        in_barriers.push_back(barrier);

        if(pass_index == get_pass_count() -1)
        {
            // Ensure that we don't get a race condition between passes.
            barrier.srcAccessMask = barrier.dstAccessMask;
            barrier.dstAccessMask = {};
            barrier.oldLayout = vk::ImageLayout::eGeneral;
            barrier.newLayout = target.layout;
        }
        else
        {
            barrier.srcAccessMask = barrier.dstAccessMask;
            barrier.oldLayout = barrier.newLayout = vk::ImageLayout::eGeneral;
        }

        out_barriers.push_back(barrier);
    });

    if(pass_index == 0)
    {
        distribution_data.upload(dev->id, frame_index, cb);
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            {}, {}, {}, in_barriers
        );
    }

    record_command_buffer_pass(
        cb, frame_index, pass_index,
        uvec3(get_ray_count(opt.distribution), opt.active_viewport_count),
        first_in_command_buffer
    );

    if(pass_index == get_pass_count()-1)
    {
        // Last pass, so transition the image to the output layout.
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            vk::PipelineStageFlagBits::eBottomOfPipe,
            {}, {}, {}, out_barriers
        );
    }
    else
    {
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            {}, {}, {}, out_barriers
        );
    }
}

}
