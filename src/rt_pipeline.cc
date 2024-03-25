#include "rt_pipeline.hh"
#include "misc.hh"
#include <map>

namespace tr
{

rt_pipeline::rt_pipeline(device& dev)
: basic_pipeline(dev, vk::PipelineBindPoint::eRayTracingKHR)
{
}

void rt_pipeline::trace_rays(vk::CommandBuffer buf, uvec3 size)
{
    buf.traceRaysKHR(
        &rgen_sbt, &rmiss_sbt, &rchit_sbt, &rcallable_sbt,
        size.x, size.y, size.z
    );
}

void rt_pipeline::init(
    rt_shader_sources src,
    std::vector<tr::descriptor_set_layout*> layout,
    int max_recursion_depth,
    vk::SpecializationInfo specialization
){
    basic_pipeline::init(get_push_constant_ranges(src), layout);
    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> rt_shader_groups;

    if(!src.rgen.data.empty())
    {
        load_shader_module(
            src.rgen,
            vk::ShaderStageFlagBits::eRaygenKHR,
            stages,
            specialization
        );
        rt_shader_groups.push_back({
            vk::RayTracingShaderGroupTypeKHR::eGeneral,
            (uint32_t)stages.size()-1, VK_SHADER_UNUSED_KHR,
            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR
        });
    }

    size_t hit_group_count = src.rhit.size();
    for(size_t i = 0; i < hit_group_count; ++i)
    {
        auto& hg = src.rhit[i];
        uint32_t chit_index = VK_SHADER_UNUSED_KHR;
        uint32_t ahit_index = VK_SHADER_UNUSED_KHR;
        uint32_t rint_index = VK_SHADER_UNUSED_KHR;

        if(!hg.rchit.data.empty())
        {
            load_shader_module(
                hg.rchit, vk::ShaderStageFlagBits::eClosestHitKHR, stages,
                specialization
            );
            chit_index = stages.size()-1;
        }

        if(!hg.rahit.data.empty())
        {
            load_shader_module(
                hg.rahit, vk::ShaderStageFlagBits::eAnyHitKHR, stages,
                specialization
            );
            ahit_index = stages.size()-1;
        }

        if(!hg.rint.data.empty())
        {
            load_shader_module(
                hg.rint, vk::ShaderStageFlagBits::eIntersectionKHR, stages,
                specialization
            );
            rint_index = stages.size()-1;
        }

        rt_shader_groups.push_back({
            hg.type,
            VK_SHADER_UNUSED_KHR, chit_index,
            ahit_index, rint_index
        });
    }

    for(shader_source src: src.rmiss)
    {
        load_shader_module(
            src, vk::ShaderStageFlagBits::eMissKHR, stages, specialization
        );
        rt_shader_groups.push_back({
            vk::RayTracingShaderGroupTypeKHR::eGeneral,
            (uint32_t)stages.size()-1, VK_SHADER_UNUSED_KHR,
            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR
        });
    }

    vk::RayTracingPipelineCreateInfoKHR pipeline_info(
        {}, stages.size(), stages.data(),
        rt_shader_groups.size(), rt_shader_groups.data(),
        max_recursion_depth,
        {},
        nullptr,
        {},
        pipeline_layout,
        {},
        -1
    );

    pipeline = vkm(*dev, dev->logical.createRayTracingPipelineKHR({}, dev->pp_cache, pipeline_info).value);

    // Create shader binding table
    uint32_t group_handle_size = align_up_to(
        dev->rt_props.shaderGroupHandleSize,
        dev->rt_props.shaderGroupHandleAlignment
    );

    // Fetch handles
    std::vector<uint8_t> shader_handles(rt_shader_groups.size() * dev->rt_props.shaderGroupHandleSize);
    (void)dev->logical.getRayTracingShaderGroupHandlesKHR(
        pipeline, 0, rt_shader_groups.size(),
        shader_handles.size(), shader_handles.data()
    );

    // Put handles into memory with the correct alignments (yes, it's super
    // annoying)
    size_t offset = 0;
    size_t group_index = 0;
    std::vector<uint8_t> sbt_mem(group_handle_size);
    rgen_sbt = vk::StridedDeviceAddressRegionKHR{
        offset, group_handle_size, group_handle_size
    };
    memcpy(
        sbt_mem.data(),
        shader_handles.data() + group_index * dev->rt_props.shaderGroupHandleSize,
        group_handle_size
    );
    group_index++;
    offset += group_handle_size;
    offset = align_up_to(offset, dev->rt_props.shaderGroupBaseAlignment);

    rchit_sbt = vk::StridedDeviceAddressRegionKHR{
        offset, group_handle_size, group_handle_size * hit_group_count
    };
    for(size_t g = 0; g < hit_group_count; ++g, ++group_index)
    {
        sbt_mem.resize(offset + group_handle_size);
        memcpy(
            sbt_mem.data() + offset,
            shader_handles.data() + group_index * dev->rt_props.shaderGroupHandleSize,
            group_handle_size
        );
        offset += group_handle_size;
    }
    offset = align_up_to(offset, dev->rt_props.shaderGroupBaseAlignment);

    rmiss_sbt = vk::StridedDeviceAddressRegionKHR{
        offset, group_handle_size, group_handle_size * src.rmiss.size()
    };
    for(size_t g = 0; g < src.rmiss.size(); ++g, ++group_index)
    {
        sbt_mem.resize(offset + group_handle_size);
        memcpy(
            sbt_mem.data() + offset,
            shader_handles.data() + group_index * dev->rt_props.shaderGroupHandleSize,
            group_handle_size
        );
        offset += group_handle_size;
    }
    offset = align_up_to(offset, dev->rt_props.shaderGroupBaseAlignment);

    sbt_buffer = create_buffer(
        *dev,
        {
            {},
            (uint32_t)sbt_mem.size(),
            vk::BufferUsageFlagBits::eTransferSrc |
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eShaderBindingTableKHR,
            vk::SharingMode::eExclusive
        },
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        sbt_mem.data()
    );

    vk::DeviceAddress sbt_addr = sbt_buffer.get_address();
    rgen_sbt.deviceAddress += sbt_addr;
    rchit_sbt.deviceAddress += sbt_addr;
    rmiss_sbt.deviceAddress += sbt_addr;
}

}
