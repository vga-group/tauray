#include "gfx_pipeline.hh"
#include "descriptor_state.hh"
#include "misc.hh"
#include <map>

namespace tr
{

gfx_pipeline::gfx_pipeline(
    device_data& dev, const pipeline_state& state
):  basic_pipeline(
        dev, state.src, state.binding_array_lengths, MAX_FRAMES_IN_FLIGHT,
        state.src.rgen.data.empty() ? vk::PipelineBindPoint::eGraphics : vk::PipelineBindPoint::eRayTracingKHR,
        state.use_push_descriptors
    ),
    state(state)
{
    init_pipeline();
}

vk::Framebuffer gfx_pipeline::get_framebuffer(uint32_t frame_index) const
{
    return framebuffers[frame_index];
}

const gfx_pipeline::pipeline_state& gfx_pipeline::get_state() const
{
    return state;
}

void gfx_pipeline::begin_render_pass(vk::CommandBuffer buf, uint32_t frame_index)
{
    uvec2 size = state.output_size;
    uvec4 vp = state.viewport;
    vk::RenderPassBeginInfo render_pass_info(
        render_pass,
        framebuffers[frame_index],
        vk::Rect2D({(int)vp.x, (int)(size.y-vp.y-vp.w)}, {vp.z, vp.w}),
        (uint32_t)clear_values.size(),
        clear_values.data()
    );
    buf.beginRenderPass(render_pass_info, vk::SubpassContents::eInline);
}

void gfx_pipeline::begin_render_pass(vk::CommandBuffer buf, uint32_t frame_index, uvec4 rect)
{
    uvec2 size = state.output_size;
    vk::RenderPassBeginInfo render_pass_info(
        render_pass,
        framebuffers[frame_index],
        {{(int)rect.x, (int)(size.y-rect.y-rect.w)}, {rect.z, rect.w}},
        (uint32_t)clear_values.size(),
        clear_values.data()
    );
    buf.beginRenderPass(render_pass_info, vk::SubpassContents::eInline);
}

void gfx_pipeline::end_render_pass(vk::CommandBuffer buf)
{
    buf.endRenderPass();
}

void gfx_pipeline::trace_rays(vk::CommandBuffer buf, uvec3 size)
{
    buf.traceRaysKHR(
        &rgen_sbt, &rmiss_sbt, &rchit_sbt, &rcallable_sbt,
        size.x, size.y, size.z
    );
}

uint32_t gfx_pipeline::get_multiview_layer_count() const
{
    uint32_t layer_count = 1;
    for(const auto& att: state.color_attachments)
    {
        if(att.target)
            layer_count = std::max(att.target.get_layer_count(), layer_count);
        else
            layer_count = std::max(att.target.get_layer_count(), layer_count);
    }
    if(state.depth_attachment)
    {
        layer_count = std::max(
            state.depth_attachment->target.get_layer_count(),
            layer_count
        );
    }
    return layer_count;
}

void gfx_pipeline::init_render_pass()
{
    std::vector<vk::AttachmentDescription> all_attachments;
    std::vector<vk::AttachmentReference> color_attachment_refs;
    std::vector<vk::AttachmentReference> depth_attachment_refs;
    uint32_t index = 0;
    for(const auto& att: state.color_attachments)
    {
        if(att.target)
        {
            all_attachments.push_back(att.desc);
            color_attachment_refs.push_back({
                index++, vk::ImageLayout::eColorAttachmentOptimal
            });
        }
        else
        {
            color_attachment_refs.push_back({
                VK_ATTACHMENT_UNUSED, vk::ImageLayout::eColorAttachmentOptimal
            });
        }
    }

    if(state.depth_attachment)
    {
        all_attachments.push_back(state.depth_attachment->desc);
        depth_attachment_refs.push_back({
            index++, vk::ImageLayout::eDepthStencilAttachmentOptimal
        });
    }

    vk::SubpassDescription subpass(
        {},
        vk::PipelineBindPoint::eGraphics,
        0, nullptr, // TODO: Input attachments
        color_attachment_refs.size(), color_attachment_refs.data(),
        nullptr,
        depth_attachment_refs.size() > 0 ?
            depth_attachment_refs.data() : nullptr,
        0,
        nullptr
    );

    vk::SubpassDependency subpass_dep(
        VK_SUBPASS_EXTERNAL, 0,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, vk::AccessFlagBits::eColorAttachmentWrite
    );

    vk::RenderPassCreateInfo render_pass_info(
        {}, all_attachments.size(), all_attachments.data(), 1, &subpass, 1,
        &subpass_dep
    );
    uint32_t layer_count = get_multiview_layer_count();
    uint32_t full_mask = (1lu << layer_count)-1;
    int32_t view_offset = 0;
    vk::RenderPassMultiviewCreateInfo multiview_info(
        1, &full_mask, 1, &view_offset, 1, &full_mask
    );
    if(state.raster_options.multiview)
    {
        render_pass_info.pNext = &multiview_info;
        if(dev->mv_props.maxMultiviewViewCount < layer_count)
        {
            throw std::runtime_error(
                "Rasterizer requested to do " + std::to_string(layer_count) +
                " views simultaneously, but can only do " +
                std::to_string(dev->mv_props.maxMultiviewViewCount) + "!"
            );
        }
    }
    render_pass = vkm(*dev, dev->dev.createRenderPass(render_pass_info));
}

void gfx_pipeline::init_pipeline()
{
    init_render_pass();

    std::vector<vk::PipelineShaderStageCreateInfo> stages;

    load_shader_module(state.src.vert, vk::ShaderStageFlagBits::eVertex, stages);
    load_shader_module(state.src.frag, vk::ShaderStageFlagBits::eFragment, stages);

    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> rt_shader_groups;

    if(!state.src.rgen.data.empty())
    {
        load_shader_module(
            state.src.rgen,
            vk::ShaderStageFlagBits::eRaygenKHR,
            stages
        );
        rt_shader_groups.push_back({
            vk::RayTracingShaderGroupTypeKHR::eGeneral,
            (uint32_t)stages.size()-1, VK_SHADER_UNUSED_KHR,
            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR
        });
    }

    size_t hit_group_count = state.src.rhit.size();
    for(size_t i = 0; i < hit_group_count; ++i)
    {
        auto& hg = state.src.rhit[i];
        uint32_t chit_index = VK_SHADER_UNUSED_KHR;
        uint32_t ahit_index = VK_SHADER_UNUSED_KHR;
        uint32_t rint_index = VK_SHADER_UNUSED_KHR;

        if(!hg.rchit.data.empty())
        {
            load_shader_module(
                hg.rchit, vk::ShaderStageFlagBits::eClosestHitKHR, stages
            );
            chit_index = stages.size()-1;
        }

        if(!hg.rahit.data.empty())
        {
            load_shader_module(
                hg.rahit, vk::ShaderStageFlagBits::eAnyHitKHR, stages
            );
            ahit_index = stages.size()-1;
        }

        if(!hg.rint.data.empty())
        {
            load_shader_module(
                hg.rint, vk::ShaderStageFlagBits::eIntersectionKHR, stages
            );
            rint_index = stages.size()-1;
        }

        rt_shader_groups.push_back({
            hg.type,
            VK_SHADER_UNUSED_KHR, chit_index,
            ahit_index, rint_index
        });
    }

    for(shader_source src: state.src.rmiss)
    {
        load_shader_module(src, vk::ShaderStageFlagBits::eMissKHR, stages);
        rt_shader_groups.push_back({
            vk::RayTracingShaderGroupTypeKHR::eGeneral,
            (uint32_t)stages.size()-1, VK_SHADER_UNUSED_KHR,
            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR
        });
    }

    vk::PipelineVertexInputStateCreateInfo vertex_input(
        {},
        state.vertex_bindings.size(), state.vertex_bindings.data(),
        state.vertex_attributes.size(), state.vertex_attributes.data()
    );

    vk::PipelineInputAssemblyStateCreateInfo input_assembly(
        {}, vk::PrimitiveTopology::eTriangleList, false
    );

    // The negative height fixes Vulkan's IMMORAL DEFAULT Y-AXIS!
    vk::Viewport viewport(
        state.viewport.x, state.output_size.y-state.viewport.y,
        (float)state.viewport.z, -(float)state.viewport.w,
        0.0f, 1.0f
    );
    vk::Rect2D scissor({0,0}, {state.output_size.x, state.output_size.y});

    vk::PipelineViewportStateCreateInfo viewport_state(
        {}, 1, &viewport, 1, &scissor
    );

    vk::PipelineRasterizationStateCreateInfo rasterization(
        {},
        false,
        false,
        vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eNone,
        vk::FrontFace::eCounterClockwise,
        false,
        0.0f,
        0.0f,
        0.0f,
        0.0f
    );

    vk::PipelineDynamicStateCreateInfo dynamic_state = {
        {}, 0, nullptr
    };

    std::vector<vk::PipelineColorBlendAttachmentState> color_blend_attachments;

    vk::SampleCountFlagBits msaa = vk::SampleCountFlagBits::e1;
    for(auto& att: state.color_attachments)
    {
        color_blend_attachments.push_back({
            att.blend,
            att.blend_src_color,
            att.blend_dst_color,
            att.blend_color_op,
            att.blend_src_alpha,
            att.blend_dst_alpha,
            att.blend_alpha_op,
            vk::ColorComponentFlagBits::eR|vk::ColorComponentFlagBits::eG|
            vk::ColorComponentFlagBits::eB|vk::ColorComponentFlagBits::eA
        });
        msaa = std::max(att.desc.samples, msaa);
        clear_values.push_back(att.clear);
    }

    vk::PipelineColorBlendStateCreateInfo color_blending(
        {},
        false,
        vk::LogicOp::eCopy,
        color_blend_attachments.size(),
        color_blend_attachments.data(),
        std::array<float,4>{0.0f, 0.0f, 0.0f, 0.0f}
    );

    vk::PipelineDepthStencilStateCreateInfo depth_stencil;

    if(state.depth_attachment)
    {
        depth_stencil.depthTestEnable = state.depth_attachment->depth_test;
        depth_stencil.depthWriteEnable = state.depth_attachment->depth_write;
        depth_stencil.depthCompareOp = state.depth_attachment->depth_compare;
        depth_stencil.minDepthBounds = 0.0f;
        depth_stencil.minDepthBounds = 1.0f;
        msaa = std::max(state.depth_attachment->desc.samples, msaa);
        clear_values.push_back(state.depth_attachment->clear);
    }

    vk::PipelineMultisampleStateCreateInfo multisampling(
        {}, msaa, state.raster_options.sample_shading, 1.0f, nullptr,
        state.raster_options.alpha_to_coverage, true
    );

    if(rt_shader_groups.size() == 0)
    {
        vk::GraphicsPipelineCreateInfo pipeline_info(
            {}, stages.size(), stages.data(), &vertex_input, &input_assembly,
            nullptr, &viewport_state, &rasterization, &multisampling,
            &depth_stencil, &color_blending, &dynamic_state, pipeline_layout,
            render_pass,
            0,
            nullptr,
            -1
        );

        pipeline = vkm(*dev, dev->dev.createGraphicsPipeline(dev->pp_cache, pipeline_info).value);

        init_framebuffers();
    }
    else
    {
        vk::RayTracingPipelineCreateInfoKHR pipeline_info(
            {}, stages.size(), stages.data(),
            rt_shader_groups.size(), rt_shader_groups.data(),
            state.rt_options.max_recursion_depth,
            {},
            nullptr,
            {},
            pipeline_layout,
            {},
            -1
        );

        pipeline = vkm(*dev, dev->dev.createRayTracingPipelineKHR({}, dev->pp_cache, pipeline_info).value);

        // Create shader binding table
        uint32_t group_handle_size = align_up_to(
            dev->rt_props.shaderGroupHandleSize,
            dev->rt_props.shaderGroupHandleAlignment
        );

        // Fetch handles
        std::vector<uint8_t> shader_handles(rt_shader_groups.size() * dev->rt_props.shaderGroupHandleSize);
        (void)dev->dev.getRayTracingShaderGroupHandlesKHR(
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
            offset, group_handle_size, group_handle_size * state.src.rmiss.size()
        };
        for(size_t g = 0; g < state.src.rmiss.size(); ++g, ++group_index)
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
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
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

void gfx_pipeline::init_framebuffers()
{
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        std::vector<vk::ImageView> fb_attachments;

        for(auto& att: state.color_attachments)
        {
            if(att.target)
            {
                assert(att.target.get_size() == state.output_size);
                fb_attachments.push_back(att.target[i].view);
            }
        }
        if(auto& att = state.depth_attachment)
        {
            assert(att->target.get_size() == state.output_size);
            fb_attachments.push_back(att->target[i].view);
        }

        vk::FramebufferCreateInfo framebuffer_info(
            {},
            render_pass,
            fb_attachments.size(),
            fb_attachments.data(),
            state.output_size.x,
            state.output_size.y,
            1
        );
        framebuffers[i] = vkm(*dev, dev->dev.createFramebuffer(framebuffer_info));
    }
}

void gfx_pipeline::load_shader_module(
    shader_source src,
    vk::ShaderStageFlagBits stage,
    std::vector<vk::PipelineShaderStageCreateInfo>& stages
){
    if(src.data.empty()) return;

    vkm<vk::ShaderModule> mod = vkm(*dev, dev->dev.createShaderModule({
        {}, src.data.size()*sizeof(uint32_t), src.data.data()
    }));
    stages.push_back({{}, stage, mod, "main", state.specialization.pData != nullptr ? &state.specialization : nullptr});
}

}
