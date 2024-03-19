#include "raster_pipeline.hh"
#include "misc.hh"
#include <map>

namespace tr
{

raster_pipeline::raster_pipeline(device& dev)
: basic_pipeline(dev, vk::PipelineBindPoint::eGraphics)
{
}

void raster_pipeline::init(const pipeline_state& state)
{
    this->state = state;
    basic_pipeline::init(get_push_constant_ranges(state.src), state.layout);
    init_pipeline();
}

vk::Framebuffer raster_pipeline::get_framebuffer(uint32_t frame_index) const
{
    return framebuffers[frame_index];
}

const raster_pipeline::pipeline_state& raster_pipeline::get_state() const
{
    return state;
}

void raster_pipeline::begin_render_pass(vk::CommandBuffer buf, uint32_t frame_index)
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

void raster_pipeline::begin_render_pass(vk::CommandBuffer buf, uint32_t frame_index, uvec4 rect)
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

void raster_pipeline::end_render_pass(vk::CommandBuffer buf)
{
    buf.endRenderPass();
}

uint32_t raster_pipeline::get_multiview_layer_count() const
{
    uint32_t layer_count = 1;
    for(const auto& att: state.color_attachments)
    {
        if(att.target)
            layer_count = std::max(att.target.layer_count, layer_count);
        else
            layer_count = std::max(att.target.layer_count, layer_count);
    }
    if(state.depth_attachment)
    {
        layer_count = std::max(
            state.depth_attachment->target.layer_count,
            layer_count
        );
    }
    return layer_count;
}

void raster_pipeline::init_render_pass()
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
    if(state.multiview)
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
    render_pass = vkm(*dev, dev->logical.createRenderPass(render_pass_info));
}

void raster_pipeline::init_pipeline()
{
    init_render_pass();

    std::vector<vk::PipelineShaderStageCreateInfo> stages;

    load_shader_module(state.src.vert, vk::ShaderStageFlagBits::eVertex, stages, state.specialization);
    load_shader_module(state.src.frag, vk::ShaderStageFlagBits::eFragment, stages, state.specialization);

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

    std::vector<vk::DynamicState> dynamic_states;
    if(state.dynamic_viewport)
        dynamic_states.push_back(vk::DynamicState::eViewport);
    vk::PipelineDynamicStateCreateInfo dynamic_state = {
        {}, (uint32_t)dynamic_states.size(), dynamic_states.data()
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
        {}, msaa, state.sample_shading, 1.0f, nullptr,
        state.alpha_to_coverage, false
    );

    vk::GraphicsPipelineCreateInfo pipeline_info(
        {}, stages.size(), stages.data(), &vertex_input, &input_assembly,
        nullptr, &viewport_state, &rasterization, &multisampling,
        &depth_stencil, &color_blending, &dynamic_state, pipeline_layout,
        render_pass,
        0,
        nullptr,
        -1
    );

    pipeline = vkm(*dev, dev->logical.createGraphicsPipeline(dev->pp_cache, pipeline_info).value);

    init_framebuffers();
}

void raster_pipeline::init_framebuffers()
{
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        std::vector<vk::ImageView> fb_attachments;

        for(auto& att: state.color_attachments)
        {
            if(att.target)
            {
                assert(att.target.size == state.output_size);
                fb_attachments.push_back(att.target.view);
            }
        }
        if(auto& att = state.depth_attachment)
        {
            assert(att->target.size == state.output_size);
            fb_attachments.push_back(att->target.view);
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
        framebuffers[i] = vkm(*dev, dev->logical.createFramebuffer(framebuffer_info));
    }
}

}
