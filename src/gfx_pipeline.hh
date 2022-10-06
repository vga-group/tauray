#ifndef TAURAY_GFX_PIPELINE_HH
#define TAURAY_GFX_PIPELINE_HH
#include "context.hh"
#include "texture.hh"
#include "basic_pipeline.hh"
#include "render_target.hh"
#include <optional>

namespace tr
{

struct shader_sources;

// Pipelines are per-device. A renderer is responsible for juggling between
// pipelines. The gfx_pipeline acts as a base class for all graphics pipelines
// (as opposed to compute pipelines that don't have such clear targets)
class gfx_pipeline: public basic_pipeline
{
public:
    struct pipeline_state
    {
        uvec2 output_size = uvec2(0);
        uvec4 viewport = uvec4(0);
        shader_sources src;

        // The actual bindings are automatically determined from shader
        // source, so you can only define array lengths here (those can't
        // be deduced since they can change).
        binding_array_length_info binding_array_lengths;

        std::vector<vk::VertexInputBindingDescription> vertex_bindings;
        std::vector<vk::VertexInputAttributeDescription> vertex_attributes;

        struct color_attachment_state
        {
            render_target target;
            vk::AttachmentDescription desc;
            bool blend = false;
            vk::BlendFactor blend_src_color = vk::BlendFactor::eSrcAlpha;
            vk::BlendFactor blend_dst_color = vk::BlendFactor::eOneMinusSrcAlpha;
            vk::BlendOp blend_color_op = vk::BlendOp::eAdd;
            vk::BlendFactor blend_src_alpha = vk::BlendFactor::eOne;
            vk::BlendFactor blend_dst_alpha = vk::BlendFactor::eZero;
            vk::BlendOp blend_alpha_op = vk::BlendOp::eAdd;
            vk::ClearColorValue clear = {};
        };
        std::vector<color_attachment_state> color_attachments;

        struct depth_attachment_state
        {
            render_target target;
            vk::AttachmentDescription desc;
            bool depth_test = true;
            bool depth_write = true;
            vk::CompareOp depth_compare = vk::CompareOp::eLess;
            vk::ClearDepthStencilValue clear = vk::ClearDepthStencilValue(1.0f, 0);
        };
        std::optional<depth_attachment_state> depth_attachment;

        struct
        {
            int max_recursion_depth = 1;
        } rt_options = {};

        struct
        {
            bool sample_shading = false;
            bool alpha_to_coverage = false;
            bool multiview = false;
        } raster_options = {};

        vk::SpecializationInfo specialization = {};
        bool use_push_descriptors = false;
    };

    gfx_pipeline(device_data& dev, const pipeline_state& state);

    vk::Framebuffer get_framebuffer(uint32_t frame_index) const;
    const pipeline_state& get_state() const;

    void begin_render_pass(vk::CommandBuffer buf, uint32_t frame_index);
    void begin_render_pass(vk::CommandBuffer buf, uint32_t frame_index, uvec4 rect);
    void end_render_pass(vk::CommandBuffer buf);

    void trace_rays(vk::CommandBuffer buf, uvec3 size);

    uint32_t get_multiview_layer_count() const;

protected:
    vkm<vk::RenderPass> render_pass;
    vkm<vk::Buffer> sbt_buffer; // Only used in a ray tracing pipeline
    vk::StridedDeviceAddressRegionKHR rgen_sbt;
    vk::StridedDeviceAddressRegionKHR rchit_sbt;
    vk::StridedDeviceAddressRegionKHR rmiss_sbt;
    vk::StridedDeviceAddressRegionKHR rcallable_sbt;

private:
    void init_render_pass();
    void init_pipeline();
    void init_framebuffers();

    void load_shader_module(
        shader_source src,
        vk::ShaderStageFlagBits stage,
        std::vector<vk::PipelineShaderStageCreateInfo>& stages
    );

    pipeline_state state;
    std::vector<vk::ClearValue> clear_values;
    vkm<vk::Framebuffer> framebuffers[MAX_FRAMES_IN_FLIGHT];
};

}

#endif
