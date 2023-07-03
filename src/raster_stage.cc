#include "raster_stage.hh"
#include "mesh.hh"
#include "shader_source.hh"
#include "shadow_map_renderer.hh"
#include "placeholders.hh"
#include "scene.hh"
#include "camera.hh"
#include "misc.hh"
#include "sh_grid.hh"
#include "texture.hh"
#include "sampler.hh"

namespace
{
using namespace tr;

// This must match the push_constant_buffer in shader/forward.glsl
struct push_constant_buffer
{
    uint32_t instance_id;
    uint32_t pcf_samples;
    uint32_t omni_pcf_samples;
    uint32_t pcss_samples;
    float pcss_minimum_radius;
    float noise_scale;
    pvec2 shadow_map_atlas_pixel_margin;
    pvec3 ambient_color;
};

raster_shader_sources load_sources(const raster_stage::options& opt, const gbuffer_target& gbuf)
{
    std::map<std::string, std::string> defines;
    defines["SH_ORDER"] = std::to_string(opt.sh_order);
    defines["SH_COEF_COUNT"] = std::to_string(
        sh_grid::get_coef_count(opt.sh_order)
    );
    if(!opt.use_probe_visibility)
        defines["SH_INTERPOLATION_TRILINEAR"];
    gbuf.get_location_defines(defines);
    return {
        {"shader/forward.vert"},
        {"shader/forward.frag", defines}
    };
}

using color_attachment_state = raster_pipeline::pipeline_state::color_attachment_state;
std::vector<color_attachment_state> get_color_attachments(
    const raster_stage::options& opt,
    const gbuffer_target& gbuf
){
    std::vector<color_attachment_state> states;
    color_attachment_state proto = {
        {},
        {
            {},
            {},
            (vk::SampleCountFlagBits)gbuf.get_msaa(),
            vk::AttachmentLoadOp::eLoad,
            vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare,
            {},
            {}
        }
    };

    gbuf.visit([&](const render_target& entry){
        if(&entry == &gbuf.depth)
            return;
        color_attachment_state att = proto;
        att.target = entry;
        att.desc.format = entry.get_format();
        bool clear = opt.clear_color || &entry != &gbuf.color;
        att.desc.initialLayout = clear?
            vk::ImageLayout::eUndefined :
            entry.get_layout();
        if(clear)
            att.desc.loadOp = vk::AttachmentLoadOp::eClear;
        att.desc.finalLayout = opt.output_layout;
        att.clear.setFloat32({NAN,NAN,NAN,NAN});
        states.push_back(att);
    });
    return states;
}

using depth_attachment_state = raster_pipeline::pipeline_state::depth_attachment_state;
depth_attachment_state get_depth_attachment(
    const raster_stage::options& opt,
    const gbuffer_target& gbuf
){
    depth_attachment_state depth = {
        gbuf.depth,
        {
            {},
            gbuf.depth.get_format(),
            (vk::SampleCountFlagBits)gbuf.get_msaa(),
            opt.clear_depth ? vk::AttachmentLoadOp::eClear :
                vk::AttachmentLoadOp::eLoad,
            vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare,
            opt.clear_depth ? vk::ImageLayout::eUndefined :
                vk::ImageLayout::eDepthStencilAttachmentOptimal,
            vk::ImageLayout::eDepthStencilAttachmentOptimal
        },
        true,
        true,
        vk::CompareOp::eLessOrEqual
    };
    return depth;
}

}

namespace tr
{

raster_stage::raster_stage(
    device_data& dev,
    const std::vector<gbuffer_target>& output_array_targets,
    const options& opt
):  stage(dev),
    output_targets(output_array_targets),
    opt(opt),
    brdf_integration_sampler(
        dev,
        vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest,
        0,
        true,
        false
    ),
    cur_scene(nullptr),
    brdf_integration(dev, get_resource_path("data/brdf_integration.exr")),
    noise_vector_2d(dev, get_resource_path("data/noise_vector_2d.exr")),
    noise_vector_3d(dev, get_resource_path("data/noise_vector_3d.exr")),
    raster_timer(
        dev,
        std::string(output_array_targets[0].color ? "forward" : "gbuffer") +
        " rasterization (" + std::to_string(count_array_layers(output_array_targets)) +
        " viewports)"
    )
{
    placeholders& pl = dev.ctx->get_placeholders();

    for(const gbuffer_target& target: output_array_targets)
    {
        array_pipelines.emplace_back(new raster_pipeline(dev, {
            target.get_size(),
            uvec4(0, 0, target.get_size()),
            load_sources(opt, target),
            {
                {"textures", (uint32_t)opt.max_samplers},
                {"textures3d", (uint32_t)opt.max_3d_samplers},
            },
            mesh::get_bindings(),
            mesh::get_attributes(),
            get_color_attachments(opt, target),
            get_depth_attachment(opt, target),
            opt.sample_shading, (bool)target.color || opt.force_alpha_to_coverage, true
        }));

        scene::bind_placeholders(
            *array_pipelines.back(),
            opt.max_samplers,
            opt.max_3d_samplers
        );

        array_pipelines.back()->update_descriptor_set({
            {"pcf_noise_vector_2d", {
                pl.default_sampler.get_sampler(dev.index),
                noise_vector_2d.get_image_view(dev.index),
                vk::ImageLayout::eShaderReadOnlyOptimal
            }},
            {"pcf_noise_vector_3d", {
                pl.default_sampler.get_sampler(dev.index),
                noise_vector_3d.get_image_view(dev.index),
                vk::ImageLayout::eShaderReadOnlyOptimal
            }},
            {"brdf_integration", {
                brdf_integration_sampler.get_sampler(dev.index),
                brdf_integration.get_image_view(dev.index),
                vk::ImageLayout::eShaderReadOnlyOptimal
            }}
        });
    }
}

void raster_stage::set_scene(scene* s)
{
    cur_scene = s;
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        size_t j = 0;
        for(std::unique_ptr<raster_pipeline>& gfx: array_pipelines)
        {
            // Bind descriptors
            cur_scene->bind(*gfx, i, j);
            j += gfx->get_multiview_layer_count();
        }
    }
    record_command_buffers();
}

scene* raster_stage::get_scene()
{
    return cur_scene;
}

void raster_stage::record_command_buffers()
{
    clear_commands();
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_graphics();

        raster_timer.begin(cb, dev->index, i);
        for(std::unique_ptr<raster_pipeline>& gfx: array_pipelines)
        {
            gfx->begin_render_pass(cb, i);
            gfx->bind(cb, i);

            const std::vector<scene::instance>& instances = cur_scene->get_instances();
            push_constant_buffer control;
            control.pcf_samples = opt.pcf_samples;
            control.omni_pcf_samples = opt.omni_pcf_samples;
            control.pcss_samples = opt.pcss_samples;
            control.pcss_minimum_radius = opt.pcss_minimum_radius;
            control.noise_scale =
                opt.sample_shading ? ceil(sqrt((int)output_targets[0].get_msaa())) : 1.0f;
            control.ambient_color = cur_scene->get_ambient();
            control.shadow_map_atlas_pixel_margin = cur_scene->get_shadow_map_atlas_pixel_margin();

            for(size_t i = 0; i < instances.size(); ++i)
            {
                const scene::instance& inst = instances[i];
                const mesh* m = inst.m;
                vk::Buffer vertex_buffers[] = {m->get_vertex_buffer(dev->index)};
                vk::DeviceSize offsets[] = {0};
                cb.bindVertexBuffers(0, 1, vertex_buffers, offsets);
                cb.bindIndexBuffer(
                    m->get_index_buffer(dev->index),
                    0, vk::IndexType::eUint32
                );
                control.instance_id = i;

                gfx->push_constants(cb, control);

                cb.drawIndexed(m->get_indices().size(), 1, 0, 0, 0);
            }
            gfx->end_render_pass(cb);
        }
        raster_timer.end(cb, dev->index, i);
        end_graphics(cb, i);
    }
}

}
