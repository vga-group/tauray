#include "raster_stage.hh"
#include "mesh.hh"
#include "shader_source.hh"
#include "placeholders.hh"
#include "scene_stage.hh"
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
    int32_t base_camera_index;
    int32_t pad[2];
    gpu_shadow_mapping_parameters sm_params;
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
        att.desc.format = entry.format;
        bool clear = opt.clear_color || &entry != &gbuf.color;
        att.desc.initialLayout = clear?
            vk::ImageLayout::eUndefined :
            entry.layout;
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
            gbuf.depth.format,
            (vk::SampleCountFlagBits)gbuf.get_msaa(),
            opt.clear_depth ? vk::AttachmentLoadOp::eClear :
                vk::AttachmentLoadOp::eLoad,
            vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare,
            opt.clear_depth ? vk::ImageLayout::eUndefined :
                vk::ImageLayout::eDepthStencilAttachmentOptimal,
            opt.output_layout
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
    device& dev,
    scene_stage& ss,
    const std::vector<gbuffer_target>& output_array_targets,
    const options& opt
):  single_device_stage(dev),
    output_targets(output_array_targets),
    opt(opt),
    scene_state_counter(0),
    ss(&ss),
    raster_timer(
        dev,
        std::string(output_array_targets[0].color ? "forward" : "gbuffer") +
        " rasterization (" + std::to_string(count_gbuffer_array_layers(output_array_targets)) +
        " viewports)"
    )
{
    for(const gbuffer_target& target: output_array_targets)
    {
        array_pipelines.emplace_back(new raster_pipeline(dev));
        array_pipelines.back()->init({
            target.get_size(),
            uvec4(0, 0, target.get_size()),
            load_sources(opt, target),
            {&ss.get_descriptors(), &ss.get_raster_descriptors()},
            mesh::get_bindings(),
            mesh::get_attributes(),
            get_color_attachments(opt, target),
            get_depth_attachment(opt, target),
            opt.sample_shading, (bool)target.color || opt.force_alpha_to_coverage, true,
            {}, false
        });
    }
}

void raster_stage::update(uint32_t)
{
    if(!ss->check_update(scene_stage::GEOMETRY|scene_stage::LIGHT, scene_state_counter))
        return;

    clear_commands();
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_graphics();

        raster_timer.begin(cb, dev->id, i);
        size_t j = 0;
        for(std::unique_ptr<raster_pipeline>& gfx: array_pipelines)
        {
            gfx->begin_render_pass(cb, i);
            gfx->bind(cb);
            gfx->set_descriptors(cb, ss->get_descriptors(), 0, 0);
            gfx->set_descriptors(cb, ss->get_raster_descriptors(), 0, 1);

            const std::vector<scene_stage::instance>& instances = ss->get_instances();
            push_constant_buffer control;
            control.sm_params = create_shadow_mapping_parameters(opt.filter, *ss);
            control.ambient_color = ss->get_ambient();
            control.base_camera_index = j;

            for(size_t i = 0; i < instances.size(); ++i)
            {
                const scene_stage::instance& inst = instances[i];
                const mesh* m = inst.m;
                vk::Buffer vertex_buffers[] = {m->get_vertex_buffer(dev->id)};
                vk::DeviceSize offsets[] = {0};
                cb.bindVertexBuffers(0, 1, vertex_buffers, offsets);
                cb.bindIndexBuffer(
                    m->get_index_buffer(dev->id),
                    0, vk::IndexType::eUint32
                );
                control.instance_id = i;

                gfx->push_constants(cb, control);

                cb.drawIndexed(m->get_indices().size(), 1, 0, 0, 0);
            }
            gfx->end_render_pass(cb);
            j += gfx->get_multiview_layer_count();
        }
        raster_timer.end(cb, dev->id, i);
        end_graphics(cb, i);
    }
}

}
