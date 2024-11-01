#include "taa_stage.hh"
#include "misc.hh"
#include "camera.hh"
#include "log.hh"

namespace
{
using namespace tr;

struct push_constant_buffer
{
    pivec2 size;
    int base_camera_index;
    int output_layer;
    float rounding;
    float gamma;
    float alpha;
};

static_assert(sizeof(push_constant_buffer) <= 128);

}

namespace tr
{

taa_stage::taa_stage(
    device& dev,
    scene_stage& ss,
    render_target& src,
    render_target& motion,
    render_target& depth,
    render_target& dst,
    const options& opt
):  single_device_stage(dev),
    opt(opt),
    scene(&ss),
    src(src), motion(motion), depth(depth), dst({dst}),
    pipeline(dev), descriptors(dev),
    target_sampler(
        dev, vk::Filter::eNearest, vk::Filter::eNearest,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest, 0, true, false
    ),
    history_sampler(
        dev, vk::Filter::eLinear, vk::Filter::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest, 0, true, false
    ),
    stage_timer(dev, "temporal antialiasing (" + std::to_string(opt.active_viewport_count) + " viewports)")
{
    init();

    src.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    motion.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    depth.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    dst.layout = vk::ImageLayout::eGeneral;
}

taa_stage::taa_stage(
    device& dev,
    scene_stage& ss,
    render_target& src,
    render_target& motion,
    render_target& depth,
    std::vector<render_target>& swapchain_dst,
    const options& opt
):  single_device_stage(dev),
    opt(opt),
    scene(&ss),
    src(src), motion(motion), depth(depth), dst(swapchain_dst),
    pipeline(dev), descriptors(dev),
    target_sampler(
        dev, vk::Filter::eNearest, vk::Filter::eNearest,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest, 0, true, false
    ),
    history_sampler(
        dev, vk::Filter::eLinear, vk::Filter::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest, 0, true, false
    ),
    stage_timer(dev, "temporal antialiasing (" + std::to_string(opt.active_viewport_count) + " viewports)")
{
    init();

    src.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    motion.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    depth.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    for(render_target& tgt: swapchain_dst)
        tgt.layout = dev.ctx->get_expected_display_layout();
}

void taa_stage::init()
{
    first_frame = true;
    uvec2 size = src.size;
    for(int i = 0; i < 2; ++i)
        color_history[i].emplace(
            *dev,
            size,
            src.layer_count,
            vk::Format::eR16G16B16A16Sfloat,
            0, nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
            vk::ImageLayout::eGeneral,
            vk::SampleCountFlagBits::e1
        );
    std::map<std::string, std::string> defines;
    if(opt.edge_dilation) defines["EDGE_DILATION"];
    if(opt.anti_shimmer) defines["ANTI_SHIMMER"];

    shader_source shader = {"shader/taa.comp", defines};
    descriptors.add(shader);

    pipeline.init(shader, {&descriptors, &scene->get_descriptors()});
}

void taa_stage::update(uint32_t frame_index)
{
    clear_commands();
    vk::CommandBuffer cb = begin_compute();
    stage_timer.begin(cb, dev->id, frame_index);

    uint32_t swapchain_index, frame_index_dummy;
    dev->ctx->get_indices(swapchain_index, frame_index_dummy);

    uint32_t dst_index = std::min((size_t)swapchain_index, dst.size()-1);
    uint32_t parity = dev->ctx->get_frame_counter()&1;

    render_target color_history_input_target = color_history[parity]->get_array_render_target(dev->id);
    render_target color_history_output_target = color_history[1-parity]->get_array_render_target(dev->id);

    src.transition_layout_temporary(cb, vk::ImageLayout::eShaderReadOnlyOptimal, true, true);
    motion.transition_layout_temporary(cb, vk::ImageLayout::eShaderReadOnlyOptimal, true, true);
    depth.transition_layout_temporary(cb, vk::ImageLayout::eShaderReadOnlyOptimal, true, true);
    color_history_input_target.transition_layout(cb, vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal, true, true);
    color_history_output_target.transition_layout(cb, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, true, true);
    dst[dst_index].transition_layout(cb, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, true, true);

    pipeline.bind(cb);
    descriptors.set_image(dev->id, "src_tex", {{target_sampler.get_sampler(dev->id), src.view, vk::ImageLayout::eShaderReadOnlyOptimal}});
    descriptors.set_image(dev->id, "history_tex", {{history_sampler.get_sampler(dev->id), color_history_input_target.view, vk::ImageLayout::eShaderReadOnlyOptimal}});
    descriptors.set_image(dev->id, "dst_img", {{{}, dst[dst_index].view, vk::ImageLayout::eGeneral}});
    descriptors.set_image(dev->id, "history_out_img", {{{}, color_history_output_target.view, vk::ImageLayout::eGeneral}});
    descriptors.set_image(dev->id, "motion_tex", {{target_sampler.get_sampler(dev->id), motion.view, vk::ImageLayout::eShaderReadOnlyOptimal}});
    descriptors.set_image(dev->id, "depth_tex", {{target_sampler.get_sampler(dev->id), depth.view, vk::ImageLayout::eShaderReadOnlyOptimal}});
    pipeline.push_descriptors(cb, descriptors, 0);
    pipeline.set_descriptors(cb, scene->get_descriptors(), 0, 1);

    push_constant_buffer pc;
    pc.size = src.size;
    pc.base_camera_index = opt.base_camera_index;
    pc.output_layer = opt.output_layer;
    pc.rounding = r1_noise(dev->ctx->get_frame_counter());
    pc.gamma = opt.gamma;
    pc.alpha = first_frame ? 1.0f : opt.alpha;

    pipeline.push_constants(cb, pc);
    uvec2 wg = (src.size+15u)/16u;
    cb.dispatch(wg.x, wg.y, opt.active_viewport_count);

    if(dst.size() > 1)
    {
        dst[dst_index].transition_layout(
            cb,
            vk::ImageLayout::eGeneral,
            dev->ctx->get_expected_display_layout(),
            true, true
        );
    }

    stage_timer.end(cb, dev->id, frame_index);
    end_compute(cb, frame_index);
    first_frame = false;
}

}

