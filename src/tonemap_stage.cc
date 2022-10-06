#include "tonemap_stage.hh"
#include "misc.hh"
#include <numeric>

namespace
{
using namespace tr;

struct tonemap_info_buffer
{
    pivec2 size;
    float exposure;
    float gamma;
};

shader_source load_shader_source(const tonemap_stage::options& opt)
{
    std::map<std::string, std::string> defines;
    if(opt.input_msaa > 1)
        defines["MSAA_SAMPLES"] = std::to_string(opt.input_msaa);
    if(opt.post_resolve) defines["POST_RESOLVE"];

    switch(opt.tonemap_operator)
    {
    case tonemap_stage::LINEAR:
    case tonemap_stage::GAMMA_CORRECTION:
        return {"shader/tonemap_gamma.comp", defines};
    case tonemap_stage::FILMIC:
        return {"shader/tonemap_filmic.comp", defines};
    case tonemap_stage::REINHARD:
        return {"shader/tonemap_reinhard.comp", defines};
    case tonemap_stage::REINHARD_LUMINANCE:
        return {"shader/tonemap_reinhard_luminance.comp", defines};
    }
    assert(false);
    return {};
}

}

namespace tr
{

tonemap_stage::tonemap_stage(
    device_data& dev,
    render_target& input,
    render_target& output,
    const options& opt
):  stage(dev, stage::COMMAND_BUFFER_PER_FRAME_AND_SWAPCHAIN_IMAGE),
    comp(dev, compute_pipeline::params{
        load_shader_source(opt), {},
        MAX_FRAMES_IN_FLIGHT * (uint32_t)output.get_frame_count()
    }),
    opt(opt),
    input_target(input),
    output_target(output),
    index_data(dev, sizeof(tonemap_info_buffer), vk::BufferUsageFlagBits::eUniformBuffer),
    tonemap_timer(dev, "tonemap (" + std::to_string(input.get_layer_count()) + " viewports)")
{
    input.set_layout(vk::ImageLayout::eGeneral);
    output.set_layout(vk::ImageLayout::ePresentSrcKHR);

    if(this->opt.reorder.size() != input.get_layer_count())
    {
        this->opt.reorder.resize(input.get_layer_count());
        std::iota(this->opt.reorder.begin(), this->opt.reorder.end(), 0);
    }

    vk::BufferCreateInfo info(
        {}, this->opt.reorder.size() * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer, vk::SharingMode::eExclusive
    );
    output_reorder_buf = create_buffer(dev, info, VMA_MEMORY_USAGE_GPU_ONLY, this->opt.reorder.data());

    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    for(uint32_t j = 0; j < output_target.get_frame_count(); ++j)
    {
        // Bind descriptors
        render_target input = input_target;
        render_target output = output_target;

        comp.update_descriptor_set({
            {"info", {*index_data, 0, sizeof(tonemap_info_buffer)}},
            {"in_color", {{}, input_target[i].view, vk::ImageLayout::eGeneral}},
            {"out_color", {{}, output_target[j].view, vk::ImageLayout::eGeneral}},
            {"output_reorder", {output_reorder_buf, 0, VK_WHOLE_SIZE}}
        }, get_command_buffer_index(i, j));

        // Record command buffer
        vk::CommandBuffer cb = begin_compute();

        uint32_t cb_index = get_command_buffer_index(i, j);

        input.transition_layout_temporary(cb, i, vk::ImageLayout::eGeneral, true);
        output.transition_layout_temporary(cb, j, vk::ImageLayout::eGeneral, true);

        index_data.upload(i, cb);
        tonemap_timer.begin(cb, i);

        comp.bind(cb, cb_index);

        uvec2 wg = (output_target.get_size()+15u)/16u;
        cb.dispatch(wg.x, wg.y, input.get_layer_count());

        tonemap_timer.end(cb, i);
        if(opt.transition_output_layout)
        {
            vk::ImageLayout old_layout = output.get_layout();
            output.set_layout(vk::ImageLayout::eGeneral);
            output.transition_layout_temporary(cb, j, dev.ctx->get_expected_display_layout());
            output.set_layout(old_layout);
        }
        end_compute(cb, cb_index);
    }
    input.set_layout(vk::ImageLayout::eGeneral);
    output.set_layout(opt.transition_output_layout ? dev.ctx->get_expected_display_layout() : vk::ImageLayout::eGeneral);
}

void tonemap_stage::update(uint32_t frame_index)
{
    tonemap_info_buffer info;
    info.size = input_target.get_size();
    info.exposure = opt.exposure;
    info.gamma = opt.tonemap_operator == LINEAR ? 1.0 : opt.gamma;

    index_data.update(frame_index, &info);
}

}
