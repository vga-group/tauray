#include "tonemap_stage.hh"
#include "misc.hh"
#include <numeric>

namespace
{
using namespace tr;

struct tonemap_info_buffer
{
    pivec2 size;
    int alpha_grid_background;
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
    device& dev,
    render_target& input,
    std::vector<render_target>& output_frames,
    const options& opt
):  single_device_stage(dev, single_device_stage::COMMAND_BUFFER_PER_FRAME_AND_SWAPCHAIN_IMAGE),
    comp(dev, compute_pipeline::params{
        load_shader_source(opt), MAX_FRAMES_IN_FLIGHT * (uint32_t)output_frames.size()
    }),
    opt(opt),
    input_target(input),
    index_data(dev, sizeof(tonemap_info_buffer), vk::BufferUsageFlagBits::eUniformBuffer),
    tonemap_timer(dev, "tonemap (" + std::to_string(input.layer_count) + " viewports)")
{
    input.layout = vk::ImageLayout::eGeneral;

    if(this->opt.reorder.size() != input.layer_count)
    {
        this->opt.reorder.resize(input.layer_count);
        std::iota(this->opt.reorder.begin(), this->opt.reorder.end(), 0);
    }

    vk::BufferCreateInfo info(
        {}, this->opt.reorder.size() * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer, vk::SharingMode::eExclusive
    );
    output_reorder_buf = create_buffer(dev, info, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, this->opt.reorder.data());

    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    for(uint32_t j = 0; j < output_frames.size(); ++j)
    {
        // Bind descriptors
        render_target input = input_target;
        render_target output = output_frames[j];

        comp.update_descriptor_set({
            {"info", {index_data[dev.id], 0, sizeof(tonemap_info_buffer)}},
            {"in_color", {{}, input_target.view, vk::ImageLayout::eGeneral}},
            {"out_color", {{}, output.view, vk::ImageLayout::eGeneral}},
            {"output_reorder", {output_reorder_buf, 0, VK_WHOLE_SIZE}}
        }, get_command_buffer_index(i, j));

        // Record command buffer
        vk::CommandBuffer cb = begin_compute();

        uint32_t cb_index = get_command_buffer_index(i, j);

        input.transition_layout_temporary(cb, vk::ImageLayout::eGeneral, true);
        output.transition_layout_temporary(cb, vk::ImageLayout::eGeneral, true);

        index_data.upload(dev.id, i, cb);
        bulk_upload_barrier(cb, vk::PipelineStageFlagBits::eComputeShader);
        tonemap_timer.begin(cb, dev.id, i);

        comp.bind(cb, cb_index);

        uvec2 wg = (output.size+15u)/16u;
        cb.dispatch(wg.x, wg.y, input.layer_count);

        tonemap_timer.end(cb, dev.id, i);
        if(opt.transition_output_layout)
        {
            vk::ImageLayout old_layout = output.layout;
            output.layout = vk::ImageLayout::eGeneral;
            output.transition_layout_temporary(cb, dev.ctx->get_expected_display_layout());
            output.layout = old_layout;
        }
        end_compute(cb, cb_index);
    }
    input.layout = vk::ImageLayout::eGeneral;
    for(render_target& out: output_frames)
        out.layout = opt.transition_output_layout ? dev.ctx->get_expected_display_layout() : vk::ImageLayout::eGeneral;
}

void tonemap_stage::update(uint32_t frame_index)
{
    tonemap_info_buffer info;
    info.size = input_target.size;
    info.exposure = opt.exposure;
    info.gamma = opt.tonemap_operator == LINEAR ? 1.0 : opt.gamma;
    info.alpha_grid_background = opt.alpha_grid_background ? 16 : 0;

    index_data.update(frame_index, &info);
}

}
