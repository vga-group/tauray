#include "bmfr_stage.hh"
#include "misc.hh"

namespace tr
{

struct push_constant_buffer
{
    pivec2 workset_size;
    pivec2 size;
};

static_assert(sizeof(push_constant_buffer) <= 128);

bmfr_stage::bmfr_stage(
    device_data& dev,
    gbuffer_target& current_features,
    gbuffer_target& prev_features,
    const options& opt
):  stage(dev),
    bmfr_preprocess_comp(dev, compute_pipeline::params{ load_shader_source("shader/bmfr_preprocess.comp", opt), {} }),
    bmfr_fit_comp(dev, compute_pipeline::params{ load_shader_source("shader/bmfr_fit.comp", opt), {}}),
    bmfr_weighted_sum_comp(dev, compute_pipeline::params{ load_shader_source("shader/bmfr_weighted_sum.comp", opt), {}}),
    bmfr_accumulate_output_comp(dev, compute_pipeline::params{load_shader_source("shader/bmfr_accumulate_output.comp", opt), {}}),
    current_features(current_features),
    prev_features(prev_features),
    opt(opt),
    stage_timer(dev, "bmfr complete(" + std::to_string(current_features.get_layer_count()) + " viewports)"),
    bmfr_preprocess_timer(dev, "bmfr preprocess(" + std::to_string(current_features.get_layer_count()) + " viewports)"),
    bmfr_fit_timer(dev, "bmfr fitting(" + std::to_string(current_features.get_layer_count()) + " viewports)"),
    bmfr_weighted_sum_timer(dev, "bmfr weighted sum(" + std::to_string(current_features.get_layer_count()) + " viewports)"),
    bmfr_accumulate_output_timer(dev, "accumulated output(" + std::to_string(current_features.get_layer_count()) + " viewports)"),
    image_copy_timer(dev, "image copy(" + std::to_string(current_features.get_layer_count()) + " viewports)")
{
    init_resources();
    record_command_buffers();
}

shader_source bmfr_stage::load_shader_source(const std::string& path, const options& opt)
{
    std::map<std::string, std::string> defines = {};
    if (opt.settings == bmfr_settings::DIFFUSE_ONLY)
    {
        defines.insert({ "BUFFER_COUNT", "13" });
        defines.insert({ "DIFFUSE_ONLY", "" });
        defines.insert({ "NUM_WEIGHTS_PER_FEATURE", "1" });
    }
    else
    {
        defines.insert({ "BUFFER_COUNT", "16" });
        defines.insert({ "NUM_WEIGHTS_PER_FEATURE", "2" });
    }

    return shader_source(path, defines);
}

void bmfr_stage::init_resources()
{
    constexpr uint32_t BLOCK_SIZE = 32;
    constexpr uint32_t FEATURE_COUNT = 10;
    const uint32_t BUFFER_COUNT = opt.settings == bmfr_settings::DIFFUSE_ONLY ? 13 : 16;
    const uint32_t NUM_WEIGHTS_PER_FEATURE = opt.settings == bmfr_settings::DIFFUSE_ONLY ? 1 : 2;
    const uint32_t NUM_VIEWPORTS = current_features.get_layer_count();

    for (int i = 0; i < 4; ++i)
    {
        rt_textures[i].reset(new texture(
                *dev,
                current_features.color.get_size(),
                current_features.get_layer_count(),
                vk::Format::eR16G16B16A16Sfloat,
                0, nullptr,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
                vk::ImageLayout::eGeneral,
                vk::SampleCountFlagBits::e1
            )
        );
    }
    tmp_noisy[0] = rt_textures[0]->get_array_render_target(dev->index);
    tmp_noisy[1] = rt_textures[1]->get_array_render_target(dev->index);
    tmp_filtered[0] = rt_textures[2]->get_array_render_target(dev->index);
    tmp_filtered[1] = rt_textures[3]->get_array_render_target(dev->index);

    for (int i = 4; i < 8; ++i)
    {
        rt_textures[i].reset(new texture(
            *dev,
            current_features.color.get_size(),
            current_features.get_layer_count(),
            vk::Format::eR16G16B16A16Sfloat,
            0, nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,
            vk::ImageLayout::eGeneral,
            vk::SampleCountFlagBits::e1
        )
        );
    }
    diffuse_hist = rt_textures[4]->get_array_render_target(dev->index);
    specular_hist = rt_textures[5]->get_array_render_target(dev->index);
    filtered_hist[0] = rt_textures[6]->get_array_render_target(dev->index);
    filtered_hist[1] = rt_textures[7]->get_array_render_target(dev->index);

    for (int i = 8; i < 10; ++i)
    {
        rt_textures[i].reset(new texture(
            *dev,
            current_features.color.get_size(),
            current_features.get_layer_count(),
            vk::Format::eR16G16B16A16Sfloat,
            0, nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage,
            vk::ImageLayout::eGeneral,
            vk::SampleCountFlagBits::e1
        ));
    }
    weighted_sum[0] = rt_textures[8]->get_array_render_target(dev->index);
    weighted_sum[1] = rt_textures[9]->get_array_render_target(dev->index);


    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        uvec2 wg = (current_features.get_size()+(BLOCK_SIZE - 1))/BLOCK_SIZE + 1u; // + 1 for margins

        // Min / max buffer used for normalizing world pos
        {
            // min and max for each channel of world_pos and world_pos² per work group
            const int required_size = wg.x * wg.y * 6 * 2 * sizeof(uint16_t) * NUM_VIEWPORTS;
            VkBufferCreateInfo bufferInfo = {};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = required_size;
            bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            min_max_buffer[i] = create_buffer(*dev, bufferInfo, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, nullptr);
        }

        // Linear array for all the feature vectors used in the householder QR factorization
        {
            const int required_size = BLOCK_SIZE * BLOCK_SIZE * wg.x * wg.y * BUFFER_COUNT * sizeof(uint16_t) * NUM_VIEWPORTS;
            VkBufferCreateInfo bufferInfo = {};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = required_size;
            bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            tmp_data[i] = create_buffer(*dev, bufferInfo, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, nullptr);
        }

        // Output weights from bmfr fit phase
        {
            const int required_size = wg.x * wg.y * FEATURE_COUNT * NUM_WEIGHTS_PER_FEATURE * sizeof(glm::vec3) * NUM_VIEWPORTS;
            VkBufferCreateInfo bufferInfo = {};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bufferInfo.size = required_size;
            weights[i] = create_buffer(*dev, bufferInfo, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, nullptr);
        }

        // Used to store the temporal reprojection accepts in order to reuse them in the accumulate output shader
        {
            const int required_size = current_features.get_size().x * current_features.get_size().y * NUM_VIEWPORTS;
            VkBufferCreateInfo bufferInfo = {};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = required_size;
            bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            accepts[i] = create_buffer(*dev, bufferInfo, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, nullptr);
        }

        // Used to get current frame index for block offsetting and seeding RNG
        ubos = gpu_buffer(*dev, 4, vk::BufferUsageFlagBits::eUniformBuffer);

        bmfr_preprocess_comp.update_descriptor_set({
            {"in_color", {{}, current_features.color[i].view, vk::ImageLayout::eGeneral}},
            {"in_normal", {{}, current_features.normal[i].view, vk::ImageLayout::eGeneral}},
            {"in_pos", {{}, current_features.pos[i].view, vk::ImageLayout::eGeneral}},
            {"in_screen_motion", {{}, current_features.screen_motion[i].view, vk::ImageLayout::eGeneral}},
            {"previous_normal", {{}, prev_features.normal[i].view, vk::ImageLayout::eGeneral}},
            {"previous_pos", {{}, prev_features.pos[i].view, vk::ImageLayout::eGeneral}},
            {"in_albedo", {{}, current_features.albedo[i].view, vk::ImageLayout::eGeneral}},
            {"in_diffuse", {{}, current_features.diffuse[i].view, vk::ImageLayout::eGeneral}},
            {"tmp_noisy", {
                {{}, tmp_noisy[0][i].view, vk::ImageLayout::eGeneral},
                {{}, tmp_noisy[1][i].view, vk::ImageLayout::eGeneral}
            }},
            {"bmfr_diffuse_hist", {{}, diffuse_hist[i].view, vk::ImageLayout::eGeneral}},
            {"bmfr_specular_hist", {{}, specular_hist[i].view, vk::ImageLayout::eGeneral}},
            {"tmp_buffer", {tmp_data[i], 0, VK_WHOLE_SIZE}},
            {"uniform_buffer", {ubos[dev->index], 0, VK_WHOLE_SIZE}},
            {"accept_buffer", {accepts[i], 0, VK_WHOLE_SIZE}},
        }, i);
        bmfr_fit_comp.update_descriptor_set({
            {"tmp_buffer", {tmp_data[i], 0, VK_WHOLE_SIZE}},
            {"mins_maxs_buffer", {min_max_buffer[i], 0, VK_WHOLE_SIZE}},
            {"weights_buffer", {weights[i], 0, VK_WHOLE_SIZE}},
            {"uniform_buffer", {ubos[dev->index], 0, VK_WHOLE_SIZE}},
            {"in_color", {{}, current_features.color[i].view, vk::ImageLayout::eGeneral}},
        }, i);
        bmfr_weighted_sum_comp.update_descriptor_set({
            {"weights_buffer", {weights[i], 0, VK_WHOLE_SIZE}},
            {"in_color", {{}, current_features.color[i].view, vk::ImageLayout::eGeneral}},
            {"in_normal", {{}, current_features.normal[i].view, vk::ImageLayout::eGeneral}},
            {"in_pos", {{}, current_features.pos[i].view, vk::ImageLayout::eGeneral}},
            {"mins_maxs_buffer", {min_max_buffer[i], 0, VK_WHOLE_SIZE}},
            {"in_diffuse", {{}, current_features.diffuse[i].view, vk::ImageLayout::eGeneral}},
            {"uniform_buffer", {ubos[dev->index], 0, VK_WHOLE_SIZE}},
            {"weighted_out", {
                {{}, weighted_sum[0][i].view, vk::ImageLayout::eGeneral},
                {{}, weighted_sum[1][i].view, vk::ImageLayout::eGeneral}
            }},
            {"tmp_noisy", {
                {{}, tmp_noisy[0][i].view, vk::ImageLayout::eGeneral},
                {{}, tmp_noisy[1][i].view, vk::ImageLayout::eGeneral}
            }},
        }, i);
        bmfr_accumulate_output_comp.update_descriptor_set({
            {"out_color", {{}, current_features.color[i].view, vk::ImageLayout::eGeneral}},
            {"in_screen_motion", {{}, current_features.screen_motion[i].view, vk::ImageLayout::eGeneral}},
            {"in_albedo", {{}, current_features.albedo[i].view, vk::ImageLayout::eGeneral}},
            {"filtered_hist", {
                {{}, filtered_hist[0][i].view, vk::ImageLayout::eGeneral},
                {{}, filtered_hist[1][i].view, vk::ImageLayout::eGeneral}
            }},
            {"accept_buffer", {accepts[i], 0, VK_WHOLE_SIZE}},
            {"tmp_hist", {
                {{}, tmp_filtered[0][i].view, vk::ImageLayout::eGeneral},
                {{}, tmp_filtered[1][i].view, vk::ImageLayout::eGeneral}
            }},
            {"weighted_in", {
                {{}, weighted_sum[0][i].view, vk::ImageLayout::eGeneral},
                {{}, weighted_sum[1][i].view, vk::ImageLayout::eGeneral}
            }},
            {"tmp_noisy", {
                {{}, tmp_noisy[0][i].view, vk::ImageLayout::eGeneral},
                {{}, tmp_noisy[1][i].view, vk::ImageLayout::eGeneral}
            }},
        }, i);
    }
}

void bmfr_stage::record_command_buffers()
{
    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();

        stage_timer.begin(cb, dev->index, i);

        ubos.upload(dev->index, i, cb);
        uvec2 workset_size = ((current_features.get_size()+(31u))/32u) + 1u; // One workset = one 32*32 block
        uvec2 wg = (current_features.get_size()+15u)/16u;
        push_constant_buffer control;
        control.size = current_features.get_size();
        control.workset_size = pivec2(workset_size.x, workset_size.y);


        bmfr_preprocess_comp.bind(cb, i);
        bmfr_preprocess_comp.push_constants(cb, control);
        bmfr_preprocess_timer.begin(cb, dev->index, i);
        cb.dispatch(workset_size.x * 2, workset_size.y * 2, current_features.get_layer_count());
        bmfr_preprocess_timer.end(cb, dev->index, i);

        vk::MemoryBarrier barrier{
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead
        };
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );


        bmfr_fit_comp.bind(cb, i);
        bmfr_fit_comp.push_constants(cb, control);
        bmfr_fit_timer.begin(cb, dev->index, i);
        cb.dispatch(workset_size.x, workset_size.y, current_features.get_layer_count());
        bmfr_fit_timer.end(cb, dev->index, i);

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        wg = (workset_size*32u+15u)/16u;
        // wg = (current_features.get_size()+15u)/16u;
        bmfr_weighted_sum_comp.bind(cb, i);
        bmfr_weighted_sum_comp.push_constants(cb, control);
        bmfr_weighted_sum_timer.begin(cb, dev->index, i);
        cb.dispatch(wg.x, wg.y, current_features.get_layer_count());
        bmfr_weighted_sum_timer.end(cb, dev->index, i);

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        wg = (current_features.get_size()+15u)/16u;
        bmfr_accumulate_output_comp.bind(cb, i);
        bmfr_accumulate_output_comp.push_constants(cb, control);
        bmfr_accumulate_output_timer.begin(cb, dev->index, i);
        cb.dispatch(wg.x, wg.y, current_features.get_layer_count());
        bmfr_accumulate_output_timer.end(cb, dev->index, i);

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        image_copy_timer.begin(cb, dev->index, i);
        copy_image(cb, i, tmp_filtered[0], filtered_hist[0]);
        copy_image(cb, i, tmp_filtered[1], filtered_hist[1]);
        copy_image(cb, i, tmp_noisy[0], diffuse_hist);
        copy_image(cb, i, tmp_noisy[1], specular_hist);
        image_copy_timer.end(cb, dev->index, i);

        stage_timer.end(cb, dev->index, i);
        end_compute(cb, i);
    }
}

void bmfr_stage::update(uint32_t frame_index)
{
    uint32_t frame_counter = dev->ctx->get_frame_counter();
    ubos.update(frame_index, &frame_counter, 0, sizeof(uint32_t));
}

void bmfr_stage::copy_image(vk::CommandBuffer& cb, uint32_t frame_index, render_target& src, render_target& dst)
{
    uvec3 size = uvec3(current_features.get_size(), 1);

    src.transition_layout_temporary(cb, frame_index, vk::ImageLayout::eTransferSrcOptimal);
    dst.transition_layout_temporary(
        cb, frame_index, vk::ImageLayout::eTransferDstOptimal, true, true
    );

    cb.copyImage(
        src[frame_index].image,
        vk::ImageLayout::eTransferSrcOptimal,
        dst[frame_index].image,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageCopy(
            src.get_layers(),
            {0,0,0},
            dst.get_layers(),
            {0,0,0},
            {size.x, size.y, size.z}
        )
    );

    vk::ImageLayout old_layout = src.get_layout();
    src.set_layout(vk::ImageLayout::eTransferSrcOptimal);
    src.transition_layout_temporary(cb, frame_index, vk::ImageLayout::eGeneral);
    src.set_layout(old_layout);

    old_layout = dst.get_layout();
    dst.set_layout(vk::ImageLayout::eTransferDstOptimal);
    dst.transition_layout_temporary(
        cb, frame_index, vk::ImageLayout::eGeneral, true, true
    );
    dst.set_layout(old_layout);
}

}


