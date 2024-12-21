#include "bmfr_stage.hh"
#include "misc.hh"
#include "log.hh"
#include <algorithm>

namespace tr
{

struct push_constant_buffer
{
    pivec2 workset_size;
    pivec2 size;
};

static_assert(sizeof(push_constant_buffer) <= 128);

bmfr_stage::bmfr_stage(
    device& dev,
    gbuffer_target& current_features,
    gbuffer_target& prev_features,
    const options& opt
):  single_device_stage(dev),
    bmfr_preprocess_desc(dev),
    bmfr_preprocess_comp(dev),
    bmfr_fit_desc(dev),
    bmfr_fit_comp(dev),
    bmfr_weighted_sum_desc(dev),
    bmfr_weighted_sum_comp(dev),
    bmfr_accumulate_output_desc(dev),
    bmfr_accumulate_output_comp(dev),
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
    {
        shader_source src = load_shader_source("shader/bmfr_preprocess.comp", opt);
        bmfr_preprocess_desc.add(src);
        bmfr_preprocess_comp.init(src, {&bmfr_preprocess_desc});
    }
    {
        shader_source src = load_shader_source("shader/bmfr_fit.comp", opt);
        bmfr_fit_desc.add(src);
        bmfr_fit_comp.init(src, {&bmfr_fit_desc});
    }
    {
        shader_source src = load_shader_source("shader/bmfr_weighted_sum.comp", opt);
        bmfr_weighted_sum_desc.add(src);
        bmfr_weighted_sum_comp.init(src, {&bmfr_weighted_sum_desc});
    }
    {
        shader_source src = load_shader_source("shader/bmfr_accumulate_output.comp", opt);
        bmfr_accumulate_output_desc.add(src);
        bmfr_accumulate_output_comp.init(src, {&bmfr_accumulate_output_desc});
    }
    init_resources();
    record_command_buffers();
}

void create_bd_macro(
        std::vector<std::string> source, 
        uint32_t& bd_feature_count, 
        std::string& macro
){
    std::map<std::string, std::string> macro_map =
    {
        {"normal", "curr_normal"},
        {"position", "curr_pos"},
        {"position-2", "curr_pos"},

        {"bounce-count", "bd"},
        {"bounce-contribution", "bd"},
        {"material-id", "bd"},
        {"bsdf-sum", "bd"},
        {"bsdf-variance", "bd"},
        {"bsdf-contribution", "bd"},
        {"bsdf-nee-contribution", "bd"}
    };

    std::vector<std::string> channel_map = {"x", "y", "z", "w"};

    macro += "1.f,";

    int indx = 0;
    int count = 1;
    for(auto a : source)
    {
        if(a == "normal" || a == "position")
        {
            macro += macro_map.at(a) + ".x,";
            macro += macro_map.at(a) + ".y,";
            macro += macro_map.at(a) + ".z,";
            count += 3;
        }
        else if(a == "position-2")
        {
            macro += macro_map.at(a) + ".x * " + macro_map.at(a) + ".x,";
            macro += macro_map.at(a) + ".y * " + macro_map.at(a) + ".y,";
            macro += macro_map.at(a) + ".z * " + macro_map.at(a) + ".z,";
            count += 3;
        }
        else
        {
            macro += macro_map.at(a) + "." + channel_map.at(indx % 4) + ",";
            count += 1;
            indx++;
        }
    }

    bd_feature_count = count;
}

shader_source bmfr_stage::load_shader_source(const std::string& path, const options& opt)
{
    std::map<std::string, std::string> defines = {};

    std::string s = "";
    int ic = 0;
    uint32_t normalization_mask = 0;
    uint32_t mask_index = 1;
    if (opt.bd_mode)
    { 
        std::string macro = "";
        uint32_t _feature_count = 0;

        create_bd_macro(opt.bd_vec, _feature_count, macro);
        this->feature_count = _feature_count;

        // features to normalize
        std::vector<std::string> to_normalize_v =
        {
            "position",
            "position-2"
        };

        auto needs_normalization = [](const std::string& feature_str) -> bool {
            return feature_str == "position" || feature_str == "position-2";
            };

        auto get_feature_count = [](const std::string feature_str) -> uint32_t {
            return (feature_str == "position" || feature_str == "position-2" || feature_str == "normal") ? 3 : 1;
            };

        for(int i = 0; i < (int)opt.bd_vec.size(); i++)
        {
            uint32_t fc = get_feature_count(opt.bd_vec[i]);
            if (needs_normalization(opt.bd_vec[i]))
            {
                for (uint32_t j = 0; j < fc; ++j)
                {
                    normalization_mask |= (1 << mask_index++);
                }
            }
            else
            {
                mask_index += fc;
            }

            auto it = std::find(to_normalize_v.begin(), to_normalize_v.end(), opt.bd_vec.at(i));
            if(it != to_normalize_v.end())
            {
                if(*it == "position" || *it == "position-2" || *it == "normal")
                {
                    s += std::to_string(ic) + ",\n";
                    s += std::to_string(++ic) + ",\n";
                    s += std::to_string(++ic) + ",\n";
                    ++ic;
                }
                else
                {
                    s += std::to_string(ic) + ",\n";
                    ++ic;
                }
            }
            else
            {
                if(opt.bd_vec.at(i) == "position" || opt.bd_vec.at(i) == "position-2" || opt.bd_vec.at(i) == "normal")
                {
                    s += " -1 ,\n";
                    s += " -1 ,\n";
                    s += " -1 ,\n";
                }
                else
                {
                    s += " -1 ,\n";
                }
            }
        }

        defines.insert({ "FEATURES", macro });
        defines.insert({ "FEATURE_COUNT", std::to_string(_feature_count) });
        defines.insert({ "NORMALIZATION_MASK", std::to_string(normalization_mask) });
        defines.insert({ "BD_FEATURES_LIST", macro });
        defines.insert({ "BD_FEATURE_COUNT", std::to_string(_feature_count) });
        defines.insert({ "BD_NORM_COUNT", std::to_string(ic)});

        defines.insert({ "BD_NORMALIZE_INDICES", "const int norm_ids["
                 + to_string(feature_count)
                 + "] = { \n"
                 + " -1, \n"
                 + s
                 + "};"
             });


        std::replace(macro.begin(), macro.end(), ',', '\n');
        TR_LOG(macro);
    }

    if (opt.settings == bmfr_settings::DIFFUSE_ONLY)
    {
        defines.insert({ "BUFFER_COUNT", "(FEATURE_COUNT + 3)" });
        defines.insert({ "DIFFUSE_ONLY", "" });
        defines.insert({ "NUM_WEIGHTS_PER_FEATURE", "1" });
    }
    else
    {
        defines.insert({ "BUFFER_COUNT", "(FEATURE_COUNT + 6)" });
        defines.insert({ "NUM_WEIGHTS_PER_FEATURE", "2" });
    }

    for(auto& a : defines)
    {
        TR_LOG(a.first, " : ", a.second);
    }
    
    return shader_source(path, defines);
}

void bmfr_stage::init_resources()
{
    constexpr uint32_t BLOCK_SIZE = 32;
    const uint32_t BUFFER_COUNT = opt.settings == bmfr_settings::DIFFUSE_ONLY ? feature_count + 3 : feature_count + 6;
    const uint32_t NUM_WEIGHTS_PER_FEATURE = opt.settings == bmfr_settings::DIFFUSE_ONLY ? 1 : 2;
    const uint32_t NUM_VIEWPORTS = current_features.get_layer_count();

    for (int i = 0; i < 4; ++i)
    {
        rt_textures[i].reset(new texture(
                *dev,
                current_features.color.size,
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
    tmp_noisy[0] = rt_textures[0]->get_array_render_target(dev->id);
    tmp_noisy[1] = rt_textures[1]->get_array_render_target(dev->id);
    tmp_filtered[0] = rt_textures[2]->get_array_render_target(dev->id);
    tmp_filtered[1] = rt_textures[3]->get_array_render_target(dev->id);

    for (int i = 4; i < 8; ++i)
    {
        rt_textures[i].reset(new texture(
            *dev,
            current_features.color.size,
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
    diffuse_hist = rt_textures[4]->get_array_render_target(dev->id);
    specular_hist = rt_textures[5]->get_array_render_target(dev->id);
    filtered_hist[0] = rt_textures[6]->get_array_render_target(dev->id);
    filtered_hist[1] = rt_textures[7]->get_array_render_target(dev->id);

    for (int i = 8; i < 10; ++i)
    {
        rt_textures[i].reset(new texture(
            *dev,
            current_features.color.size,
            current_features.get_layer_count(),
            vk::Format::eR16G16B16A16Sfloat,
            0, nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage,
            vk::ImageLayout::eGeneral,
            vk::SampleCountFlagBits::e1
        ));
    }
    weighted_sum[0] = rt_textures[8]->get_array_render_target(dev->id);
    weighted_sum[1] = rt_textures[9]->get_array_render_target(dev->id);

    bmfr_preprocess_desc.reset(dev->id, MAX_FRAMES_IN_FLIGHT);
    bmfr_fit_desc.reset(dev->id, MAX_FRAMES_IN_FLIGHT);
    bmfr_weighted_sum_desc.reset(dev->id, MAX_FRAMES_IN_FLIGHT);
    bmfr_accumulate_output_desc.reset(dev->id, MAX_FRAMES_IN_FLIGHT);

    // Used to get current frame index for block offsetting and seeding RNG
    uniform_buffer = gpu_buffer(*dev, 4, vk::BufferUsageFlagBits::eUniformBuffer);

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
            const int required_size = wg.x * wg.y * feature_count * NUM_WEIGHTS_PER_FEATURE * sizeof(glm::vec3) * NUM_VIEWPORTS;
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

        bmfr_preprocess_desc.set_image(dev->id, i, "in_color", {{{}, current_features.color.view, vk::ImageLayout::eGeneral}});
        bmfr_preprocess_desc.set_image(dev->id, i, "in_normal", {{{}, current_features.normal.view, vk::ImageLayout::eGeneral}});
        bmfr_preprocess_desc.set_image(dev->id, i, "in_pos", {{{}, current_features.pos.view, vk::ImageLayout::eGeneral}});
        bmfr_preprocess_desc.set_image(dev->id, i, "in_screen_motion", {{{}, current_features.screen_motion.view, vk::ImageLayout::eGeneral}});
        bmfr_preprocess_desc.set_image(dev->id, i, "previous_normal", {{{}, prev_features.normal.view, vk::ImageLayout::eGeneral}});
        bmfr_preprocess_desc.set_image(dev->id, i, "previous_pos", {{{}, prev_features.pos.view, vk::ImageLayout::eGeneral}});
        bmfr_preprocess_desc.set_image(dev->id, i, "in_albedo", {{{}, current_features.albedo.view, vk::ImageLayout::eGeneral}});
        bmfr_preprocess_desc.set_image(dev->id, i, "in_diffuse", {{{}, current_features.diffuse.view, vk::ImageLayout::eGeneral}});
        bmfr_preprocess_desc.set_image(dev->id, i, "tmp_noisy", {{{}, tmp_noisy[0].view, vk::ImageLayout::eGeneral}, {{}, tmp_noisy[1].view, vk::ImageLayout::eGeneral}});
        bmfr_preprocess_desc.set_image(dev->id, i, "bmfr_diffuse_hist", {{{}, diffuse_hist.view, vk::ImageLayout::eGeneral}});
        bmfr_preprocess_desc.set_image(dev->id, i, "bmfr_specular_hist", {{{}, specular_hist.view, vk::ImageLayout::eGeneral}});
        bmfr_preprocess_desc.set_buffer(dev->id, i, "tmp_buffer", {{tmp_data[i], 0, VK_WHOLE_SIZE}});
        bmfr_preprocess_desc.set_buffer(dev->id, i, "uniform_buffer", {{uniform_buffer[dev->id], 0, VK_WHOLE_SIZE}});
        bmfr_preprocess_desc.set_buffer(dev->id, i, "accept_buffer", {{accepts[i], 0, VK_WHOLE_SIZE}});

#if 1
        bmfr_fit_desc.set_buffer(dev->id, i, "tmp_buffer", {{tmp_data[i], 0, VK_WHOLE_SIZE}});
        bmfr_fit_desc.set_buffer(dev->id, i, "mins_maxs_buffer", {{min_max_buffer[i], 0, VK_WHOLE_SIZE}});
        bmfr_fit_desc.set_buffer(dev->id, i, "weights_buffer", {{weights[i], 0, VK_WHOLE_SIZE}});
        bmfr_fit_desc.set_buffer(dev->id, i, "uniform_buffer", {{uniform_buffer[dev->id], 0, VK_WHOLE_SIZE}});
        bmfr_fit_desc.set_image(dev->id, i, "in_color", {{{}, current_features.color.view, vk::ImageLayout::eGeneral}});
#endif
        //TR_LOG((uintptr_t)(VkImageView)current_features.prob[i].view);

        bmfr_weighted_sum_desc.set_buffer(dev->id, i, "weights_buffer", {{weights[i], 0, VK_WHOLE_SIZE}});
        bmfr_weighted_sum_desc.set_image(dev->id, i, "in_color", {{{}, current_features.color.view, vk::ImageLayout::eGeneral}});
        bmfr_weighted_sum_desc.set_image(dev->id, i, "in_normal", {{{}, current_features.normal.view, vk::ImageLayout::eGeneral}});
        bmfr_weighted_sum_desc.set_image(dev->id, i, "in_pos", {{{}, current_features.pos.view, vk::ImageLayout::eGeneral}});
        bmfr_weighted_sum_desc.set_buffer(dev->id, i, "mins_maxs_buffer", {{min_max_buffer[i], 0, VK_WHOLE_SIZE}});
        bmfr_weighted_sum_desc.set_image(dev->id, i, "in_diffuse", {{{}, current_features.diffuse.view, vk::ImageLayout::eGeneral}});
        bmfr_weighted_sum_desc.set_buffer(dev->id, i, "uniform_buffer", {{uniform_buffer[dev->id], 0, VK_WHOLE_SIZE}});
        bmfr_weighted_sum_desc.set_image(dev->id, i, "weighted_out", {{{}, weighted_sum[0].view, vk::ImageLayout::eGeneral}, {{}, weighted_sum[1].view, vk::ImageLayout::eGeneral}});
        bmfr_weighted_sum_desc.set_image(dev->id, i, "tmp_noisy", {{{}, tmp_noisy[0].view, vk::ImageLayout::eGeneral}, {{}, tmp_noisy[1].view, vk::ImageLayout::eGeneral}});

        bmfr_accumulate_output_desc.set_image(dev->id, i, "out_color", {{{}, current_features.color.view, vk::ImageLayout::eGeneral}});
        bmfr_accumulate_output_desc.set_image(dev->id, i, "in_screen_motion", {{{}, current_features.screen_motion.view, vk::ImageLayout::eGeneral}});
        bmfr_accumulate_output_desc.set_image(dev->id, i, "in_albedo", {{{}, current_features.albedo.view, vk::ImageLayout::eGeneral}});
        bmfr_accumulate_output_desc.set_image(dev->id, i, "filtered_hist", {{{}, filtered_hist[0].view, vk::ImageLayout::eGeneral}, {{}, filtered_hist[1].view, vk::ImageLayout::eGeneral}});
        bmfr_accumulate_output_desc.set_buffer(dev->id, i, "accept_buffer", {{accepts[i], 0, VK_WHOLE_SIZE}});
        bmfr_accumulate_output_desc.set_image(dev->id, i, "tmp_hist", {{{}, tmp_filtered[0].view, vk::ImageLayout::eGeneral}, {{}, tmp_filtered[1].view, vk::ImageLayout::eGeneral}});
        bmfr_accumulate_output_desc.set_image(dev->id, i, "weighted_in", {{{}, weighted_sum[0].view, vk::ImageLayout::eGeneral}, {{}, weighted_sum[1].view, vk::ImageLayout::eGeneral}});
        bmfr_accumulate_output_desc.set_image(dev->id, i, "tmp_noisy", {{{}, tmp_noisy[0].view, vk::ImageLayout::eGeneral}, {{}, tmp_noisy[1].view, vk::ImageLayout::eGeneral}});
    }
}

void bmfr_stage::record_command_buffers()
{
    clear_commands();
    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();

        stage_timer.begin(cb, dev->id, i);

        uniform_buffer.upload(dev->id, i, cb);
        uvec2 workset_size = ((current_features.get_size()+(31u))/32u) + 1u; // One workset = one 32*32 block
        uvec2 wg = (current_features.get_size()+15u)/16u;
        push_constant_buffer control;
        control.size = current_features.get_size();
        control.workset_size = pivec2(workset_size.x, workset_size.y);


        bmfr_preprocess_comp.bind(cb);
        bmfr_preprocess_comp.set_descriptors(cb, bmfr_preprocess_desc, i, 0);
        bmfr_preprocess_comp.push_constants(cb, control);
        bmfr_preprocess_timer.begin(cb, dev->id, i);
        cb.dispatch(workset_size.x * 2, workset_size.y * 2, current_features.get_layer_count());
        bmfr_preprocess_timer.end(cb, dev->id, i);

        vk::MemoryBarrier barrier{
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead
        };
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );


        bmfr_fit_comp.bind(cb);
        bmfr_fit_comp.set_descriptors(cb, bmfr_fit_desc, i, 0);
        bmfr_fit_comp.push_constants(cb, control);
        bmfr_fit_timer.begin(cb, dev->id, i);
        cb.dispatch(workset_size.x, workset_size.y, current_features.get_layer_count());
        bmfr_fit_timer.end(cb, dev->id, i);

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        wg = (workset_size*32u+15u)/16u;
        // wg = (current_features.get_size()+15u)/16u;
        bmfr_weighted_sum_comp.bind(cb);
        bmfr_weighted_sum_comp.set_descriptors(cb, bmfr_weighted_sum_desc, i, 0);
        bmfr_weighted_sum_comp.push_constants(cb, control);
        bmfr_weighted_sum_timer.begin(cb, dev->id, i);
        cb.dispatch(wg.x, wg.y, current_features.get_layer_count());
        bmfr_weighted_sum_timer.end(cb, dev->id, i);

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        wg = (current_features.get_size()+15u)/16u;
        bmfr_accumulate_output_comp.bind(cb);
        bmfr_accumulate_output_comp.set_descriptors(cb, bmfr_accumulate_output_desc, i, 0);
        bmfr_accumulate_output_comp.push_constants(cb, control);
        bmfr_accumulate_output_timer.begin(cb, dev->id, i);
        cb.dispatch(wg.x, wg.y, current_features.get_layer_count());
        bmfr_accumulate_output_timer.end(cb, dev->id, i);

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        image_copy_timer.begin(cb, dev->id, i);
        copy_image(cb, tmp_filtered[0], filtered_hist[0]);
        copy_image(cb, tmp_filtered[1], filtered_hist[1]);
        copy_image(cb, tmp_noisy[0], diffuse_hist);
        copy_image(cb, tmp_noisy[1], specular_hist);
        image_copy_timer.end(cb, dev->id, i);

        stage_timer.end(cb, dev->id, i);
        end_compute(cb, i);
    }
}

void bmfr_stage::update(uint32_t frame_index)
{
    uint32_t frame_counter = dev->ctx->get_frame_counter();
    uniform_buffer.update(frame_index, &frame_counter, 0, sizeof(uint32_t));
}

void bmfr_stage::copy_image(vk::CommandBuffer& cb, render_target& src, render_target& dst)
{
    uvec3 size = uvec3(current_features.get_size(), 1);

    src.transition_layout_temporary(cb, vk::ImageLayout::eTransferSrcOptimal);
    dst.transition_layout_temporary(
        cb, vk::ImageLayout::eTransferDstOptimal, true, true
    );

    cb.copyImage(
        src.image,
        vk::ImageLayout::eTransferSrcOptimal,
        dst.image,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageCopy(
            src.get_layers(),
            {0,0,0},
            dst.get_layers(),
            {0,0,0},
            {size.x, size.y, size.z}
        )
    );

    vk::ImageLayout old_layout = src.layout;
    src.layout = vk::ImageLayout::eTransferSrcOptimal;
    src.transition_layout_temporary(cb, vk::ImageLayout::eGeneral);
    src.layout = old_layout;

    old_layout = dst.layout;
    dst.layout = vk::ImageLayout::eTransferDstOptimal;
    dst.transition_layout_temporary(
        cb, vk::ImageLayout::eGeneral, true, true
    );
    dst.layout = old_layout;
}

}


