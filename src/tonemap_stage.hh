#ifndef TAURAY_TONEMAP_STAGE_HH
#define TAURAY_TONEMAP_STAGE_HH
#include "context.hh"
#include "texture.hh"
#include "compute_pipeline.hh"
#include "descriptor_set.hh"
#include "timer.hh"
#include "gpu_buffer.hh"
#include "stage.hh"

namespace tr
{

class tonemap_stage: public single_device_stage
{
public:
    enum operator_type
    {
        LINEAR = 0,
        GAMMA_CORRECTION,
        FILMIC,
        REINHARD,
        REINHARD_LUMINANCE
    };

    struct options
    {
        operator_type tonemap_operator = FILMIC;
        float exposure = 1.0f;
        float gamma = 2.2f;
        int input_msaa = 1;
        bool post_resolve = false;
        bool transition_output_layout = true;
        bool alpha_grid_background = false;
        std::vector<uint32_t> reorder = {};
        // If you only want to tonemap one layer of an array, use this.
        int limit_to_input_layer = -1;
        int limit_to_output_layer = -1;
        // eUndefined deduces.
        vk::ImageLayout output_image_layout = vk::ImageLayout::eUndefined;
    };

    tonemap_stage(
        device& dev, 
        render_target& input,
        std::vector<render_target>& output_frames,
        const options& opt
    );

private:
    void update(uint32_t frame_index) override;

    descriptor_set desc;
    compute_pipeline comp;
    options opt;
    render_target input_target;
    vkm<vk::Buffer> output_reorder_buf;
    gpu_buffer index_data;
    timer tonemap_timer;
};

}

#endif

