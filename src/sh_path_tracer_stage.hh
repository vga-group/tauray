#ifndef TAURAY_SH_PATH_TRACER_HH
#define TAURAY_SH_PATH_TRACER_HH
#include "rt_stage.hh"
#include "path_tracer_stage.hh"

namespace tr
{

class scene;
class sh_path_tracer_stage: public rt_stage
{
public:
    struct options: public rt_stage::options
    {
        int samples_per_probe = 1;
        int samples_per_invocation = 1;
        film_filter film = film_filter::BLACKMAN_HARRIS;
        multiple_importance_sampling_mode mis_mode =
            multiple_importance_sampling_mode::MIS_POWER_HEURISTIC;
        float film_radius = 1.0f; // 0.5 is "correct" for the box filter.
        float russian_roulette_delta = 0;
        float temporal_ratio = 0.02f;
        float indirect_clamping = 100.0f;
        float regularization_gamma = 1.0f; // 0 disables path regularization

        light_sampling_weights sampling_weights;

        int sh_grid_index = 0;
        int sh_order = 2;
    };

    sh_path_tracer_stage(
        device_data& dev,
        texture& output_grid,
        vk::ImageLayout output_layout,
        const options& opt
    );

protected:
    void update(uint32_t frame_index) override;
    void init_scene_resources() override;
    void record_command_buffer(
        vk::CommandBuffer cb, uint32_t frame_index, uint32_t pass_index
    ) override;

private:
    void record_command_buffer_push_constants(
        vk::CommandBuffer cb,
        uint32_t frame_index,
        uint32_t pass_index
    );

    options opt;
    texture* output_grid;
    vk::ImageLayout output_layout;
    gpu_buffer grid_data;
};

}

#endif
