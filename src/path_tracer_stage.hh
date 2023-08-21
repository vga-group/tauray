#ifndef TAURAY_PATH_TRACER_STAGE_HH
#define TAURAY_PATH_TRACER_STAGE_HH
#include "rt_camera_stage.hh"
#include "rt_common.hh"

namespace tr
{

class scene;
class path_tracer_stage: public rt_camera_stage
{
public:
    struct options: public rt_camera_stage::options
    {
        bool use_shadow_terminator_fix = false;
        bool use_white_albedo_on_first_bounce = false;
        bool hide_lights = false;
        film_filter film = film_filter::BLACKMAN_HARRIS;
        multiple_importance_sampling_mode mis_mode =
            multiple_importance_sampling_mode::MIS_POWER_HEURISTIC;
        float film_radius = 1.0f; // 0.5 is "correct" for the box filter.
        float russian_roulette_delta = 0; // 0 disables russian roulette.
        float indirect_clamping = 0; // 0 disables indirect clamping.
        float regularization_gamma = 0.0f; // 0 disables path regularization
        bool depth_of_field = false; // false disregards camera focus parameters.

        light_sampling_weights sampling_weights;
        bounce_sampling_mode bounce_mode = bounce_sampling_mode::MATERIAL;
        tri_light_sampling_mode tri_light_mode = tri_light_sampling_mode::HYBRID;
    };

    path_tracer_stage(
        device& dev,
        scene_stage& ss,
        const gbuffer_target& output_target,
        const options& opt
    );

protected:
    void init_scene_resources() override;

    void record_command_buffer_pass(
        vk::CommandBuffer cb,
        uint32_t frame_index,
        uint32_t pass_index,
        uvec3 expected_dispatch_size,
        bool first_in_command_buffer
    ) override;

private:
    rt_pipeline gfx;
    options opt;
};

}

#endif
