#ifndef TAURAY_PATH_TRACER_STAGE_HH
#define TAURAY_PATH_TRACER_STAGE_HH
#include "rt_camera_stage.hh"
#include "film.hh"

namespace tr
{

enum class multiple_importance_sampling_mode
{
    MIS_DISABLED,
    MIS_BALANCE_HEURISTIC,
    MIS_POWER_HEURISTIC
};

class scene;
class path_tracer_stage: public rt_camera_stage
{
public:
    struct options: public rt_camera_stage::options
    {
        bool use_shadow_terminator_fix = false;
        bool use_white_albedo_on_first_bounce = false;
        bool hide_lights = false;
        film::filter film = film::BLACKMAN_HARRIS;
        multiple_importance_sampling_mode mis_mode =
            multiple_importance_sampling_mode::MIS_POWER_HEURISTIC;
        float film_radius = 1.0f; // 0.5 is "correct" for the box filter.
        float russian_roulette_delta = 0; // 0 disables russian roulette.
        float indirect_clamping = 0; // 0 disables indirect clamping.
        bool importance_sample_envmap = true;
        float regularization_gamma = 0.0f; // 0 disables path regularization
        bool depth_of_field = false; // false disregards camera focus parameters.
        bool sample_emissive_triangles = true;
    };

    path_tracer_stage(
        device_data& dev,
        uvec2 ray_count,
        const gbuffer_target& output_target,
        const options& opt
    );

protected:
    void record_command_buffer_push_constants(
        vk::CommandBuffer cb,
        uint32_t frame_index,
        uint32_t pass_index
    ) override;

private:
    options opt;
};

}

#endif
