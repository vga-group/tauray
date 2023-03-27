#ifndef TAURAY_DIRECT_STAGE_HH
#define TAURAY_DIRECT_STAGE_HH
#include "path_tracer_stage.hh"

namespace tr
{

class scene;
class direct_stage: public rt_camera_stage
{
public:
    struct options: public rt_camera_stage::options
    {
        film_filter film = film_filter::BLACKMAN_HARRIS;
        float film_radius = 1.0f; // 0.5 is "correct" for the box filter.

        light_sampling_weights sampling_weights;
        bounce_sampling_mode bounce_mode = bounce_sampling_mode::MATERIAL;
        tri_light_sampling_mode tri_light_mode = tri_light_sampling_mode::HYBRID;
    };

    direct_stage(
        device_data& dev,
        uvec2 ray_count,
        const gbuffer_target& output_target,
        const options& opt
    );

protected:
    void init_scene_resources() override;
    void record_command_buffer_pass(
        vk::CommandBuffer cb,
        uint32_t frame_index,
        uint32_t pass_index,
        uvec3 expected_dispatch_size
    ) override;

private:
    gfx_pipeline gfx;
    options opt;
};

}

#endif
