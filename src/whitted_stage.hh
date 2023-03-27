#ifndef TAURAY_WHITTED_STAGE_HH
#define TAURAY_WHITTED_STAGE_HH
#include "rt_camera_stage.hh"

namespace tr
{

class scene;
class whitted_stage: public rt_camera_stage
{
public:
    struct options: public rt_camera_stage::options
    {
        // TODO: Whitted-specific options.
    };

    whitted_stage(
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
