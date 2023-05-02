#ifndef TAURAY_RT_CAMERA_STAGE_HH
#define TAURAY_RT_CAMERA_STAGE_HH
#include "rt_stage.hh"
#include "distribution_strategy.hh"
#include "gbuffer.hh"
#include "camera.hh"

namespace tr
{

// Same as rt_stage, but additionally assumes that:
// * There is a single camera
// * The result is a 2D image
class rt_camera_stage: public rt_stage
{
public:
    struct options: public rt_stage::options
    {
        distribution_params distribution;
        size_t active_viewport_count = 1;
        int samples_per_pixel = 1;
        int samples_per_pass = 1;
        camera::projection_type projection = camera::PERSPECTIVE;
        bool transparent_background = false;
    };

    static void get_common_defines(
        std::map<std::string, std::string>& defines,
        const options& opt
    );

    rt_camera_stage(
        device_data& dev,
        const gbuffer_target& output_target,
        const options& opt,
        const std::string& timer_name = "ray tracing",
        unsigned pass_count = 1
    );

    void reset_accumulated_samples();

    // You can change everything except the distribution strategy.
    void reset_distribution_params(distribution_params distribution);

protected:
    void update(uint32_t frame_index) override;
    void record_command_buffer(
        vk::CommandBuffer cb, uint32_t frame_index, uint32_t pass_index,
        bool first_in_command_buffer
    ) override;
    int get_accumulated_samples() const;

    void init_descriptors(basic_pipeline& pp);

    virtual void record_command_buffer_pass(
        vk::CommandBuffer cb,
        uint32_t frame_index,
        uint32_t pass_index,
        uvec3 expected_dispatch_size,
        bool first_in_command_buffer
    ) = 0;

private:
    options opt;
    gpu_buffer distribution_data;
    gbuffer_target target;

    int accumulated_samples;
};

}

#endif
