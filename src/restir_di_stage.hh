#ifndef TAURAY_RESTIR_DI_STAGE_HH
#define TAURAY_RESTIR_DI_STAGE_HH

#include "gpu_buffer.hh"
#include "rt_camera_stage.hh"
#include "rt_common.hh"
#include "descriptor_set.hh"

namespace tr
{

class restir_di_stage: public rt_camera_stage
{
public:
    struct options: public rt_camera_stage::options
    {
        float search_radius;
        uint32_t spatial_sample_count;
        uint32_t ris_sample_count;
        float max_confidence;
        bool temporal_reuse;
        bool spatial_reuse;
        bool shared_visibility;
        bool sample_visibility;

        light_sampling_weights sampling_weights;
        tri_light_sampling_mode tri_light_mode = tri_light_sampling_mode::HYBRID;
    };

    restir_di_stage(
        device& dev,
        scene_stage& ss,
        const gbuffer_target& output_target,
        const options& opt
    );

protected:
    void update(uint32_t frame_index) override;
    void record_command_buffer_pass(
        vk::CommandBuffer cb,
        uint32_t frame_index,
        uint32_t pass_index,
        uvec3 expected_dispatch_size,
        bool first_in_command_buffer
    ) override;

private:
    push_descriptor_set desc;
    rt_pipeline gfx;
    push_descriptor_set spatial_desc;
    rt_pipeline spatial_reuse;
    options opt;

    gpu_buffer param_buffer;
    texture reservoir_data;
    texture light_data;
    texture previous_normal_data;
    texture previous_pos_data;
};

}

#endif
