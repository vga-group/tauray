#ifndef TAURAY_RT_STAGE_HH
#define TAURAY_RT_STAGE_HH
#include "context.hh"
#include "texture.hh"
#include "basic_pipeline.hh"
#include "material.hh"
#include "distribution_strategy.hh"
#include "sampler_table.hh"
#include "timer.hh"
#include "gpu_buffer.hh"
#include "stage.hh"
#include "rt_pipeline.hh"

namespace tr
{

class scene_stage;
class rt_stage: public single_device_stage
{
public:
    enum class sampler_type
    {
        UNIFORM_RANDOM = 0,
        SOBOL_OWEN,
        SOBOL_Z_ORDER_2D,
        SOBOL_Z_ORDER_3D
    };

    struct options
    {
        int max_ray_depth = 8;
        float min_ray_dist = 0.001f;

        int rng_seed = 0;
        sampler_type local_sampler  = sampler_type::UNIFORM_RANDOM;

        // Small values add overhead but allow more detailed progression
        // tracking. 0 puts all in one command buffer and is fastest.
        size_t max_passes_per_command_buffer = 0;
    };

    rt_stage(
        device& dev,
        scene_stage& ss,
        const options& opt,
        const std::string& timer_name,
        unsigned pass_count = 1
    );
    rt_stage(const rt_stage& other) = delete;
    rt_stage(rt_stage&& other) = delete;

    void reset_sample_counter();

    void get_common_defines(std::map<std::string, std::string>& defines);


protected:
    void update(uint32_t frame_index) override;
    virtual void record_command_buffer(
        vk::CommandBuffer cb,
        uint32_t frame_index,
        uint32_t pass_index,
        bool first_in_command_buffer
    ) = 0;
    void get_descriptors(push_descriptor_set& desc);
    void record_command_buffers();
    void force_command_buffer_refresh();

    unsigned get_pass_count() const;

    scene_stage* ss;
    unsigned sample_count_multiplier;

private:
    options opt;
    unsigned pass_count = 1;
    timer rt_timer;

    gpu_buffer sampling_data;
    uint32_t frame_counter;
    uint32_t scene_state_counter;
    bool force_refresh;
};

}

#endif
