#ifndef TAURAY_RT_PIPELINE_HH
#define TAURAY_RT_PIPELINE_HH
#include "context.hh"
#include "texture.hh"
#include "gfx_pipeline.hh"
#include "material.hh"
#include "distribution_strategy.hh"
#include "sampler_table.hh"
#include "timer.hh"
#include "gpu_buffer.hh"
#include "stage.hh"

namespace tr
{

class scene;
class rt_stage: public stage
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
        size_t max_instances = 1024;
        size_t max_samplers = 128;

        int max_ray_depth = 8;
        float min_ray_dist = 0.001f;

        bool pre_transformed_vertices = false;

        int rng_seed = 0;
        sampler_type local_sampler  = sampler_type::UNIFORM_RANDOM;
    };

    static gfx_pipeline::pipeline_state get_common_state(
        uvec2 output_size,
        uvec4 viewport,
        const shader_sources& s,
        const options& opt,
        vk::SpecializationInfo specialization = {}
    );

    static void get_common_defines(
        std::map<std::string, std::string>& defines,
        const options& opt
    );

    rt_stage(
        device_data& dev,
        const options& opt,
        const std::string& timer_name,
        unsigned pass_count = 1
    );
    rt_stage(const rt_stage& other) = delete;
    rt_stage(rt_stage&& other) = delete;

    void set_scene(scene* s);
    scene* get_scene();

    void reset_sample_counter();

    void set_local_sampler_parameters(
        uvec3 target_size,
        uint32_t frame_counter_increment
    );

protected:
    void update(uint32_t frame_index) override;
    virtual void record_command_buffer(
        vk::CommandBuffer cb,
        uint32_t frame_index,
        uint32_t pass_index
    ) = 0;
    virtual void init_scene_resources() = 0;
    void init_descriptors(basic_pipeline& pp);
    void record_command_buffers();

    unsigned get_pass_count() const;

private:
    options opt;
    unsigned pass_count = 1;
    timer rt_timer;

    scene* cur_scene;
    gpu_buffer sampling_data;
    uvec3 sampling_target_size;
    uint32_t sampling_frame_counter_increment;
    uint32_t sample_counter;
};

}

#endif

