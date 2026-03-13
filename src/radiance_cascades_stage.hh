#ifndef TR_RADIANCE_CASCADES_STAGE_HH
#define TR_RADIANCE_CASCADES_STAGE_HH
#include "texture.hh"
#include "compute_pipeline.hh"
#include "scene_stage.hh"
#include "timer.hh"
#include <vector>

namespace tr
{

class radiance_cascades_stage: public single_device_stage
{
public:
    enum texel_sampling_type
    {
        UNIFORM = 0,
        BRDF,
        HYBRID
    };

    struct options
    {
        aabb volume;

        // Used for culling dead probes
        texture* occupancy = nullptr;

        // +1 => 8x computation needed, general resolution doubled, less fudge
        // in smallest cascade.
        uint32_t log2_resolution = 8;
        // x2 => 4x computation needed, general resolution doubled
        uint32_t c0_probe_resolution = 4; // 4x4 = 16 rays per probe

        bool jitter = true;
        // Only effective with jittering enabled
        float temporal_ratio = 0.005f;

        // Get lighting for current pass from previous pass. Requires double
        // the memory!
        bool recursive = false;

        // Ambient brightness used for when recursive lighting isn't enabled.
        float ambient = 0.01f;

        // Use rasterizer-style direct illumination (requires shadow maps to be
        // available). If disabled, DI uses radiance cascades and can have a 
        // multitude of issues due to that.
        bool use_raster_di = false;

        // Adjusts how samples are taken at the individual texel level.
        texel_sampling_type texel_sampling = UNIFORM;

        // 0.0f = texel tracks average
        // 1.0f = texel tracks maximum
        float avg_bias = 0.0f;
    };

    radiance_cascades_stage(
        device& dev,
        scene_stage& ss,
        const options& opt
    );

    descriptor_set& get_descriptors();
    size_t get_cascade_count() const;
    float get_cascade_t0(int cascade) const;
    vec2 get_cascade_interval(int cascade) const;
    uvec3 get_cascade_size(int cascade) const;
    void add_defines(std::map<std::string, std::string>& defines) const;

    scene_stage* ss;
private:
    void update(uint32_t frame_index) override;

    push_descriptor_set trace_desc;
    push_descriptor_set gather_desc;
    push_descriptor_set live_counter_desc;
    push_descriptor_set live_dispatcher_desc;
    descriptor_set cascade_descriptors;
    sampler cascade_sampler;
    compute_pipeline trace; // Traces rays and updates probes
    compute_pipeline trace_c0; // Traces rays and updates probes (cascade 0)
    compute_pipeline gather; // Propagates average brightness to missed rays.
    compute_pipeline live_counter;
    compute_pipeline live_dispatcher;

    options opt;
    bool prev_cascades_valid;
    timer stage_timer;
    timer trace_timer;
    timer gather_timer;
    timer live_counter_timer;
    int history_frames;
    size_t dispatch_index_max_size;

    gpu_buffer cascades_metadata;
    vkm<vk::Buffer> dispatch_info_buffer;
    vkm<vk::Buffer> dispatch_size_buffer;
    std::vector<texture> cascades;
    std::vector<texture> alt_cascades;
    std::vector<texture> cascades_visibility;
    std::vector<texture> alt_cascades_visibility;
};

}

#endif


