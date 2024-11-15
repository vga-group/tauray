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
    struct options
    {
        aabb volume;
        uint32_t log2_resolution = 8;
        // Negative: automatic, based on volume and resolution
        float t0 = -1.0f;

        bool jitter_rays = false;
        // Only effective with jittering enabled
        float temporal_ratio = 0.01f;
        // Get lighting for current pass from previous pass. Requires double
        // the memory!
        bool recursive = false;

        // Use rasterizer-style direct illumination (requires shadow maps to be
        // available). If disabled, DI is path traced.
        bool use_raster_di = false;

        // If path traced DI is enabled, this controls how many DI rays are done
        // per pass.
        size_t di_samples = 1;
    };

    radiance_cascades_stage(
        device& dev,
        scene_stage& ss,
        const options& opt
    );

    descriptor_set& get_descriptors();
    size_t get_cascade_count() const;
    uvec3 get_cascade_size(int cascade) const;

    scene_stage* ss;
private:
    void update(uint32_t frame_index) override;

    push_descriptor_set trace_desc;
    push_descriptor_set gather_desc;
    descriptor_set cascade_descriptors;
    sampler cascade_sampler;
    compute_pipeline trace; // Traces rays and updates probes
    compute_pipeline gather; // Propagates average brightness to missed rays.

    options opt;
    bool prev_cascades_valid;
    timer stage_timer;

    gpu_buffer cascades_metadata;
    std::vector<texture> cascades;
    std::vector<texture> alt_cascades;
};

}

#endif


