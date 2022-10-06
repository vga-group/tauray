#ifndef TAURAY_DISTRIBUTION_STRATEGY_HH
#define TAURAY_DISTRIBUTION_STRATEGY_HH
#include "math.hh"

namespace tr
{

enum distribution_strategy
{
    // Just duplicate renders on each device == no real distribution
    DISTRIBUTION_DUPLICATE = 0,
    // The output image is divided evenly among rendering devices using
    // interleaved scanlines. The primary device draws directly into a full-size
    // image, others into vertically smaller images. These smaller images are
    // then merged into the full-size image.
    DISTRIBUTION_SCANLINE = 1,
    DISTRIBUTION_SHUFFLED_STRIPS = 2
};

struct distribution_params
{
    uvec2 size = uvec2(0);
    distribution_strategy strategy = DISTRIBUTION_SCANLINE;
    unsigned index = 0;
    unsigned count = 1;
    bool primary = true;
};

// Size of the active portion of the render target.
uvec2 get_distribution_target_size(const distribution_params& params);
// Maximum size of the render target, so that buffer resizing can be avoided.
uvec2 get_distribution_target_max_size(const distribution_params& params);
uvec2 get_distribution_render_size(const distribution_params& params);
uvec2 get_ray_count(const distribution_params& params);

unsigned calculate_shuffled_strips_b(uvec2 size);

distribution_params get_device_distribution_params(
    uvec2 full_image_size,
    distribution_strategy strategy,
    double workload_offset,
    double workload_size,
    unsigned device_index,
    unsigned device_count,
    bool primary
);

}

#endif
