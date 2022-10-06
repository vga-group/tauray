#include "distribution_strategy.hh"

namespace tr
{

uvec2 get_distribution_target_size(const distribution_params& params)
{
    if(params.primary) return params.size;
    else
    {
        switch(params.strategy)
        {
        case DISTRIBUTION_SHUFFLED_STRIPS:
            return uvec2(params.size.x, (params.count+params.size.x-1)/params.size.x);
        default:
            return get_distribution_render_size(params);
        }
    }
}

uvec2 get_distribution_target_max_size(const distribution_params& params)
{
    switch(params.strategy)
    {
    // Add a case for each dynamically resizable distribution strategy!
    case DISTRIBUTION_SHUFFLED_STRIPS:
        return uvec2(params.size.x, params.size.y);
    default:
        return get_distribution_target_size(params);
    }
}

uvec2 get_distribution_render_size(const distribution_params& params)
{
    switch(params.strategy)
    {
    case DISTRIBUTION_DUPLICATE:
        return params.size;
    case DISTRIBUTION_SCANLINE:
        return uvec2(
            params.size.x,
            (params.size.y-params.index+params.count-1)/params.count
        );
    case DISTRIBUTION_SHUFFLED_STRIPS:
        return uvec2(params.count, 1);
    }
    assert(false);
    return uvec2(0);
}

uvec2 get_ray_count(const distribution_params& params)
{
    switch(params.strategy)
    {
    default:
        return get_distribution_render_size(params);
    case DISTRIBUTION_SHUFFLED_STRIPS:
        return uvec2(params.count, 1);
    }
}

unsigned calculate_shuffled_strips_b(uvec2 size)
{
    unsigned n = size.x * size.y;
    unsigned b = 31;
    while((n >> b) < 128 && b > 0)
        b--;
    return b;
}

//distribution.count = device perf coefficient * number of regions * region size
//distribution.index = index of the first pixel to permute before rendering it
size_t get_region_size(size_t image_size, unsigned int b) //1 dimension size of a region (which is a strip)
{
    size_t n_regions = 1 << b;
    return (image_size + n_regions - 1 ) / n_regions;
}

unsigned calculate_shuffled_strips_pixels_per_device(uvec2 size, float max_ratio)
{
    unsigned b = calculate_shuffled_strips_b(size);
    return ceil(max_ratio * get_region_size(size.x * size.y, b) * (1 << b));
}

distribution_params get_device_distribution_params(
    uvec2 full_image_size,
    distribution_strategy strategy,
    double workload_offset,
    double workload_size,
    unsigned device_index,
    unsigned device_count,
    bool primary
){
    distribution_params d;
    d.strategy = strategy;

    switch(strategy)
    {
    case DISTRIBUTION_DUPLICATE:
    case DISTRIBUTION_SCANLINE:
        d.size = full_image_size;
        d.index = device_index;
        d.count = device_count;
        d.primary = primary;
        break;
    case DISTRIBUTION_SHUFFLED_STRIPS:
        {
            unsigned pixel_count_before = calculate_shuffled_strips_pixels_per_device(
                full_image_size,
                workload_offset
            );
            unsigned pixel_count_after = calculate_shuffled_strips_pixels_per_device(
                full_image_size,
                workload_offset + workload_size
            );
            unsigned pixel_count = pixel_count_after - pixel_count_before;

            d.size = full_image_size;
            d.index = pixel_count_before;
            d.count = pixel_count;
            d.primary = primary;
        }
        break;
    }
    return d;
}

}
