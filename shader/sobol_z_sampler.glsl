#ifndef SOBOL_Z_SAMPLER_GLSL
#define SOBOL_Z_SAMPLER_GLSL

#include "math.glsl"

// A Sobol sampler with indices drawn from the Morton / Z-curve. Similar to
// what is described in http://abdallagafar.com/publications/zsampler/ and
// https://psychopath.io/post/2022_07_24_owen_scrambling_based_dithered_blue_noise_sampling,
// but simplified for Tauray usage.
struct sobol_z_sampler
{
    uint sobol_index;
};

#ifndef SOBOL_Z_ORDER_CURVE_DIMS
#define SOBOL_Z_ORDER_CURVE_DIMS 3
#endif

uvec4 get_sobol_z_sample_uint(sobol_z_sampler ssampler, uint bounce)
{
    return generate_sobol_sample(ssampler.sobol_index, bounce);
}

sobol_z_sampler init_sobol_z_sampler(uvec4 coord)
{
    sobol_z_sampler ssampler;
#if SOBOL_Z_ORDER_CURVE_DIMS == 3
    // Worse blue noise, better accumulation
    // The seed for the Owen scrambling is that because the morton curve is
    // limited to 1024 in depth (= in paths sampled). So, every 1024 samples,
    // we scramble differently.
    ssampler.sobol_index = owen_scramble_8d(morton_3d(coord.xyw), coord.w>>10u);
#else
    // Better blue noise, worse accumulation
    // The path index is used to scramble differently.
    ssampler.sobol_index = owen_scramble_4d(morton_2d(coord.xy), coord.w);
#endif
    return ssampler;
}
#endif

