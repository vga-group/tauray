#ifndef SOBOL_OWEN_SAMPLER_GLSL
#define SOBOL_OWEN_SAMPLER_GLSL

#include "random_sampler.glsl"

struct sobol_owen_sampler
{
    uvec4 seed;
};

uvec4 get_shuffled_scrambled_sobol_pt_uint(sobol_owen_sampler ssampler, uint bounce)
{
    uint index = ssampler.seed.w;
    ssampler.seed.w = bounce;
    uint shuffled_index = owen_scramble_2d(
        uvec4(index), pcg4d(ssampler.seed)
    ).x;
    ssampler.seed.w = index;

    return owen_scramble_2d(generate_sobol_sample(shuffled_index, bounce), ssampler.seed.yzwx);
}

sobol_owen_sampler init_sobol_owen_sampler(uvec4 seed)
{
    sobol_owen_sampler ssampler;
    ssampler.seed = pcg4d(seed);
    return ssampler;
}
#endif
