#ifndef RANDOM_SAMPLER_GLSL
#define RANDOM_SAMPLER_GLSL

#include "math.glsl"

struct random_sampler
{
    uvec4 seed;
};

random_sampler init_random_sampler(uvec4 coord)
{
    random_sampler rsampler;
    rsampler.seed = coord;
    rsampler.seed.y ^= pcg(rsampler.seed.x);
    rsampler.seed.z ^= pcg(rsampler.seed.y);
    rsampler.seed.w ^= pcg(rsampler.seed.z);
    return rsampler;
}

// Only the 4d version is performed, because fewer dimensioned use cases can
// just take the first N values that they need and there should be almost no
// extra cost due to SIMD stuff.
// Returns 4 uncorrelated uniform random unsigned integers.
uvec4 generate_uniform_random_uint(inout random_sampler rsampler)
{
    return pcg4d(rsampler.seed);
}

vec4 generate_uniform_random(inout random_sampler rsampler)
{
    return vec4(generate_uniform_random_uint(rsampler)) * INV_UINT32_MAX;
}

float generate_single_uniform_random(inout uint seed)
{
    return float(pcg(seed)) * INV_UINT32_MAX;
}

vec4 generate_single_uniform_random(inout uvec4 seed)
{
    return vec4(pcg4d(seed)) * INV_UINT32_MAX;
}

// Gives a random color, useful for debugging.
vec3 debug_color(uvec4 param)
{
    return vec3(pcg4d(param).xyz) * INV_UINT32_MAX;
}

#endif
