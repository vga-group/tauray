#ifndef RANDOM_SAMPLER_GLSL
#define RANDOM_SAMPLER_GLSL

struct random_sampler
{
    uvec4 seed;
};

// https://www.pcg-random.org/
uint pcg(inout uint seed)
{
    seed = seed * 747796405u + 2891336453u;
    seed = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737u;
    seed = (seed >> 22) ^ seed;
    return seed;
}

// http://www.jcgt.org/published/0009/03/02/
uvec4 pcg4d(inout uvec4 seed)
{
    seed = seed * 1664525u + 1013904223u;
    seed += seed.yzxy * seed.wxyz;
    seed = (seed >> 16) ^ seed;
    seed += seed.yzxy * seed.wxyz;
    return seed;
}

random_sampler init_random_sampler(
    uvec4 coord, uvec3 size
){
    random_sampler rsampler;
    rsampler.seed = pcg4d(coord);
    return rsampler;
}

// Only the 4d version is performed, because fewer dimensioned use cases can
// just take the first N values that they need and there should be almost no
// extra cost due to SIMD stuff.
// Returns 4 0-1 uncorrelated uniform random values.
vec4 generate_uniform_random(inout random_sampler rsampler)
{
    return ldexp(vec4(pcg4d(rsampler.seed)), ivec4(-32));
}

float generate_single_uniform_random(inout uint seed)
{
    return ldexp(float(pcg(seed)), -32);
}

// Gives a random color, useful for debugging.
vec3 debug_color(uvec4 param)
{
    return ldexp(vec3(pcg4d(param).xyz), ivec3(-32));
}

#endif
