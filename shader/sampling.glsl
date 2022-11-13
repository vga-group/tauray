#ifndef SAMPLING_GLSL
#define SAMPLING_GLSL
#include "math.glsl"

#include "random_sampler.glsl"

#ifdef SAMPLING_DATA_BINDING
layout(binding = SAMPLING_DATA_BINDING, set = 0) uniform sampling_data_buffer
{
    // xyz: size, w: sampling_start_counter
    uvec4 size_start_counter;
    uint rng_seed;
} sampling_data;
#endif

#include "sobol_z_sampler.glsl"
#include "sobol_owen_sampler.glsl"

struct local_sampler
{
    random_sampler rs;
#if defined(USE_SOBOL_Z_ORDER_SAMPLING)
    sobol_z_sampler ss;
#elif defined(USE_SOBOL_OWEN_SAMPLING)
    sobol_owen_sampler ss;
#endif
};


#ifdef SAMPLING_DATA_BINDING
// coord.xy = pixel screen coordinate
// coord.z = viewport index
// coord.w = time/path index
// size.xy = screen size (width/height)
// size.z = total viewport count
local_sampler init_local_sampler(uvec4 coord)
{
    local_sampler ls;
    coord.w += sampling_data.size_start_counter.w;
    coord.z += sampling_data.rng_seed;

#if defined(USE_SOBOL_Z_ORDER_SAMPLING)
    ls.ss = init_sobol_z_sampler(coord);
#elif defined(USE_SOBOL_OWEN_SAMPLING)
    ls.ss = init_sobol_owen_sampler(coord);
#endif
    ls.rs = init_random_sampler(coord, sampling_data.size_start_counter.xyz);
    return ls;
}
#endif

float generate_alpha_sample(inout local_sampler ls)
{
    return generate_uniform_random(ls.rs).x;
}

vec2 generate_film_sample(inout local_sampler ls)
{
    return generate_uniform_random(ls.rs).xy;
}

vec3 generate_spatial_sample(inout local_sampler ls)
{
    return generate_uniform_random(ls.rs).xyz;
}

#ifdef SAMPLING_DATA_BINDING

// Generates a 2D sample in xy, 1D samples in z and w. This sampler is only
// called once per bounce, so you can safely employ stratification or some
// other scheme, as long as you take bounce_index into account.
uvec4 generate_ray_sample_uint(inout local_sampler ls, uint bounce_index)
{
#if defined(USE_SOBOL_Z_ORDER_SAMPLING)
    return get_sobol_z_sample_uint(ls.ss, bounce_index);
#elif defined(USE_SOBOL_OWEN_SAMPLING)
    return get_shuffled_scrambled_sobol_pt_uint(ls.ss, bounce_index);
#else
    return generate_uniform_random_uint(ls.rs);
#endif
}

vec4 generate_ray_sample(inout local_sampler ls, uint bounce_index)
{
    return ldexp(vec4(generate_ray_sample_uint(ls, bounce_index)), ivec4(-32));
}

#endif

#endif
