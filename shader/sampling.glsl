#ifndef SAMPLING_GLSL
#define SAMPLING_GLSL
#include "math.glsl"

#ifdef SAMPLING_DATA_BINDING
layout(binding = SAMPLING_DATA_BINDING, set = 0) uniform sampling_data_buffer
{
    // xyz: size, w: sampling_start_counter
    uvec4 size_start_counter;
} sampling_data;
#endif

#include "random_sampler.glsl"
#include "sampling_sobol_owen.glsl"

struct local_sampler
{
    random_sampler rs;
#ifdef SOBOL_OWEN_SAMPLING
    sobol_sampler ss;
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

#ifdef SOBOL_OWEN_SAMPLING
    ls.ss = init_sobol_sampler(coord);
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

// Generates a 2D sample in xy, 1D samples in z and w. This sampler is only
// called once per bounce, so you can safely employ stratification or some
// other scheme, as long as you take bounce_index into account.
vec4 generate_ray_sample(inout local_sampler ls, uint bounce_index)
{
#ifdef SOBOL_OWEN_SAMPLING
    return get_shuffled_scrambled_sobol_pt(ls.ss, bounce_index);
#else
    return generate_uniform_random(ls.rs);
#endif
}

uvec4 generate_ray_sample_uint(inout local_sampler ls, uint bounce_index)
{
#ifdef SOBOL_OWEN_SAMPLING
    return get_shuffled_scrambled_sobol_pt_uint(ls.ss, bounce_index);
#else
    return generate_uniform_random_uint(ls.rs);
#endif
}

// Uniformly samples a disk, but strategically mapped for preserving
// stratification better.
vec2 sample_concentric_disk(vec2 u)
{
    vec2 uo = 2.0f * u - 1.0f;
    vec2 abs_uo = abs(uo);

    if(all(lessThan(abs_uo, vec2(0.0001f))))
        return vec2(0);

    vec2 rt = (abs_uo.x > abs_uo.y) ?
        vec2(uo.x, M_PI/4 * (uo.y / uo.x)) :
        vec2(uo.y, M_PI/2 - M_PI/4 * (uo.x / uo.y));
    return rt.x * vec2(cos(rt.y), sin(rt.y));
}

float sample_blackman_harris(float u)
{
    bool flip = u > 0.5;
    u = flip ? 1 - u : u;
    vec4 v = vec4(-0.33518669f, -0.51620529f, 1.87406934f, -0.66315464f) *
        pow(vec4(u), vec4(0.5f, 0.3333333333f, 0.25f, 0.2f));
    float s = 0.29627329f * u + v.x + v.y + v.z + v.w;
    return flip ? 1 - s : s;
}

vec2 sample_blackman_harris_concentric_disk(vec2 u)
{
    vec2 uo = 2.0f * u - 1.0f;
    vec2 abs_uo = abs(uo);

    if(all(lessThan(abs_uo, vec2(0.0001f))))
        return vec2(0);

    vec2 rt = (abs_uo.x > abs_uo.y) ?
        vec2(u.x, M_PI/4 * (uo.y / uo.x)) :
        vec2(u.y, M_PI/2 - M_PI/4 * (uo.x / uo.y));
    return (2.0f * sample_blackman_harris(rt.x) - 1.0f) * vec2(cos(rt.y), sin(rt.y));
}

vec3 sample_cosine_hemisphere(vec2 u)
{
    vec2 d = sample_concentric_disk(u);
    return vec3(d, sqrt(max(0, 1 - dot(d, d))));
}

float pdf_cosine_hemisphere(vec3 dir)
{
    return dir.z * (1.0/M_PI);
}

vec3 sample_sphere(vec2 u)
{
    float cos_theta = 2 * u.x - 1;
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    float phi = u.y * 2 * M_PI;
    return vec3(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );
}

vec3 sample_blackman_harris_ball(vec3 u)
{
    vec3 v = sample_sphere(u.xy);
    float r = pow(abs(2.0f * sample_blackman_harris(u.z) - 1.0f), 1.0f/3.0f);
    return r * v;
}

vec3 even_sample_sphere(int sample_index, int sample_count, float off)
{
    float o = float(sample_index)+off;
    return sample_sphere(vec2(o/float(sample_count), o * GOLDEN_RATIO));
}

vec3 sample_cone(vec2 u, vec3 dir, float cos_theta_min)
{
    float cos_theta = mix(1.0f, cos_theta_min, u.x);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    float phi = u.y * 2 * M_PI;
    vec3 o = create_tangent_space(dir) * vec3(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );
    // The tangent space transform introduces some numeric inaccuracy, which can
    // cause sampled rays to be outside of the cone. That is never allowed, we
    // fix it by just returning the original direction in those cases. Since
    // it's just caused by numeric inaccuracies, the bias caused by this should
    // be insignificant ;)
    return dot(o, dir) <= cos_theta_min ? dir : o;
}

#endif
