#ifndef LTC_GLSL
#define LTC_GLSL
#include "math.glsl"

// https://advances.realtimerendering.com/s2016/s2016_ltc_rnd.pdf

// If the angle between a and b is theta, this computes
// theta / sin(theta) with high precision.
float angle_per_sin_angle(vec3 a, vec3 b)
{
    float cos_theta = dot(a, b);
    float act = abs(cos_theta);
    float num = fma(act, 0.01436124, 0.49788271);
    num = fma(act, num, 0.86266946);
    float denom = fma(4.18814979+act, act, 3.45068008);
    float theta_per_sin_theta = num/denom;
    if(cos_theta < 0.0)
        theta_per_sin_theta = 0.5 * inversesqrt(1.0 - act*act) - theta_per_sin_theta;
    return theta_per_sin_theta;
}

// a and b are unit vectors on the +z hemisphere.
float edge_integral(vec3 a, vec3 b)
{
    return angle_per_sin_angle(a, b) * (a.x * b.y - a.y * b.x);
}

vec3 edge_vector_form_factor(vec3 a, vec3 b)
{
    return angle_per_sin_angle(a, b) * cross(a, b);
}

// If nothing else, transform should take vertices to tangent space. It can also
// do the LTC transforms at the same time as needed.
float cosine_hemisphere_poly_light(mat3 transform, vec3 pos, vec3 v[4], bool double_sided)
{
    vec3 last = normalize((v[v.length()-1]-pos) * transform);
    vec3 prev = last;
    vec3 form_factor = vec3(0);
    bool behind = false;

    [[unroll]] for(uint i = 0; i < v.length()-1; ++i)
    {
        vec3 d = normalize((v[i]-pos) * transform);
        form_factor += edge_vector_form_factor(prev, d);

        if(i == 1)
            behind = dot(prev, cross(d - prev, last - prev)) < 0.0;

        prev = d;
    }
    form_factor += edge_vector_form_factor(prev, last);

    // Approx horizon clipping
    float len = length(form_factor);
    float z = behind ? -form_factor.z : form_factor.z;
    float irradiance = max((len*len+z)/(len+1.0), 0.0);
    return behind && !double_sided ? 0.0 : irradiance;
}

float cosine_hemisphere_poly_light_always_front(mat3 transform, vec3 pos, vec3 v[4], bool double_sided)
{
    vec3 last = normalize((v[v.length()-1]-pos) * transform);
    vec3 prev = last;
    vec3 form_factor = vec3(0);

    [[unroll]] for(uint i = 0; i < v.length()-1; ++i)
    {
        vec3 d = normalize((v[i]-pos) * transform);
        form_factor += edge_vector_form_factor(prev, d);
        prev = d;
    }
    form_factor += edge_vector_form_factor(prev, last);

    // Approx horizon clipping
    float len = length(form_factor);
    float irradiance = max((len*len+form_factor.z)/(len+1.0), 0.0);
    return irradiance;
}

#endif
