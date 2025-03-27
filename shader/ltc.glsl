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
    float num = 0.01436124 * act + 0.49788271;
    num = num * act + 0.86266946;
    float denom = act + 4.18814979;
    denom = denom * act + 3.45068008;

    float theta_per_sin_theta = num/denom;
    return cos_theta < 0.0 ? 0.5 * inversesqrt(max(1.0 - act*act, 1e-6f)) - theta_per_sin_theta : theta_per_sin_theta;
}

float angle_per_sin_angle_fast(vec3 a, vec3 b)
{
    float cos_theta = dot(a, b);
    float num = -0.709693 * cos_theta + 3.787823;
    num = num * cos_theta + 20.448902;
    num = num * cos_theta + 15.951386;
    return inversesqrt(num);
}

// a and b are unit vectors on the +z hemisphere.
float edge_integral(vec3 a, vec3 b)
{
    return angle_per_sin_angle_fast(a, b) * (a.x * b.y - a.y * b.x);
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

float cosine_hemisphere_poly_light_always_front(mat3 transform, vec3 pos, vec3 v[4])
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

// This has been fit in a very specific manner that ensures that the horizon
// matches the real horizon. This has IMPLICATIONS for quality, but is useful
// for importance sampling.
vec3 ltc_ggx_transform(
    float vdotn,
    float roughness
){
    float x = vdotn;
    float y = sqrt(roughness);

    float x2 = x*x;
    float x4 = x2*x2;
    float x5 = x4*x;

    float y2 = roughness;
    float y4 = roughness * roughness;
    float y3 = y2 * y;
    float y5 = y4 * y;
    float y8 = y4 * y4;

    float a = 0.439155 * y2 - 0.452528;
    a = a * y4 + 0.013373;
    a = a * x4 + y4 * (1.479211 * y2 - 2.465839) + 0.986627;
    a = a * x2 + y4 * (-1.918366 * y2 + 2.918366);

    float b = y3 * (11.695919 * y5 + 1.158242) - 12.854162;
    b = b * x - 23.287031 * y8 + 23.287031;
    b = b * x + 11.372538 * y8 - 11.372538;
    b = b * x5 + y3 * (0.980265 * y5 - 1.919935) + 0.939669;
    b = b * x + y3 * (0.761692 - 0.761692 * y5);

    float c = 1.160692 * y + 1.702139;
    c = c * y + x * (3.131133 * x - 5.134411);
    c = c * y + 0.972169;
    c = x * (-2.467024 * x + 4.108407) + c;
    c = c * y2;

    return vec3(a, b, c);
}

// xyz: transformed vector, w: jacobian
vec4 ltc_transform_dir(vec3 transform, vec3 dir, out float inv_len)
{
    vec3 new_dir = vec3(
        dir.x * transform.x + dir.z * transform.y, dir.y, dir.z * transform.z
    );
    float transformed_dir_inv_len = inversesqrt(dot(new_dir.xyz, new_dir.xyz));
    inv_len = transformed_dir_inv_len;
    float inv_len3 =
        transformed_dir_inv_len*
        transformed_dir_inv_len*
        transformed_dir_inv_len;
    float det = abs(transform.x * transform.z);
    return vec4(new_dir*transformed_dir_inv_len, det * inv_len3);
}

vec3 ltc_transform_dir3(vec3 transform, vec3 dir)
{
    vec3 new_dir = vec3(
        dir.x * transform.x + dir.z * transform.y,
        dir.y,
        dir.z * transform.z
    );
    return normalize(new_dir);
}

float ggx_albedo(float vdotn, float roughness)
{
    float x = vdotn;
    float y = sqrt(roughness);
    float y2 = roughness;
    float xy = vdotn * y;
    float xy3 = xy * xy * xy;
    float num = 0.356334 * xy3 - 2.189899;
    num = num * xy + y*(y*(30.106651 * y - 72.778887) + 57.887217) - 11.910812;
    num = num * x + y*(y*(-34.517973 * y + 86.062015) - 69.862862) + 16.205007;
    num = num * x + y*(y*(7.556232 * y - 19.236401) + 16.191891) - 4.561471;
    num = num * y2 + 1.0;
    return clamp(num, 0.0f, 1.0f);
}

float ggx_fresnel(float vdotn, float roughness)
{
    float x = vdotn;
    float x2 = x * x;
    float y = sqrt(roughness);
    float y2 = roughness;
    float num = 0.398996 * y + 0.799070;
    num = num * x - 1.799070;
    num = num * x + -0.398996 * y + 1.0;
    float denom = 25.826333 * x - 15.469869 * y;
    denom = denom * x + 16.436949 * y2 + 1;
    return clamp(num*(1.0f/denom), 0.0f, 1.0f);
}

#endif
