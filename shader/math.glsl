#ifndef MATH_GLSL
#define MATH_GLSL

#define M_PI 3.14159265359
#define M_1_SQRT3 0.57735026918962576451
#define SQRT2 1.41421356237
#define SQRT3 1.73205080756
#define GOLDEN_RATIO 1.61803398874989484820
#define FLT_MAX 3.402823466e+38

// Creates an arbitrary but valid tangent space. Not suitable for normal
// mapping, good for some isotropic shading calculations.
// 'normal' must be normalized.
// In the created tangent space, normal corresponds to the Z axis.
mat3 create_tangent_space(vec3 normal)
{
    vec3 major;
    if(abs(normal.x) < M_1_SQRT3) major = vec3(1,0,0);
    else if(abs(normal.y) < M_1_SQRT3) major = vec3(0,1,0);
    else major = vec3(0,0,1);

    vec3 tangent = normalize(cross(normal, major));
    vec3 bitangent = cross(normal, tangent);

    return mat3(tangent, bitangent, normal);
}

uint find_bit_by_cardinality(uint r, uint v)
{
    uint c = 0;
    uvec4 parts = uvec4(v, v>>8u, v>>16u, v>>24u)&0xFFu;
    ivec4 bits = bitCount(parts);

    if(r >= bits.x + bits.y) { c += 16; r -= bits.x + bits.y; bits.x = bits.z; }
    if(r >= bits.x) { c += 8; r -= bits.x;}

    parts = (uvec4(v, v>>2u, v>>4u, v>>6u)>>c)&0x3u;
    bits = bitCount(parts);

    if(r >= bits.x + bits.y) { c += 4; r -= bits.x + bits.y; bits.x = bits.z;}
    if(r >= bits.x) { c += 2; r -= bits.x;}
    if(r >= ((v >> c)&1)) { c++; }

    return c;
}

uint find_bit_by_cardinality(uint bit, uvec4 mask)
{
    ivec4 count = bitCount(mask);
    uint low = count.x;
    uint i = 0;
    if(bit >= count.x + count.y) { bit -= count.x + count.y; i = 2; low = count.z; }
    if(bit >= low) { bit -= low; i++; }
    return i * 32u + find_bit_by_cardinality(bit, mask[i]);
}
//
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

// Based on CC0 code from https://gist.github.com/juliusikkala/1b2d9cee959c16c7c7ccc9cb6fb50754
vec2 sample_regular_polygon(vec2 u, float angle, uint sides)
{
    float side = floor(u.x * sides);
    u.x = fract(u.x * sides);
    float side_radians = (2.0f*M_PI)/sides;
    float a1 = side_radians * side + angle;
    float a2 = side_radians * (side + 1) + angle;
    vec2 b = vec2(sin(a1), cos(a1));
    vec2 c = vec2(sin(a2), cos(a2));
    u = u.x+u.y > 1 ? 1 - u : u;
    return b * u.x + c * u.y;
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

vec3 even_sample_sphere(int sample_index, int sample_count, vec2 off)
{
    float o = (float(sample_index)+off.x) * 0.38196601125;
    return sample_sphere(vec2((float(sample_index) + off.y)/float(sample_count), o));
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
