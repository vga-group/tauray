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

vec3 orthogonalize(vec3 a, vec3 b)
{
    return normalize(b - dot(a, b) * a);
}

mat3 orthogonalize(mat3 m)
{
    m[0] = normalize(m[0] - dot(m[0], m[2]) * m[2]);
    m[1] = cross(m[2], m[0]);
    return m;
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

// https://www.pcg-random.org/
uint pcg(inout uint seed)
{
    seed = seed * 747796405u + 2891336453u;
    seed = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737u;
    seed = (seed >> 22) ^ seed;
    return seed;
}

// http://www.jcgt.org/published/0009/03/02/
uvec2 pcg2d(inout uvec2 seed)
{
    seed = seed * 1664525u + 1013904223u;
    seed += seed.yx * 1664525u;
    seed = (seed >> 16) ^ seed;
    seed += seed.yx * 1664525u;
    seed = (seed >> 16) ^ seed;
    return seed;
}

// http://www.jcgt.org/published/0009/03/02/
uvec3 pcg3d(inout uvec3 seed)
{
    seed = seed * 1664525u + 1013904223u;
    seed += seed.yzx * seed.zxy;
    seed = (seed >> 16) ^ seed;
    seed += seed.yzx * seed.zxy;
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

#include "sobol_lookup_table.glsl"

// Returns in groups of 4 dimensions. 'bounce' defines the bounce.
uvec4 generate_sobol_sample(uint index, uint bounce)
{
    // We determined that using this function is actually faster than
    // precalculating the sobol values in a buffer, at least on a 3090.
    uvec4 x = uvec4(0);
    if(bounce >= MAX_SOBOL_BOUNCES)
    {
        x = uvec4(index, bounce, bounce * index, 0);
        return pcg4d(x);
    }

    // This variant seems to be faster on 2080 ti, and equal on 3090.
    for(int bit = findLSB(index); bit < findMSB(index); bit++)
    {
        uint mask = (index >> bit) & 1u;
        if(mask != 0)
            x ^= sobol_lookup_table[bounce * 32 + bit];
    }

    /*
    // This variant seems to be slower on 2080 ti
    for(int bit = 0; bit < 32; bit++)
    {
        uint mask = (index >> bit) & 1u;
        x ^= mask * sobol_lookup_table[bounce * 32 + bit];
    }
    */
    return x;
}

uint get_permutation_n(int n, uint permutation, uint dimension)
{
    uint res = 0;
    for(int i = n-1; i >= 0; --i)
    {
        uint q = permutation % (n-i);
        permutation /= n-i;
        if(i == dimension) res = q;
        if(dimension > i) res += uint(res >= q);
    }
    return res;
}

uvec4 owen_scramble_2d(uvec4 x, uvec4 seed)
{
    x = bitfieldReverse(x);
    // https://psychopath.io/post/2021_01_30_building_a_better_lk_hash
    x ^= x * 0x3D20ADEAu;
    x += seed;
    x *= (seed >> 16u) | 1u;
    x ^= x * 0x05526C56u;
    x ^= x * 0x53A22864u;
    x = bitfieldReverse(x);
    return x;
}

uint owen_scramble_4d(uint x, uint seed)
{
    uint result = 0;
    for(uint i = 0; i < 32; i += 2)
    {
        uvec2 s = uvec2(seed, x & ((~3u)<<i));
        result |= get_permutation_n(4, pcg2d(s).x % 24u, (x >> i) & 3u) << i;
    }
    return result;
}

uint owen_scramble_8d(uint x, uint seed)
{
    uint result = 0;
    for(uint i = 0; i < 32; i += 3)
    {
        uvec2 s = uvec2(seed, x & ((~7u)<<i));
        result |= get_permutation_n(8, pcg2d(s).x % 40320u, (x >> i) & 7u) << i;
    }
    return result;
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

uint morton_2d(uvec2 x)
{
    x &= 0x0000ffff;
    x = (x ^ (x << 8u)) & 0x00ff00ffu;
    x = (x ^ (x << 4u)) & 0x0f0f0f0fu;
    x = (x ^ (x << 2u)) & 0x33333333u;
    x = (x ^ (x << 1u)) & 0x55555555u;
    return x.x + 2u * x.y;
}

uint morton_3d(uvec3 x)
{
    x &= 0x000003ffu;
    x = (x ^ (x << 16u)) & 0xff0000ffu;
    x = (x ^ (x << 8u)) & 0x0300f00fu;
    x = (x ^ (x << 4u)) & 0x030c30c3u;
    x = (x ^ (x << 2u)) & 0x09249249u;
    return x.x + 2u * x.y + 4u * x.z;
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

vec3 sample_hemisphere(vec2 u)
{
    float cos_theta = u.x;
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

// https://www.graphics.cornell.edu/pubs/1995/Arv95c.pdf
vec3 sample_spherical_triangle(
    vec2 xi, vec3 A, vec3 B, vec3 C, out float solid_angle
){
    vec3 nA = normalize(A);
    vec3 nB = normalize(B);
    vec3 nC = normalize(C);

    vec3 cos_side = clamp(vec3(dot(nB, nC), dot(nA, nC), dot(nA, nB)), vec3(-1.0f), vec3(1.0f));
    solid_angle = 2.0f * atan(abs(dot(nA, cross(nB, nC))), (1.0f + cos_side.x + cos_side.y + cos_side.z));

    float chosen_split = xi.x * solid_angle;

    float cos_alpha = clamp(dot(orthogonalize(nA, nB-nA), orthogonalize(nA, nC-nA)), -1.0f, 1.0f);
    float alpha = acos(cos_alpha);
    float sin_alpha = sin(alpha);

    float s = sin(chosen_split - alpha);
    float t = cos(chosen_split - alpha);
    float u = t - cos_alpha;
    float v = s + sin_alpha * cos_side.z;

    float q = ((v*t - u*s) * cos_alpha - v)/((v*s + u*t) * sin_alpha);

    vec3 Ch = q * nA + sqrt(1 - q*q) * normalize(nC - cos_side.y * nA);
    float d = dot(Ch, nB);
    float z = 1 - xi.y + d * xi.y;
    return z * nB + sqrt(1 - z*z) * normalize(Ch - d * nB);
}

float spherical_triangle_solid_angle(
    vec3 A, vec3 B, vec3 C
){
    vec3 nA = normalize(A);
    vec3 nB = normalize(B);
    vec3 nC = normalize(C);

    return 2.0f * atan(
        abs(dot(nA, cross(nB, nC))),
        (1.0f+dot(nA,nB)+dot(nB,nC)+dot(nA,nC))
    );
}

float ray_plane_intersection_dist(
    vec3 dir, vec3 A, vec3 B, vec3 C
){
    vec4 plane = vec4(normalize(cross(A-B, A-C)), 0);
    plane.w = dot(A, plane.xyz);
    return abs(plane.w / dot(plane.xyz, dir));
}

vec3 get_barycentric_coords(vec3 p, vec3 A, vec3 B, vec3 C)
{
    vec3 ba = B - A, ca = C - A, pa = p - A;
    float bb = dot(ba, ba);
    float bc = dot(ba, ca);
    float cc = dot(ca, ca);
    float pb = dot(pa, ba);
    float pc = dot(pa, ca);
    float denom = 1.0f / (bb * cc - bc * bc);

    vec3 bary;
    bary.y = (cc * pb - bc * pc) * denom;
    bary.z = (bb * pc - bc * pb) * denom;
    bary.x = 1.0f - bary.y - bary.z;
    return bary;
}

#endif
