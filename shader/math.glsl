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

#endif
