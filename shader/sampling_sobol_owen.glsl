#ifndef SOBOL_OWEN_SAMPLER_GLSL
#define SOBOL_OWEN_SAMPLER_GLSL

#include "random_sampler.glsl"

struct sobol_sampler
{
    uvec4 seed;
};

// https://psychopath.io/post/2021_01_30_building_a_better_lk_hash
uvec4 lk_style_hash(uvec4 x, uvec4 seed)
{
    x ^= x * 0x3D20ADEAu;
    x += seed;
    x *= (seed >> 16u) | 1u;
    x ^= x * 0x05526C56u;
    x ^= x * 0x53A22864u;
    return x;
}

uvec4 sobol(uint index)
{
    // We determined that using this function is actually faster than
    // precalculating the sobol values in a buffer, at least on a 3090.
    const uvec4 directions[32] = uvec4[](
        uvec4(0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u),
        uvec4(0x40000000u, 0xC0000000u, 0xC0000000u, 0xC0000000u),
        uvec4(0x20000000u, 0xA0000000u, 0x60000000u, 0x20000000u),
        uvec4(0x10000000u, 0xF0000000u, 0x90000000u, 0x50000000u),
        uvec4(0x08000000u, 0x88000000u, 0xE8000000u, 0xF8000000u),
        uvec4(0x04000000u, 0xCC000000u, 0x5C000000u, 0x74000000u),
        uvec4(0x02000000u, 0xAA000000u, 0x8E000000u, 0xA2000000u),
        uvec4(0x01000000u, 0xFF000000u, 0xC5000000u, 0x93000000u),
        uvec4(0x00800000u, 0x80800000u, 0x68800000u, 0xD8800000u),
        uvec4(0x00400000u, 0xC0C00000u, 0x9CC00000u, 0x25400000u),
        uvec4(0x00200000u, 0xA0A00000u, 0xEE600000u, 0x59E00000u),
        uvec4(0x00100000u, 0xF0F00000u, 0x55900000u, 0xE6D00000u),
        uvec4(0x00080000u, 0x88880000u, 0x80680000u, 0x78080000u),
        uvec4(0x00040000u, 0xCCCC0000u, 0xC09C0000u, 0xB40C0000u),
        uvec4(0x00020000u, 0xAAAA0000u, 0x60EE0000u, 0x82020000u),
        uvec4(0x00010000u, 0xFFFF0000u, 0x90550000u, 0xC3050000u),
        uvec4(0x00008000u, 0x80008000u, 0xE8808000u, 0x208F8000u),
        uvec4(0x00004000u, 0xC000C000u, 0x5CC0C000u, 0x51474000u),
        uvec4(0x00002000u, 0xA000A000u, 0x8E606000u, 0xFBEA2000u),
        uvec4(0x00001000u, 0xF000F000u, 0xC5909000u, 0x75D93000u),
        uvec4(0x00000800u, 0x88008800u, 0x6868E800u, 0xA0858800u),
        uvec4(0x00000400u, 0xCC00CC00u, 0x9C9C5C00u, 0x914E5400u),
        uvec4(0x00000200u, 0xAA00AA00u, 0xEEEE8E00u, 0xDBE79E00u),
        uvec4(0x00000100u, 0xFF00FF00u, 0x5555C500u, 0x25DB6D00u),
        uvec4(0x00000080u, 0x80808080u, 0x8000E880u, 0x58800080u),
        uvec4(0x00000040u, 0xC0C0C0C0u, 0xC0005CC0u, 0xE54000C0u),
        uvec4(0x00000020u, 0xA0A0A0A0u, 0x60008E60u, 0x79E00020u),
        uvec4(0x00000010u, 0xF0F0F0F0u, 0x9000C590u, 0xB6D00050u),
        uvec4(0x00000008u, 0x88888888u, 0xE8006868u, 0x800800F8u),
        uvec4(0x00000004u, 0xCCCCCCCCu, 0x5C009C9Cu, 0xC00C0074u),
        uvec4(0x00000002u, 0xAAAAAAAAu, 0x8E00EEEEu, 0x200200A2u),
        uvec4(0x00000001u, 0xFFFFFFFFu, 0xC5005555u, 0x50050093u)
    );

    uvec4 x = uvec4(0u);
    for(int bit = 0; bit < 32; bit++)
    {
        uint mask = (index >> bit) & 1u;
        x ^= mask * directions[bit];
    }
    return x;
}

uvec4 nested_uniform_scramble_base2(uvec4 x, uvec4 seed)
{
    x = bitfieldReverse(x);
    x = lk_style_hash(x, seed);
    x = bitfieldReverse(x);
    return x;
}

uvec4 get_shuffled_scrambled_sobol_pt_uint(sobol_sampler ssampler, uint bounce)
{
    uint index = ssampler.seed.w;
    ssampler.seed.w = bounce;
    uint shuffled_index = nested_uniform_scramble_base2(
        uvec4(index), pcg4d(ssampler.seed)
    ).x;
    ssampler.seed.w = index;

    return nested_uniform_scramble_base2(sobol(shuffled_index), ssampler.seed.yzwx);
}

vec4 get_shuffled_scrambled_sobol_pt(sobol_sampler ssampler, uint bounce)
{
    return ldexp(vec4(get_shuffled_scrambled_sobol_pt_uint(ssampler, bounce)), ivec4(-32));
}

sobol_sampler init_sobol_sampler(uvec4 seed)
{
    sobol_sampler ssampler;
    ssampler.seed = pcg4d(seed);
    return ssampler;
}
#endif
