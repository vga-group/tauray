#ifndef COLOR_GLSL
#define COLOR_GLSL

// As described in
// https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation,
// but branchless and SIMD-optimized.
vec3 inverse_srgb_correction(vec3 col)
{
    vec3 low = col * 0.07739938f;
    vec3 high = pow(fma(col, vec3(0.94786729f), vec3(0.05213270f)), vec3(2.4f));
    return mix(low, high, lessThan(vec3(0.04045f), col));
}

float rgb_to_luminance(vec3 col)
{
    return dot(col, vec3(0.2126, 0.7152, 0.0722));
}

uint rgb_to_r9g9b9e5(vec3 color)
{
    ivec3 ex;
    frexp(color, ex);
    int e = clamp(max(ex.r, max(ex.g, ex.b)), -16, 15);
    ivec3 icolor = clamp(ivec3(
        floor(ldexp(color, ivec3(-e)) * 512.0f)
    ), ivec3(0), ivec3(511));
    return icolor.r|(icolor.g<<9)|(icolor.b<<18)|((e+16)<<27);
}

vec3 r9g9b9e5_to_rgb(uint rgbe)
{
    ivec4 icolor = ivec4(rgbe, rgbe>>9, rgbe>>18, rgbe>>27) & 0x1FF;
    return ldexp(vec3(icolor) * (1.0f / 512.0f), ivec3(icolor.a-16));
}

#endif
