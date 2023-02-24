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

#endif
