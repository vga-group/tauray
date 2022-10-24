#ifndef SVGF_GLSL
#define SVGF_GLSL

const float gaussian_kernel[2][2] = {
    { 1.0 / 4.0, 1.0 / 8.0  },
    { 1.0 / 8.0, 1.0 / 16.0 }
};
layout(push_constant) uniform push_constant_buffer
{
    ivec2 size;
    int level;
    int iteration_count;
    int spec_iteration_count;
    int atrous_kernel_radius;
    float sigma_n;
    float sigma_z;
    float sigma_l;
    float temporal_alpha_color;
    float temporal_alpha_moments;
} control;


#endif