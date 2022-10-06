#ifndef FORWARD_GLSL
#define FORWARD_GLSL
#define USE_SHADOW_MAPPING
#include "scene.glsl"

layout(push_constant) uniform push_constant_buffer
{
    uint instance_id;
    int pcf_samples;
    int omni_pcf_samples;
    int pcss_samples;
    float pcss_minimum_radius;
    float noise_scale;
    vec2 shadow_map_atlas_pixel_margin;
    vec3 ambient_color;
} control;

#include "shadow_mapping.glsl"

#ifdef BRDF_INTEGRATION_BINDING
layout(binding = BRDF_INTEGRATION_BINDING, set = 0) uniform sampler2D brdf_integration;
#endif

#endif
