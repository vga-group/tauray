#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#define PAYLOAD_IN
#include "rt_common_payload.glsl"

void main()
{
    // If we're here, we hit an opaque object that didn't go to the any hit
    // shader.
    shadow_visibility = 0.0f;
}

