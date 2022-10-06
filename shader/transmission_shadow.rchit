#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 1) rayPayloadInEXT vec3 shadow_transmittance;

void main()
{
    // If we're here, we hit an opaque object that didn't go to the any hit
    // shader.
    shadow_transmittance = vec3(0.0f);
}
