
#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 1) rayPayloadInEXT float shadow_visibility;

void main()
{
    // If we're here, we hit an opaque object that didn't go to the any hit
    // shader.
    shadow_visibility = 0.0f;
}
