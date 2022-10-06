#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 1) rayPayloadInEXT vec3 shadow_transmittance;

void main()
{
    //shadow_transmittance = vec3(1.0f);
}

