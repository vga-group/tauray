#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 1) rayPayloadInEXT bool is_occluded;

void main()
{
    is_occluded = false;
}

