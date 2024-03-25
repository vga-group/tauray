#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#define PAYLOAD_IN
#include "rt_common_payload.glsl"

void main()
{
    payload.instance_id = -1;
    payload.primitive_id = -1;
}
