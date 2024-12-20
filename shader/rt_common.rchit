#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

hitAttributeEXT vec2 attribs;

#define PAYLOAD_IN
#include "rt_common_payload.glsl"

void main()
{
    payload.instance_id = gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT;
    payload.primitive_id = gl_PrimitiveID;
    payload.barycentrics = attribs;
}
