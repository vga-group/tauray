#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

hitAttributeEXT vec2 attribs;

#include "restir_di.glsl"

layout(location = 0) rayPayloadInEXT hit_payload payload;

void main()
{
    payload.instance_id = gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT;
    payload.primitive_id = gl_PrimitiveID;
    payload.barycentrics = attribs;
}
