#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

hitAttributeEXT vec2 attribs;

#include "path_tracer.glsl"

layout(location = 0) rayPayloadInEXT hit_payload payload;

void main()
{
    payload.instance_id = -1;
    payload.primitive_id = gl_PrimitiveID;
    payload.barycentrics.x = gl_HitTEXT;
}
