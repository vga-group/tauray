#version 450
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_multiview : enable

layout(location = 0) in vec3 in_pos;

#define SCENE_SET 1
#include "scene.glsl"

layout(push_constant) uniform push_constant_buffer
{
    vec4 offset;
    vec4 scale;
    uint instance_id;
} control;

void main()
{
    instance o = instances.o[control.instance_id];
    vec3 pos = vec3(o.model * vec4(in_pos, 1.0f));
    vec3 projected_coord = pos * control.scale.xyz + control.offset.xyz;
    gl_Position = vec4(projected_coord, 1.0);

}
