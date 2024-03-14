#version 450
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_multiview : enable

layout(location = 0) in vec3 in_pos;

#include "scene.glsl"

layout(push_constant) uniform push_constant_buffer
{
    uint instance_id;
    int base_camera_index;
} control;

void main()
{
    instance o = instances.o[control.instance_id];
    vec3 pos = vec3(o.model * vec4(in_pos, 1.0f));
    gl_Position = camera.pairs[control.base_camera_index+gl_ViewIndex].current.view_proj * vec4(pos, 1.0f);
}

