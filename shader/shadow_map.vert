#version 450
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

#include "shadow_map_common.glsl"

layout(location = 0) in vec3 in_pos;
layout(location = 2) in vec2 in_uv;
layout(location = 0) out vec2 out_uv;

void main()
{
    instance o = instances.o[control.instance_id];
    vec3 pos = vec3(o.model * vec4(in_pos, 1.0f));
    gl_Position = camera.view_proj[control.camera_index] * vec4(pos, 1.0f);
    out_uv = in_uv;
}

