#version 450
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_multiview : enable

#define CAMERA_DATA_BINDING 4
#define CALC_PREV_VERTEX_POS
#include "forward.glsl"

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec4 in_tangent;

layout(location = 0) out vec3 out_pos;
layout(location = 1) out vec3 out_prev_pos;
layout(location = 2) out vec3 out_normal;
layout(location = 3) out vec2 out_uv;
layout(location = 4) out vec3 out_tangent;
layout(location = 5) out vec3 out_bitangent;

void main()
{
    instance o = instances.o[control.instance_id];
    out_pos = vec3(o.model * vec4(in_pos, 1.0f));
    out_prev_pos = vec3(o.model_prev * vec4(in_pos, 1.0f));
    gl_Position = camera.pairs[gl_ViewIndex].current.view_proj * vec4(out_pos, 1.0f);
    out_normal = normalize(mat3(o.model_normal) * in_normal);
    out_tangent = normalize(mat3(o.model_normal) * in_tangent.xyz);
    out_bitangent = (cross(out_normal, out_tangent) * in_tangent.w);

    out_uv = in_uv;
}
