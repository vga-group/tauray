#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

hitAttributeEXT vec2 attribs;

#define CALC_PREV_VERTEX_POS
#include "rt_feature.glsl"

layout(location = 0) rayPayloadInEXT hit_payload payload;

void main()
{
    vec3 view = gl_WorldRayDirectionEXT;
    int instance_id = gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT;
    vertex_data v = get_interpolated_vertex(view, attribs, instance_id, gl_PrimitiveID);

    sampled_material mat = sample_material(instance_id, v);
    const camera_data cam = get_camera();
    const camera_data prev_cam = get_prev_camera();

    payload.data = FEATURE;
}

