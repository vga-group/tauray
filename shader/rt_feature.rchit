#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

hitAttributeEXT vec2 attribs;

#define TLAS_BINDING 0
#define SCENE_DATA_BUFFER_BINDING 1
#define VERTEX_BUFFER_BINDING 2
#define INDEX_BUFFER_BINDING 3
#define TEXTURE_ARRAY_BINDING 4
#define DISTRIBUTION_DATA_BINDING 9
#define CAMERA_DATA_BINDING 10
#define CALC_PREV_VERTEX_POS
#include "rt_feature.glsl"

layout(location = 0) rayPayloadInEXT hit_payload payload;

void main()
{
    vec3 view = gl_WorldRayDirectionEXT;
    vertex_data v = get_interpolated_vertex(view, attribs, gl_InstanceID, gl_PrimitiveID);

    sampled_material mat = sample_material(gl_InstanceID, v);
    const camera_data cam = get_camera();
    const camera_data prev_cam = get_prev_camera();

    payload.data = FEATURE;
}

