#ifndef FORWARD_GLSL
#define FORWARD_GLSL
#define USE_SHADOW_MAPPING
#define SCENE_SET 0
#define SCENE_RASTER_SET 1
#include "scene_raster.glsl"

layout(push_constant) uniform push_constant_buffer
{
    ivec2 size;
    uint instance_id;
    int base_camera_index;
    int frame_index;
    int flipped_winding_order;
    int has_prev_pos_data;
    int pad[1];
    shadow_mapping_parameters sm_params;
    vec3 ambient_color;
} control;

#endif
