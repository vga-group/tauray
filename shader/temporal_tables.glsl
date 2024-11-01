#ifndef TEMPORAL_TABLES_GLSL
#define TEMPORAL_TABLES_GLSL
#include "scene.glsl"

#ifndef TEMPORAL_TABLE_SET
#define TEMPORAL_TABLE_SET 3
#endif

layout(binding = 0, set = TEMPORAL_TABLE_SET) readonly buffer instance_forward_map_buffer
{
    uint array[];
} instance_forward_map;
layout(binding = 1, set = TEMPORAL_TABLE_SET) readonly buffer instance_backward_map_buffer
{
    uint array[];
} instance_backward_map;
layout(binding = 2, set = TEMPORAL_TABLE_SET) readonly buffer point_light_forward_map_buffer
{
    uint array[];
} point_light_forward_map;
layout(binding = 3, set = TEMPORAL_TABLE_SET) readonly buffer point_light_backward_map_buffer
{
    uint array[];
} point_light_backward_map;

layout(binding = 4, set = TEMPORAL_TABLE_SET, scalar) buffer prev_point_light_buffer
{
    point_light lights[];
} prev_point_lights;

#ifdef RAY_TRACING
layout(binding = 5, set = TEMPORAL_TABLE_SET) uniform accelerationStructureEXT prev_tlas;
#endif

#endif
