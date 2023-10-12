#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#define USE_PUSH_CONSTANTS
#define ENVIRONMENT_MAP_BINDING 7
#define SCENE_METADATA_BINDING 11
#include "whitted.glsl"

layout(location = 0) rayPayloadInEXT hit_payload payload;

void main()
{
    vec4 color = scene_metadata.environment_factor;
    if(scene_metadata.environment_proj >= 0)
    {
        vec3 view = gl_WorldRayDirectionEXT;
        vec2 uv = vec2(0);
        uv.y = asin(-view.y)/3.141592+0.5f;
        uv.x = atan(view.z, view.x)/(2*3.141592)+0.5f;
        color.rgb = texture(environment_map_tex, uv).rgb;
    }
    payload.color = color;
}
