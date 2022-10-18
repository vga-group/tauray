#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_multiview : enable

#define ENVIRONMENT_MAP_BINDING 0
#define CAMERA_DATA_BINDING 1
#include "scene.glsl"

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform push_constant_buffer
{
    vec4 environment_factor;
    vec2 screen_size;
    int environment_proj;
} control;

void main()
{
    vec3 origin;
    vec3 view_dir;
    get_camera_ray(
        camera.pairs[gl_ViewIndex].current,
        vec2(
            gl_FragCoord.x,
            control.screen_size.y - gl_FragCoord.y
        ),
        control.screen_size,
        vec2(0),
        origin,
        view_dir
    );

    out_color = control.environment_factor;

    if(control.environment_proj >= 0)
    {
        vec3 view = normalize(view_dir);
        vec2 uv = vec2(0);
        uv.y = asin(-view.y)/3.141592+0.5f;
        uv.x = atan(view.z, view.x)/(2*3.141592)+0.5f;
        out_color *= vec4(texture(environment_map_tex, uv).rgb, 1.0f);
    }
}
