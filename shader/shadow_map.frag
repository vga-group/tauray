#version 450
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

#include "shadow_map_common.glsl"

layout(location = 0) in vec2 in_uv;

void main()
{
    instance o = instances.o[control.instance_id];
    if(control.alpha_clip < 1.0f)
    {
        float alpha = o.mat.albedo_factor.a;
        if(o.mat.albedo_tex_id >= 0)
            alpha *= texture(textures[nonuniformEXT(o.mat.albedo_tex_id)], in_uv).a;
        if(alpha < control.alpha_clip)
            discard;
    }
}
