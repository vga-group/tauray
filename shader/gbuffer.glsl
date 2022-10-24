#ifndef GBUFFER_GLSL
#define GBUFFER_GLSL
#include "material.glsl"

//==============================================================================
// Color
//==============================================================================
#ifdef COLOR_TARGET_BINDING
layout(binding = COLOR_TARGET_BINDING, set = 0, rgba32f) uniform image2DArray color_target;

void write_gbuffer_color(vec4 color, ivec3 pos)
{
    imageStore(color_target, pos, color);
}

void accumulate_gbuffer_color(
    vec4 color, ivec3 pos, uint samples, uint previous_samples
){
    if(previous_samples != 0)
    {
        vec4 prev_color = imageLoad(color_target, pos);
        uint total = samples + previous_samples;
        color = mix(color, prev_color, float(previous_samples)/float(total));
    }
    imageStore(color_target, pos, color);
}

vec4 read_gbuffer_color(ivec3 pos) { return imageLoad(color_target, pos); }

#elif defined(COLOR_TARGET_LOCATION)

layout(location = COLOR_TARGET_LOCATION) out vec4 color_target;

void write_gbuffer_color(vec4 color)
{
    color_target = color;
}

#else

void write_gbuffer_color(vec4 color, ivec3 pos) {}
void write_gbuffer_color(vec4 color) {}
void accumulate_gbuffer_color(
    vec4 color, ivec3 pos, uint samples, uint previous_samples
){}
vec4 read_gbuffer_color(ivec3 pos) { return vec4(0); }

#endif

//==============================================================================
// Direct lighting
//==============================================================================
#ifdef DIRECT_TARGET_BINDING
layout(binding = DIRECT_TARGET_BINDING, set = 0, rgba32f) uniform image2DArray direct_target;

void write_gbuffer_direct(vec4 light, ivec3 pos)
{
    imageStore(direct_target, pos, light);
}

void accumulate_gbuffer_direct(
    vec4 light, ivec3 pos, uint samples, uint previous_samples
){
    if(previous_samples != 0)
    {
        vec4 prev_light = imageLoad(direct_target, pos);
        uint total = samples + previous_samples;
        light = mix(light, prev_light, float(previous_samples)/float(total));
    }
    imageStore(direct_target, pos, light);
}

vec4 read_gbuffer_direct(ivec3 pos) { return imageLoad(direct_target, pos); }

#elif defined(DIRECT_TARGET_LOCATION)

layout(location = DIRECT_TARGET_LOCATION) out vec4 direct_target;

void write_gbuffer_direct(vec4 color)
{
    direct_target = color;
}

#else

void write_gbuffer_direct(vec4 color, ivec3 pos) {}
void write_gbuffer_direct(vec4 color) {}
void accumulate_gbuffer_direct(
    vec4 direct, ivec3 pos, uint samples, uint previous_samples
) {}
vec4 read_gbuffer_direct(ivec3 pos) { return vec4(0); }

#endif

//==============================================================================
// Diffuse lighting
//==============================================================================
#ifdef DIFFUSE_TARGET_BINDING
layout(binding = DIFFUSE_TARGET_BINDING, set = 0, rgba32f) uniform image2DArray diffuse_target;

void write_gbuffer_diffuse(vec4 light, ivec3 pos)
{
    imageStore(diffuse_target, pos, light);
}

void accumulate_gbuffer_diffuse(
    vec4 light, ivec3 pos, uint samples, uint previous_samples
){
    if(previous_samples != 0)
    {
        vec4 prev_light = imageLoad(diffuse_target, pos);
        uint total = samples + previous_samples;
        light = mix(light, prev_light, float(previous_samples)/float(total));
    }
    imageStore(diffuse_target, pos, light);
}

vec4 read_gbuffer_diffuse(ivec3 pos) { return imageLoad(diffuse_target, pos); }

#elif defined(DIFFUSE_TARGET_LOCATION)

layout(location = DIFFUSE_TARGET_LOCATION) out vec4 diffuse_target;

void write_gbuffer_diffuse(vec4 color)
{
    diffuse_target = color;
}

#else

void write_gbuffer_diffuse(vec4 color, ivec3 pos) {}
void write_gbuffer_diffuse(vec4 color) {}
void accumulate_gbuffer_diffuse(
    vec4 direct, ivec3 pos, uint samples, uint previous_samples
) {}
vec4 read_gbuffer_diffuse(ivec3 pos) { return vec4(0); }

#endif

//==============================================================================
// Albedo
//==============================================================================
#ifdef ALBEDO_TARGET_BINDING
layout(binding = ALBEDO_TARGET_BINDING, set = 0, rgba16) uniform image2DArray albedo_target;

void write_gbuffer_albedo(vec4 albedo, ivec3 pos)
{
    imageStore(albedo_target, pos, albedo);
}

vec4 read_gbuffer_albedo(ivec3 pos) { return imageLoad(albedo_target, pos); }

#elif defined(ALBEDO_TARGET_LOCATION)

layout(location = ALBEDO_TARGET_LOCATION) out vec4 albedo_target;

void write_gbuffer_albedo(vec4 albedo)
{
    albedo_target = albedo;
}

#else

void write_gbuffer_albedo(vec4 color, ivec3 pos) {}
void write_gbuffer_albedo(vec4 color) {}
vec4 read_gbuffer_albedo(ivec3 pos) { return vec4(0); }

#endif

//==============================================================================
// Material
//==============================================================================

vec2 pack_gbuffer_material(sampled_material mat)
{
    return vec2(mat.metallic, mat.roughness);
}

#ifdef MATERIAL_TARGET_BINDING
layout(binding = MATERIAL_TARGET_BINDING, set = 0, rg16) uniform image2DArray material_target;

void write_gbuffer_material(vec2 metallic_roughness, ivec3 pos)
{
    imageStore(material_target, pos, vec4(metallic_roughness, 0, 0));
}

vec2 read_gbuffer_material(ivec3 pos)
{
    return imageLoad(material_target, pos).xy;
}

#elif defined(MATERIAL_TARGET_LOCATION)

layout(location = MATERIAL_TARGET_LOCATION) out vec2 material_target;

void write_gbuffer_material(vec2 metallic_roughness)
{
    material_target = metallic_roughness;
}

#else

void write_gbuffer_material(vec2 metallic_roughness, ivec3 pos) {}
void write_gbuffer_material(vec2 metallic_roughness) {}
vec2 read_gbuffer_material(ivec3 pos) { return vec2(0); }

#endif

//==============================================================================
// Normal
//==============================================================================

vec2 pack_gbuffer_normal(vec3 normal)
{
    normal /= abs(normal.x) + abs(normal.y) + abs(normal.z);
    return normal.z >= 0.0 ?
        normal.xy : (1 - abs(normal.yx)) * (step(vec2(0), normal.xy)*2-1);
}

vec3 unpack_gbuffer_normal(vec2 packed_normal)
{
    vec3 normal = vec3(
        packed_normal.x,
        packed_normal.y,
        1 - abs(packed_normal.x) - abs(packed_normal.y)
    );
    normal.xy += clamp(normal.z, -1.0f, 0.0f) * (step(vec2(0), normal.xy) * 2 - 1);
    return normalize(normal);
}
#ifdef NORMAL_TARGET_BINDING
layout(binding = NORMAL_TARGET_BINDING, set = 0, rg16_snorm) uniform image2DArray normal_target;

void write_gbuffer_normal(vec3 normal, ivec3 pos)
{
    imageStore(
        normal_target, pos,
        vec4(pack_gbuffer_normal(normal), 0, 0)
    );
}

vec3 read_gbuffer_normal(ivec3 pos)
{
    return unpack_gbuffer_normal(imageLoad(normal_target, pos).xy);
}

#elif defined(NORMAL_TARGET_LOCATION)

layout(location = NORMAL_TARGET_LOCATION) out vec2 normal_target;

void write_gbuffer_normal(vec3 normal)
{
    normal_target = pack_gbuffer_normal(normal);
}

#else

void write_gbuffer_normal(vec3 normal, ivec3 pos) {}
void write_gbuffer_normal(vec3 normal) {}
vec3 read_gbuffer_normal(ivec3 pos) { return vec3(0,0,1); }

#endif

//==============================================================================
// Position
//==============================================================================

#ifdef POS_TARGET_BINDING
layout(binding = POS_TARGET_BINDING, set = 0, rgba32f) uniform image2DArray pos_target;

void write_gbuffer_pos(vec3 view_pos, ivec3 pos)
{
    imageStore(pos_target, pos, vec4(view_pos, 0));
}

vec3 read_gbuffer_pos(ivec3 pos)
{
    return imageLoad(pos_target, pos).xyz;
}

#elif defined(POS_TARGET_LOCATION)

layout(location = POS_TARGET_LOCATION) out vec3 pos_target;

void write_gbuffer_pos(vec3 view_pos)
{
    pos_target = view_pos;
}

#else

void write_gbuffer_pos(vec3 view_pos, ivec3 pos) {}
void write_gbuffer_pos(vec3 view_pos) {}
vec3 read_gbuffer_pos(ivec3 pos) { return vec3(0); }

#endif

//==============================================================================
// Screen-space motion
//==============================================================================

#ifdef SCREEN_MOTION_TARGET_BINDING
layout(binding = SCREEN_MOTION_TARGET_BINDING, set = 0, rg32f) uniform image2DArray screen_motion_target;

void write_gbuffer_screen_motion(vec3 prev_frag_uv, ivec3 pos)
{
    imageStore(screen_motion_target, pos, vec4(prev_frag_uv, 0));
}

vec2 read_gbuffer_screen_motion(ivec3 pos)
{
    return imageLoad(screen_motion_target, pos).xy;
}

#elif defined(SCREEN_MOTION_TARGET_LOCATION)

layout(location = SCREEN_MOTION_TARGET_LOCATION) out vec4 screen_motion_target;

void write_gbuffer_screen_motion(vec3 prev_frag_uv)
{
    screen_motion_target = vec4(prev_frag_uv, 0.0);
}

#else

void write_gbuffer_screen_motion(vec2 prev_frag_uv, ivec3 pos) {}
void write_gbuffer_screen_motion(vec2 prev_frag_uv) {}
vec2 read_gbuffer_screen_motion(ivec3 pos) { return vec2(0); }

#endif

//==============================================================================
// Instance ID
//==============================================================================

#ifdef INSTANCE_ID_TARGET_BINDING
layout(binding = INSTANCE_ID_TARGET_BINDING, set = 0, r32i) uniform iimage2DArray instance_id_target;

void write_gbuffer_instance_id(int id, ivec3 pos)
{
    imageStore(instance_id_target, pos, ivec4(id, 0, 0, 0));
}

int read_gbuffer_instance_id(ivec3 pos)
{
    return imageLoad(instance_id_target, pos).x;
}

#elif defined(INSTANCE_ID_TARGET_LOCATION)

layout(location = INSTANCE_ID_TARGET_LOCATION) out int instance_id_target;

void write_gbuffer_instance_id(int id)
{
    instance_id_target = id;
}

#else

void write_gbuffer_instance_id(int id, ivec3 pos) {}
void write_gbuffer_instance_id(int id) {}
int read_gbuffer_instance_id(ivec3 pos) { return 0; }

#endif

//==============================================================================
// Linear depth
//==============================================================================

#ifdef LINEAR_DEPTH_TARGET_BINDING
layout(binding = LINEAR_DEPTH_TARGET_BINDING, set = 0, rgba32f) uniform image2DArray linear_depth_target;

void write_gbuffer_linear_depth(ivec3 pos)
{
    // TODO: Write linear depth from the path tracer as well
    imageStore(linear_depth_target, pos, vec4(0.0, 0.0, 0.0, 1.0));
}

vec3 read_gbuffer_linear_depth(ivec3 pos)
{
    return imageLoad(linear_depth_target, pos).rgb;
}

#elif defined(LINEAR_DEPTH_TARGET_LOCATION)

layout(location = LINEAR_DEPTH_TARGET_LOCATION) out vec4 linear_depth_target;

void write_gbuffer_linear_depth()
{
    const float linear_depth = gl_FragCoord.z / gl_FragCoord.w;
    const vec2 grad = vec2(dFdx(linear_depth), dFdy(linear_depth));
    linear_depth_target = vec4(linear_depth, grad, 0.0);
}

#else

void write_gbuffer_linear_depth(ivec3 pos) {}
void write_gbuffer_linear_depth() {}
vec3 read_gbuffer_linear_depth(ivec3 pos) { return vec3(0.0); }

#endif

#endif

