#ifndef GBUFFER_GLSL
#define GBUFFER_GLSL
#include "material.glsl"
#include "math.glsl"
#extension GL_EXT_debug_printf : enable

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

vec4 sample_gbuffer_color(sampler2D tex, ivec2 p)
{
    return texelFetch(tex, p, 0);
}

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
    vec4 diffuse, ivec3 pos, uint samples, uint previous_samples
) {}
vec4 read_gbuffer_diffuse(ivec3 pos) { return vec4(0); }

#endif

vec4 sample_gbuffer_diffuse(sampler2D tex, ivec2 p)
{
    return texelFetch(tex, p, 0);
}

//==============================================================================
// Reflection
//==============================================================================
#ifdef REFLECTION_TARGET_BINDING
layout(binding = REFLECTION_TARGET_BINDING, set = 0, rgba32f) uniform image2DArray reflection_target;

void write_gbuffer_reflection(vec4 light, ivec3 pos)
{
    imageStore(reflection_target, pos, light);
}

void accumulate_gbuffer_reflection(
    vec4 light, ivec3 pos, uint samples, uint previous_samples
){
    if(previous_samples != 0)
    {
        vec4 prev_light = imageLoad(reflection_target, pos);
        uint total = samples + previous_samples;
        light = mix(light, prev_light, float(previous_samples)/float(total));
    }
    imageStore(reflection_target, pos, light);
}

vec4 read_gbuffer_reflection(ivec3 pos) { return imageLoad(reflection_target, pos); }

#elif defined(REFLECTION_TARGET_LOCATION)

layout(location = REFLECTION_TARGET_LOCATION) out vec4 reflection_target;

void write_gbuffer_reflection(vec4 color)
{
    reflection_target = color;
}

#else

void write_gbuffer_reflection(vec4 color, ivec3 pos) {}
void write_gbuffer_reflection(vec4 color) {}
void accumulate_gbuffer_reflection(
    vec4 reflection, ivec3 pos, uint samples, uint previous_samples
) {}
vec4 read_gbuffer_reflection(ivec3 pos) { return vec4(0); }

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

vec4 sample_gbuffer_albedo(sampler2D tex, ivec2 p)
{
    return texelFetch(tex, p, 0);
}

//==============================================================================
// Emission
//==============================================================================

#ifdef EMISSION_TARGET_BINDING
layout(binding = EMISSION_TARGET_BINDING, set = 0, rgba32f) uniform image2DArray emission_target;

void write_gbuffer_emission(vec3 emission, ivec3 pos) { imageStore(emission_target, pos, vec4(emission, 0)); }
vec3 read_gbuffer_emission(ivec3 pos) { return imageLoad(emission_target, pos).xyz; }

#elif defined(EMISSION_TARGET_LOCATION)

layout(location = EMISSION_TARGET_LOCATION) out vec4 emission_target;

void write_gbuffer_emission(vec3 emission)
{
    emission_target = vec4(emission, 0.0f);
}

#else

void write_gbuffer_emission(vec3 emission, ivec3 pos) {}
void write_gbuffer_emission(vec3 emission) {}
vec3 read_gbuffer_emission(ivec3 pos) { return vec3(0); }

#endif

vec3 sample_gbuffer_emission(sampler2D tex, ivec2 p)
{
    return texelFetch(tex, p, 0).rgb;
}

//==============================================================================
// Curvature
//==============================================================================

#ifdef CURVATURE_TARGET_BINDING
layout(binding = CURVATURE_TARGET_BINDING, set = 0, r32f) uniform image2DArray curvature_target;

void write_gbuffer_curvature(float curvature, ivec3 pos) { imageStore(curvature_target, pos, vec4(curvature)); }
float read_gbuffer_curvature(ivec3 pos) { return imageLoad(curvature_target, pos).x; }

#elif defined(CURVATURE_TARGET_LOCATION)

layout(location = CURVATURE_TARGET_LOCATION) out float curvature_target;

void write_gbuffer_curvature(float curvature)
{
    curvature_target = curvature;
}

#else

void write_gbuffer_curvature(float curvature, ivec3 pos) {}
void write_gbuffer_curvature(float curvature) {}
float read_gbuffer_curvature(ivec3 pos) { return 0; }

#endif

float sample_gbuffer_curvature(sampler2D tex, ivec2 p)
{
    return texelFetch(tex, p, 0).r;
}

//==============================================================================
// Material
//==============================================================================

vec4 pack_gbuffer_material(sampled_material mat)
{
    float ior = mat.ior_out / mat.ior_in;
    return vec4(mat.metallic, mat.roughness, ior*0.25f, mat.transmittance);
}

sampled_material unpack_gbuffer_material(vec4 packed_mat, vec4 albedo, vec3 emission)
{
    sampled_material ret;
    ret.albedo = albedo;
    ret.metallic = packed_mat[0];
    ret.roughness = packed_mat[1];
    ret.emission = emission;
    ret.transmittance = packed_mat[3];
    ret.ior_in = 1.0f;
    ret.ior_out = packed_mat[2] * 4.0f;
    ret.f0 = (ret.ior_out - ret.ior_in)/(ret.ior_out + ret.ior_in);
    ret.f0 *= ret.f0;
    ret.flags = MATERIAL_FLAG_DOUBLE_SIDED;
    ret.shadow_terminator_mul = 0.0f;

    return ret;
}

#ifdef MATERIAL_TARGET_BINDING
layout(binding = MATERIAL_TARGET_BINDING, set = 0, rgba8) uniform image2DArray material_target;

void write_gbuffer_material(sampled_material mat, ivec3 pos)
{
    imageStore(material_target, pos, pack_gbuffer_material(mat));
}

vec4 read_gbuffer_material(ivec3 pos)
{
    return imageLoad(material_target, pos);
}

#elif defined(MATERIAL_TARGET_LOCATION)

layout(location = MATERIAL_TARGET_LOCATION) out vec4 material_target;

void write_gbuffer_material(sampled_material mat)
{
    material_target = pack_gbuffer_material(mat);
}

#else

void write_gbuffer_material(sampled_material mat, ivec3 pos) {}
void write_gbuffer_material(sampled_material packed_mat) {}
vec4 read_gbuffer_material(ivec3 pos) { return vec4(0); }

#endif

sampled_material sample_gbuffer_material(
    sampler2D albedo,
    sampler2D material,
    sampler2D emission,
    ivec2 p
){
    return unpack_gbuffer_material(
        texelFetch(material, p, 0),
        sample_gbuffer_albedo(albedo, p),
        sample_gbuffer_emission(emission, p)
    );
}

sampled_material sample_gbuffer_material(
    sampler2D albedo,
    sampler2D material,
    ivec2 p
){
    return unpack_gbuffer_material(
        texelFetch(material, p, 0),
        sample_gbuffer_albedo(albedo, p),
        vec3(0)
    );
}

//==============================================================================
// Normals
//==============================================================================

vec2 pack_gbuffer_normal(vec3 normal)
{
    return octahedral_pack(normal);
}

vec3 unpack_gbuffer_normal(vec2 packed_normal)
{
    return octahedral_unpack(packed_normal);
}

//==============================================================================
// Mapped normal
//==============================================================================

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

vec3 sample_gbuffer_normal(sampler2D tex, ivec2 p)
{
    return unpack_gbuffer_normal(texelFetch(tex, p, 0).rg);
}

//==============================================================================
// Flat normal
//==============================================================================

#ifdef FLAT_NORMAL_TARGET_BINDING
layout(binding = FLAT_NORMAL_TARGET_BINDING, set = 0, rg16_snorm) uniform image2DArray flat_normal_target;

void write_gbuffer_flat_normal(vec3 normal, ivec3 pos)
{
    imageStore(
        normal_target, pos,
        vec4(pack_gbuffer_normal(normal), 0, 0)
    );
}

vec3 read_gbuffer_flat_normal(ivec3 pos)
{
    return unpack_gbuffer_normal(imageLoad(flat_normal_target, pos).xy);
}

#elif defined(FLAT_NORMAL_TARGET_LOCATION)

layout(location = FLAT_NORMAL_TARGET_LOCATION) out vec2 flat_normal_target;

void write_gbuffer_flat_normal(vec3 normal)
{
    flat_normal_target = pack_gbuffer_normal(normal);
}

#else

void write_gbuffer_flat_normal(vec3 normal, ivec3 pos) {}
void write_gbuffer_flat_normal(vec3 normal) {}
vec3 read_gbuffer_flat_normal(ivec3 pos) { return vec3(0,0,1); }

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

vec3 sample_gbuffer_position(sampler2D tex, ivec2 p)
{
    return texelFetch(tex, p, 0).rgb;
}

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

void write_gbuffer_screen_motion(vec3 prev_frag_uv, ivec3 pos) {}
void write_gbuffer_screen_motion(vec3 prev_frag_uv) {}
vec2 read_gbuffer_screen_motion(ivec3 pos) { return vec2(0); }

#endif

vec2 sample_gbuffer_screen_motion(sampler2D tex, ivec2 p)
{
    return texelFetch(tex, p, 0).rg;
}

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
    //const float linear_depth = gl_FragCoord.z / gl_FragCoord.w;
    const float linear_depth = 1.0 / gl_FragCoord.w; // Viewspace Z coordinate
    const vec2 grad = vec2(dFdx(linear_depth), dFdy(linear_depth));
    linear_depth_target = vec4(linear_depth, grad, 0.0);
}

#else

void write_gbuffer_linear_depth(ivec3 pos) {}
void write_gbuffer_linear_depth() {}
vec3 read_gbuffer_linear_depth(ivec3 pos) { return vec3(0.0); }

#endif

//==============================================================================
// PROB
//==============================================================================

#ifdef PROB_TARGET_BINDING
layout(binding = PROB_TARGET_BINDING, set = 0, rgba32f) uniform image2DArray prob_target;

void write_gbuffer_prob(vec4 prob, ivec3 pos)
{
    imageStore(prob_target, pos, vec4(prob.rgb, 0));
}

vec3 read_gbuffer_prob(ivec3 pos)
{
    return imageLoad(prob_target, pos).xyz;
}

#elif defined(PROB_TARGET_LOCATION)

layout(location = PROB_TARGET_LOCATION) out vec3 prob_target;

void write_gbuffer_prob(vec4 prob)
{
    prob_target = prob;
}

#else

void write_gbuffer_prob(vec4 prob, ivec3 pos) {
    if(pos.xy == ivec2(640, 360))
        debugPrintfEXT("jei");
}

void write_gbuffer_prob(vec4 prob) {}
vec3 read_gbuffer_prob(ivec3 pos) { return vec3(0); }

#endif
#endif

