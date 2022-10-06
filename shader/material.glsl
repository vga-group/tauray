#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL

struct material
{
    vec4 albedo_factor;
    vec4 metallic_roughness_factor;
    vec4 emission_factor_double_sided;
    float transmittance;
    float ior;
    float normal_factor;
    float pad;
    int albedo_tex_id;
    int metallic_roughness_tex_id;
    int normal_tex_id;
    int emission_tex_id;
};

struct sampled_material
{
    vec4 albedo;
    float metallic;
    float roughness;
    vec3 emission;
    float transmittance;
    float ior_in;
    float ior_out;
    float f0;
    vec3 f0_m;
    bool double_sided;
    float shadow_terminator_mul;
};

#endif
