#ifndef GGX_GLSL
#define GGX_GLSL
#include "material.glsl"

// All terms except the Fresnel term are directly from the GGX paper:
// http://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
// Performance gains using other approximations could be done, but the initial
// focus is on correctness.
//
// Note that all of the functions here operate in tangent space, where the
// normal is always equal to vec3(0,0,1). Use create_tangent_space() to
// transform view and out directions before calling these functions. This
// approach simplifies several functions and is common in path tracing stuff.
//
// The results from most material functions are split in two: the surface
// reflectance and diffuse + transmission. These are called 'specular' and
// 'diffuse' respectively, despite not really matching 1:1 with those concepts.
// Basically, we want to separate the sides based on what Fresnel equations
// consider as different sides, and there, diffuse clearly lies on the other
// side than specular reflections.
//
// To the person in the future optimizing this: Please make sure the BTDF works
// correctly as well. Lots of optimized formulae on the internet only work with
// the BRDF.
//
// Optimization tip: precalculate ETA, refactor transmission equations to use
// ETA alone instead of both ior_in and ior_out.

// Also known as F
float ggx_fresnel_schlick(float cos_d, float f0)
{
    return f0 + (1.0f - f0) * pow(max(1.0f - cos_d, 0.0f), 5.0f);
}

// Same as above, but works deals with refractions and metals properly.
float ggx_fresnel(float cos_d, sampled_material mat)
{
    if(mat.ior_in > mat.ior_out)
    {
        float inv_eta = mat.ior_in / mat.ior_out;
        float sin_theta2 = inv_eta * inv_eta * (1.0f - cos_d * cos_d);
        if(sin_theta2 >= 1.0f)
            return 1.0f;
        cos_d = sqrt(1.0f - sin_theta2);
    }
    else if(mat.ior_in == mat.ior_out)
        return 0.0f;
    return ggx_fresnel_schlick(cos_d, mat.f0);
}

// Used in importance sampling which sometimes can't depend on actual fresnel
// values. This one works with the cos_d being dot(view, surface normal)
// (not microfacet normal).
float fresnel_importance(float cos_d, sampled_material mat)
{
    if(mat.ior_in > mat.ior_out)
    {
        float inv_eta = mat.ior_in / mat.ior_out;
        float sin_theta2 = inv_eta * inv_eta * (1.0f - cos_d * cos_d);
        if(sin_theta2 >= 1.0f)
            return 1.0f;
        cos_d = sqrt(1.0f - sin_theta2);
    }
    else if(mat.ior_in == mat.ior_out)
        return 0.0f;
    return mat.f0 + (max(1.0f - mat.roughness, mat.f0) - mat.f0) * pow(1.0f - cos_d, 5.0f);
}

float fresnel_schlick_attenuated(float cos_d, float f0, float roughness)
{
    return f0 + (max(1.0f - roughness, f0) - f0) * pow(1.0f - cos_d, 5.0f);
}

// Only valid when refraction isn't possible. Faster than the generic
// ggx_fresnel. Specular-only, has no diffuse component!
float ggx_fresnel_refl(float cos_d, sampled_material mat)
{
    return ggx_fresnel_schlick(cos_d, mat.f0);
}

// Also known as G1
float ggx_masking(float v_dot_n, float v_dot_h, float a)
{
    float a2 = a*a;
    return step(0.0f, v_dot_n * v_dot_h) * 2.0f /
        (1.0f + sqrt(1.0f + a2 / (v_dot_n * v_dot_n) - a2));
}

// Also known as G
float ggx_masking_shadowing(
    float v_dot_n, float v_dot_h, float l_dot_n, float l_dot_h, float a
){
    float a2 = a*a;
    return step(0.0f, v_dot_n * v_dot_h) * step(0.0f, l_dot_n * l_dot_h) * 4.0f /
        ((1.0f + sqrt(1.0f + a2 / max(v_dot_n * v_dot_n, 1e-18) - a2)) *
         (1.0f + sqrt(1.0f + a2 / max(l_dot_n * l_dot_n, 1e-18) - a2)));
}

// The above function, but pre-divided by 4.0 * cos_l * cos_v. In addition to
// optimization, this removes some singularities.
float ggx_masking_shadowing_predivided(
    float v_dot_n, float v_dot_h, float l_dot_n, float l_dot_h, float a
){
    float a2 = a*a;
    float denom1 = abs(l_dot_n) * sqrt(a2 + (1.0f - a2) * v_dot_n * v_dot_n);
    float denom2 = abs(v_dot_n) * sqrt(a2 + (1.0f - a2) * l_dot_n * l_dot_n);
    return step(0.0f, v_dot_n * v_dot_h) * step(0.0f, l_dot_n * l_dot_h) * 0.5f /
        (denom1 + denom2);
}

// Also known as D. Looking for the source of fireflies with point lights? It's
// this. And there's nothing you can do about it without making the renderer
// very biased. Maybe some post-processing filter could help?
float ggx_distribution(float h_dot_n, float a)
{
    float a2 = a * a;
    float denom = h_dot_n * h_dot_n * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI * denom * denom);
}

// This separation to the inner and outer parts only exists for reuse in path
// tracing code.
void ggx_brdf_inner(
    vec3 out_dir,
    vec3 view_dir,
    vec3 h,
    float fresnel,
    float distribution,
    float cos_d,
    sampled_material mat,
    inout bsdf_lobes bsdf
){
    float cos_l = out_dir.z; // dot(normal, out_dir)
    float cos_v = view_dir.z; // dot(normal, view_dir)

    float geometry = ggx_masking_shadowing_predivided(
        cos_v, cos_d, cos_l, dot(out_dir, h), mat.roughness);

    // This is not strictly part of the GGX brdf. It's an addition to use the
    // non-transmissive part that isn't reflected for diffuse lighting.
    float kd = (1.0f - fresnel) * (1.0f - mat.metallic) * (1.0f - mat.transmittance);

    cos_l = max(cos_l, 0.0f);
    bsdf.diffuse += kd * cos_l / M_PI;
    bsdf.dielectric_reflection += fresnel * geometry * distribution * cos_l * (1.0f - mat.metallic);
    bsdf.metallic_reflection += geometry * distribution * cos_l * mat.metallic;
}

void ggx_brdf(
    vec3 out_dir,
    vec3 view_dir,
    sampled_material mat,
    inout bsdf_lobes bsdf
){
    vec3 h = normalize(view_dir + out_dir);
    float cos_h = h.z; // dot(normal, h)
    float cos_d = dot(view_dir, h);

    float fresnel = ggx_fresnel_refl(cos_d, mat);
    float distribution = ggx_distribution(cos_h, mat.roughness);

    ggx_brdf_inner(out_dir, view_dir, h, fresnel, distribution, cos_d, mat, bsdf);
}

void ggx_bsdf(
    vec3 out_dir,
    vec3 view_dir,
    sampled_material mat,
    inout bsdf_lobes bsdf
){
    float cos_l = out_dir.z; // dot(normal, out_dir)
    float cos_v = view_dir.z; // dot(normal, view_dir)

    vec3 h;
    if(cos_l > 0)
        h = normalize(view_dir + out_dir);
    else
        h = (mat.ior_in > mat.ior_out ? 1 : -1) * normalize(mat.ior_out * out_dir + mat.ior_in * view_dir);

    float cos_h = h.z; // dot(normal, h)
    float cos_d = dot(view_dir, h);
    float cos_o = dot(out_dir, h);

    float fresnel = ggx_fresnel(cos_d, mat);
    float geometry = ggx_masking_shadowing_predivided(
        cos_v, cos_d, cos_l, cos_o, mat.roughness);

    const bool zero_roughness = mat.roughness < 0.001f;
    float distribution = zero_roughness ? 0 : ggx_distribution(cos_h, mat.roughness);

    if(cos_l > 0)
    { // BRDF
        float specular = fresnel * geometry * distribution;
        float kd = (1.0f - fresnel) * (1.0f - mat.metallic) * (1.0f - mat.transmittance);

        bsdf.diffuse += kd * cos_l / M_PI;
        bsdf.dielectric_reflection += fresnel * geometry * distribution * cos_l * (1.0f - mat.metallic);
        bsdf.metallic_reflection += geometry * distribution * cos_l * mat.metallic;
    }
    else
    { // BTDF
        // Un-predivide geometry term ;)
        geometry *= 4.0;
        float denom = mat.ior_in / mat.ior_out * cos_d + cos_o;
        // This should be the reciprocal form, which is necessary when the light
        // source is inside the volume...
        bsdf.transmission += -cos_l * abs(cos_d * cos_o) * mat.transmittance * (1.0f - mat.metallic) * (1.0f - fresnel) * geometry * distribution / (denom * denom);
    }
}

// Eric Heitz. A Simpler and Exact Sampling Routine for the GGX Distribution of
// Visible Normals. [Research Report] Unity Technologies. 2017. hal-01509746
// https://hal.archives-ouvertes.fr/hal-01509746/document
// Operates in the tangent space, i.e. the normal is vec3(0, 0, 1)
vec3 ggx_vndf_sample(
    vec3 view,
    float roughness,
    float u1,
    float u2
){
    vec3 v = normalize(vec3(roughness * view.x, roughness * view.y, view.z));

    vec3 t1 = v.z < 0.9999 ? normalize(cross(v, vec3(0, 0, 1))) : vec3(1, 0, 0);
    vec3 t2 = cross(t1, v);

    float inv_a = 1.0f + v.z;
    float a = 1.0f / inv_a;
    float r = sqrt(u1);
    float phi = u2 < a ? u2 * inv_a * M_PI : M_PI + (u2 - a) / (1.0f - a) * M_PI;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi) * (u2 < a ? 1.0 : v.z);
    float p3 = sqrt(max(0.0, 1.0 - p1*p1 - p2*p2));

    vec3 n = p1 * t1 + p2 * t2 + p3 * v;
    return normalize(vec3(roughness * n.x, roughness * n.y, max(0.0, n.z)));
}

// https://hal.inria.fr/hal-00996995v1/document
// http://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
void ggx_bsdf_sample(
    vec4 uniform_random,
    vec3 view_dir,
    sampled_material mat,
    out vec3 out_dir,
    inout bsdf_lobes bsdf,
    out float pdf
){
    const bool zero_roughness = mat.roughness < 0.001f;
    vec3 h = zero_roughness ? vec3(0,0,1) : ggx_vndf_sample(view_dir, mat.roughness, uniform_random.x, uniform_random.y);
    float cos_d = dot(view_dir, h);
    float fresnel = ggx_fresnel(cos_d, mat);

    float cos_v = view_dir.z;

    float max_albedo = max(mat.albedo.r, max(mat.albedo.g, mat.albedo.b));

    // This is arbitrary, but affects the PDF. Thus, handling it is part of the
    // normalization. I selected this one to have the number of rays somewhat
    // match the intensity of the reflection.
    float specular_cutoff = mix(1, fresnel_importance(view_dir.z, mat), (1-mat.metallic) * max_albedo);
    // Lower noise, but seems to bias slightly :(
    //float specular_cutoff = mix(1, max(fresnel.r, max(fresnel.g, fresnel.b)), (1-mat.metallic) * max_albedo);
    
    // If the specular test fails, next up is the decision between diffuse /
    // transmissive. Again, arbitrary number that must be accounted for in the
    // PDF.
    float diffuse_cutoff = 1.0f - mat.transmittance;

    // The specular and diffuse PDFs are non-zero only in the upper hemisphere,
    // and the transmissive is non-zero only in the lower hemisphere! That is
    // why they can be handled very separately. Since the specular and diffuse
    // PDFs are both non-zero on the upper hemisphere, they do have to be
    // blended according to probabilities.
    float specular_probability = specular_cutoff;
    float diffuse_probability = (1.0f - specular_cutoff) * diffuse_cutoff;
    float transmissive_probability =
        (1.0f - specular_cutoff) * (1.0f - diffuse_cutoff);

    float u = uniform_random.z;

    if(u <= specular_cutoff)
    { // Reflective
        out_dir = reflect(-view_dir, h);
        float cos_l = out_dir.z;
        float cos_h = h.z; // dot(normal, h)
        float G1 = ggx_masking(cos_v, cos_d, mat.roughness);
        float D = zero_roughness ? 4 * cos_l * cos_v : ggx_distribution(cos_h, mat.roughness);
        pdf = G1 * D / (4*abs(cos_v)) * specular_probability +
            (zero_roughness ? 0 : pdf_cosine_hemisphere(out_dir) * diffuse_probability);
        ggx_brdf_inner(out_dir, view_dir, h, fresnel, D, cos_d, mat, bsdf);

        // With zero roughness, the pdf really reaches infinity for the specular
        // part. Compared to that, the diffuse part basically reaches zero.
        // Since we want a finite pdf, we've essentially pre-divided both it and
        // the weights "by the same infinity" if that makes any sense ;)
        if(zero_roughness)
        {
            bsdf.diffuse = 0;
            bsdf.dielectric_reflection /= pdf;
            bsdf.metallic_reflection /= pdf;
            pdf = 0;
        }
    }
    else
    {
        // Renormalize u for reuse
        u = clamp((u - specular_cutoff)/(1 - specular_cutoff), 0.0f, 0.99999f);
        if(u <= diffuse_cutoff)
        { // Diffuse
            u = clamp(u/diffuse_cutoff, 0.0f, 0.99999f);
            out_dir = sample_cosine_hemisphere(vec2(u, uniform_random.w));
            // The half-vector is no longer related to the one from earlier;
            // this is why we recalculate a bunch of variables here.
            h = normalize(view_dir + out_dir);
            float cos_l = out_dir.z;
            float cos_h = h.z;
            cos_d = dot(view_dir, h);
            fresnel = ggx_fresnel_refl(cos_d, mat);

            float G1 = ggx_masking(cos_v, cos_d, mat.roughness);
            float D = (zero_roughness ? 0 : ggx_distribution(cos_h, mat.roughness));
            pdf = G1 * D / (4*abs(cos_v)) * specular_probability +
                pdf_cosine_hemisphere(out_dir) * diffuse_probability;
            ggx_brdf_inner(out_dir, view_dir, h, fresnel, D, cos_d, mat, bsdf);
            if(zero_roughness)
            {
                bsdf.dielectric_reflection = 0;
                bsdf.metallic_reflection = 0;
            }
        }
        else
        { // Transmissive
            out_dir = normalize(refract(-view_dir, h, mat.ior_in/mat.ior_out));
            if(any(isnan(out_dir)))
            {
                out_dir = vec3(0);
                pdf = 0;
                return;
            }
            float cos_l = out_dir.z;
            float cos_h = h.z;
            float cos_o = dot(out_dir, h);
            float G2 = ggx_masking_shadowing(cos_v, cos_d, cos_l, cos_o, mat.roughness);
            float G1 = ggx_masking(cos_v, cos_d, mat.roughness);
            float D = zero_roughness ? 4 * cos_l * cos_v : ggx_distribution(cos_h, mat.roughness);
            float denom = mat.ior_in/mat.ior_out * cos_d + cos_o;

            bsdf.transmission += abs(cos_d * cos_o) * mat.transmittance * (1.0f - mat.metallic) * (1.0f - fresnel) * G2 * D / (denom * denom * abs(cos_v));
            pdf = (abs(cos_d * cos_o) * G1 * D) / (denom * denom * abs(cos_v)) * transmissive_probability;

            if(zero_roughness)
            {
                bsdf.transmission /= pdf;
                pdf = 0;
            }
        }
    }
}

float ggx_bsdf_pdf(
    vec3 out_dir,
    vec3 view_dir,
    sampled_material mat,
    inout bsdf_lobes bsdf
){
    float cos_l = out_dir.z; // dot(normal, out_dir)
    float cos_v = view_dir.z; // dot(normal, view_dir)

    vec3 h;
    if(cos_l > 0)
        h = normalize(view_dir + out_dir);
    else
        h = (mat.ior_in > mat.ior_out ? 1 : -1) * normalize(mat.ior_out * out_dir + mat.ior_in * view_dir);

    float cos_h = h.z; // dot(normal, h)
    float cos_d = dot(view_dir, h);
    float cos_o = dot(out_dir, h);

    float fresnel = ggx_fresnel(cos_d, mat);
    float geometry = ggx_masking_shadowing_predivided(
        cos_v, cos_d, cos_l, cos_o, mat.roughness);

    const bool zero_roughness = mat.roughness < 0.001f;
    float distribution = zero_roughness ? 0 : ggx_distribution(cos_h, mat.roughness);

    float max_albedo = max(mat.albedo.r, max(mat.albedo.g, mat.albedo.b));

    float specular_cutoff = mix(1, fresnel_importance(view_dir.z, mat), (1-mat.metallic) * max_albedo);
    // Lower noise, but seems to bias slightly :(
    //float specular_cutoff = mix(1, max(fresnel.r, max(fresnel.g, fresnel.b)), (1-mat.metallic) * max_albedo);
    float diffuse_cutoff = 1.0f - mat.transmittance;

    float specular_probability = specular_cutoff;
    float diffuse_probability = (1.0f - specular_cutoff) * diffuse_cutoff;
    float transmissive_probability =
        (1.0f - specular_cutoff) * (1.0f - diffuse_cutoff);

    float G1 = ggx_masking(cos_v, cos_d, mat.roughness);

    float pdf = 0.0f;
    if(cos_l > 0)
    { // Reflective or diffuse
        // The transmissive part is zero here.
        float specular = fresnel * geometry * distribution;
        float kd = (1.0f - fresnel) * (1.0f - mat.metallic) * (1.0f - mat.transmittance);

        pdf = G1 * distribution / (4*abs(cos_v)) * specular_probability +
            pdf_cosine_hemisphere(out_dir) * diffuse_probability;

        if(!isnan(pdf) && !isinf(pdf) && pdf > 0.0f)
        {
            bsdf.diffuse += kd * cos_l / M_PI;
            bsdf.dielectric_reflection += fresnel * geometry * distribution * cos_l * (1.0f - mat.metallic);
            bsdf.metallic_reflection += geometry * distribution * cos_l * mat.metallic;
        }
        else pdf = 0;
    }
    else
    { // Transmissive
        float denom = mat.ior_in / mat.ior_out * cos_d + cos_o;
        // Un-predivide geometry term ;)
        geometry *= 4.0;

        pdf = (abs(cos_d * cos_o) * G1 * distribution) / (abs(cos_v) * denom * denom * M_PI) * transmissive_probability;

        if(!isnan(pdf) && !isinf(pdf) && pdf > 0.0f)
        {
            // This should be the reciprocal form, which is necessary when the light
            // source is inside the volume...
            bsdf.transmission += -cos_l * abs(cos_d * cos_o) * mat.transmittance * (1.0f - mat.metallic) * (1.0f - fresnel) * geometry * distribution / (denom * denom);
        }
        else pdf = 0;
    }
    return pdf;
}

void material_bsdf_sample(
    vec4 uniform_random,
    vec3 view_dir,
    sampled_material mat,
    out vec3 out_dir,
    inout bsdf_lobes bsdf,
    out float pdf
){
#if defined(BOUNCE_HEMISPHERE)
    if(mat.transmittance > 0.0f)
    {
        out_dir = sample_sphere(uniform_random.xy);
        pdf = 0.25f/M_PI;
    }
    else
    {
        out_dir = sample_hemisphere(uniform_random.xy);
        pdf = 0.5f/M_PI;
    }
    ggx_bsdf_pdf(out_dir, view_dir, mat, bsdf);
#elif defined(BOUNCE_COSINE_HEMISPHERE)
    float split = mat.transmittance * 0.5f;
    out_dir = (uniform_random.z < split ? -1 : 1) * sample_cosine_hemisphere(uniform_random.xy);
    pdf = abs(out_dir.z / M_PI) * (uniform_random.z < split ? split : 1.0f-split);

    ggx_bsdf_pdf(out_dir, view_dir, mat, bsdf);
#else
    ggx_bsdf_sample(uniform_random, view_dir, mat, out_dir, bsdf, pdf);
#endif
}

float material_bsdf_pdf(
    vec3 out_dir,
    vec3 view_dir,
    sampled_material mat,
    inout bsdf_lobes bsdf
){
#if defined(BOUNCE_HEMISPHERE)
    ggx_bsdf_pdf(out_dir, view_dir, mat, bsdf);
    if(mat.transmittance == 0 && out_dir.z <= 0) return 0.0f;
    return mat.transmittance > 0.0f ? 0.25f/M_PI : 0.5f/M_PI;
#elif defined(BOUNCE_COSINE_HEMISPHERE)
    ggx_bsdf_pdf(out_dir, view_dir, mat, bsdf);

    if(mat.transmittance == 0 && out_dir.z <= 0) return 0.0f;
    float split = mat.transmittance * 0.5f;
    return abs(out_dir.z / M_PI) * (out_dir.z < 0 ? split : 1.0f-split);
#else
    return ggx_bsdf_pdf(out_dir, view_dir, mat, bsdf);
#endif
}

#endif
