#ifndef SVGF_GLSL
#define SVGF_GLSL

#include "projection.glsl"
#include "camera.glsl"
#include "gbuffer.glsl"

layout(push_constant) uniform push_constant_buffer
{
    ivec2 size;
    int iteration;
    int iteration_count;
    int spec_iteration_count;
    int atrous_kernel_radius;
    float sigma_n;
    float sigma_z;
    float sigma_l;
    float temporal_alpha_color;
    float temporal_alpha_moments;
} control;

bool is_in_screen(ivec2 p)
{
    return all(greaterThanEqual(p, ivec2(0))) && all(lessThan(p, control.size));
}

float saturate(float x) {return clamp(x, 0.0, 1.0);}
vec2 saturate(vec2 x) {return clamp(x, 0.0, 1.0);}
vec3 saturate(vec3 x) {return clamp(x, 0.0, 1.0);}
vec4 saturate(vec4 x) {return clamp(x, 0.0, 1.0);}

vec3 viridis_quintic( float x )
{
	x = saturate( x );
	vec4 x1 = vec4( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
	vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
	return vec3(
		dot( x1.xyzw, vec4( +0.280268003, -0.143510503, +2.225793877, -14.815088879 ) ) + dot( x2.xy, vec2( +25.212752309, -11.772589584 ) ),
		dot( x1.xyzw, vec4( -0.002117546, +1.617109353, -1.909305070, +2.701152864 ) ) + dot( x2.xy, vec2( -1.685288385, +0.178738871 ) ),
		dot( x1.xyzw, vec4( +0.300805501, +2.614650302, -12.019139090, +28.933559110 ) ) + dot( x2.xy, vec2( -33.491294770, +13.762053843 ) ) );
}

float get_specular_dominant_factor(float n_dot_v, float roughness)
{
    // https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf

#if 1
    // Simple approximation
    float smoothness = saturate(1 - roughness);
    return smoothness * (sqrt(smoothness) + roughness);
#else
    float l = 0.298475f * log(39.4115f - 39.0029f * roughness);
    return clamp(pow(max(1.0f - n_dot_v, 0.0f), 10.8649f) * (1.0f - l) + l, 0.0f, 1.0f);
#endif
}

// View should point away from surface.
vec3 get_specular_dominant_dir(vec3 view, vec3 normal, float roughness)
{
    float n_dot_v = dot(normal, view);
    vec3 refl = reflect(view, normal);
    return normalize(mix(normal, refl, get_specular_dominant_factor(n_dot_v, roughness)));
}

// Derived through isotropic spherical gaussians; may be inaccurate. 1.0f acts
// as diffuse.
// Similarity metric from: "Specular Lobe-Aware Filtering and Upsampling for
// Interactive Indirect Illumination"
float specular_lobe_similarity(
    vec3 lobe_dir1, float ndotv1, float roughness1,
    vec3 lobe_dir2, float ndotv2, float roughness2
){
    // Exact
    //float sharpness1 = 0.533f / (roughness1 * roughness1 * max(ndotv1, 0.0001f));
    //float sharpness2 = 0.533f / (roughness2 * roughness2 * max(ndotv2, 0.0001f));
    //float amplitude1 = sqrt(sharpness1 / (M_PI * (1.0f - exp(-4.0f * sharpness1))));
    //float amplitude2 = sqrt(sharpness2 / (M_PI * (1.0f - exp(-4.0f * sharpness2))));
    //float dm = length(sharpness1 * lobe_dir1 + sharpness2 * lobe_dir2);
    //float expo = exp(dm - sharpness1 - sharpness2) * amplitude1 * amplitude2;
    //float other = 1.0f - exp(-2.0f * dm);
    //return (2.0f * M_PI * expo * other) / dm;
    // Approximate
    float sharpness1 = 0.533f / (roughness1 * roughness1);
    float sharpness2 = 0.533f / (roughness2 * roughness2);
    float dm = length(sharpness1 * lobe_dir1 + sharpness2 * lobe_dir2);
    float expo = exp(dm - sharpness1 - sharpness2) * sqrt(sharpness1 * sharpness2);
    return clamp((2.0f * expo) / dm, 0.0f, 1.0f);
}
// View points away from surface.
float specular_lobe_similarity(
    vec3 view1, vec3 normal1, float roughness1,
    vec3 view2, vec3 normal2, float roughness2
){
    vec3 axis1 = get_specular_dominant_dir(view1, normal1, roughness1);
    vec3 axis2 = get_specular_dominant_dir(view2, normal2, roughness2);
    return specular_lobe_similarity(
        axis1, dot(view1, normal1), roughness1,
        axis2, dot(view2, normal2), roughness1
    );
}

float get_specular_lobe_half_angle(float percentage_of_energy, float roughness)
{
    return atan(roughness * sqrt(percentage_of_energy / (1.0 - percentage_of_energy)));
}

// From brdf.h, pasted here for reference
// Approximates the directional-hemispherical reflectance of the micriofacet specular BRDF with GG-X distribution
// Source: "Accurate Real-Time Specular Reflections with Radiance Caching" in Ray Tracing Gems by Hirvonen et al.
vec3 specularGGXReflectanceApprox(vec3 specular_f0, float alpha, float NdotV)
{
	const mat2 A = transpose(mat2(
		0.995367f, -1.38839f,
		-0.24751f, 1.97442f
	));

	const mat3 B = transpose(mat3(
		1.0f, 2.68132f, 52.366f,
		16.0932f, -3.98452f, 59.3013f,
		-5.18731f, 255.259f, 2544.07f
	));

	const mat2 C = transpose(mat2(
		-0.0564526f, 3.82901f,
		16.91f, -11.0303f
	));

	const mat3 D = transpose(mat3(
		1.0f, 4.11118f, -1.37886f,
		19.3254f, -28.9947f, 16.9514f,
		0.545386f, 96.0994f, -79.4492f
	));

	const float alpha2 = alpha * alpha;
	const float alpha3 = alpha * alpha2;
	const float NdotV2 = NdotV * NdotV;
	const float NdotV3 = NdotV * NdotV2;

	const float E = dot(A * vec2(1.0f, NdotV), vec2(1.0f, alpha));
	const float F = dot(B * vec3(1.0f, NdotV, NdotV3), vec3(1.0f, alpha, alpha3));

	const float G = dot(C * vec2(1.0f, NdotV), vec2(1.0f, alpha));
	const float H = dot(D * vec3(1.0f, NdotV2, NdotV3), vec3(1.0f, alpha, alpha3));

	// Turn the bias off for near-zero specular 
	const float biasModifier = saturate(dot(specular_f0, vec3(0.333333f, 0.333333f, 0.333333f)) * 50.0f);

	const float bias = max(0.0f, (E / F)) * biasModifier;
	const float scale = max(0.0f, (G / H));

	return vec3(bias, bias, bias) + vec3(scale, scale, scale) * specular_f0;
}

// f0 converted to float for clarity, since colors have already been demodulated from our ipnuts
float environment_term_rtg(float f0, float NoV, float roughness)
{
    float m = roughness;

    vec4 X;
    X.x = 1.0;
    X.y = NoV;
    X.z = NoV * NoV;
    X.w = NoV * X.z;

    vec4 Y;
    Y.x = 1.0;
    Y.y = m;
    Y.z = m * m;
    Y.w = m * Y.z;

    mat2 M1 = mat2(0.99044, 1.29678, -1.28514, -0.755907);
    mat3 M2 = mat3(1.0, 20.3225, 121.563, 2.92338, -27.0302, 626.13, 59.4188, 222.592, 316.627);

    mat2 M3 = mat2(0.0365463, 9.0632, 3.32707, -9.04756);
    mat3 M4 = mat3(1.0, 9.04401, 5.56589, 3.59685, -16.3174, 19.7886, -1.36772, 9.22949, -20.2123);

    float bias = dot(M1 * X.xy, Y.xy) / (dot(M2 * X.xyw, Y.xyw));
    float scale = dot(M3 * X.xy, Y.xy) / (dot(M4 * X.xzw, Y.xyw));

    return clamp(bias + scale * f0, 0.0, 1.0);
}

bool get_view_pos(sampler2DArray depth_sampler, ivec3 p, camera_data cam, out vec3 pos)
{
    ivec2 size = control.size;
    float depth = texelFetch(depth_sampler, p, 0).r;
    float linear_depth = linearize_depth(depth * 2.0f - 1.0f, cam.projection_info);

    vec2 uv = (vec2(p.xy)+0.5f)/vec2(size);
    pos = unproject_position(linear_depth, vec2(uv.x, 1-uv.y), cam.projection_info, cam.pan.xy);
    return depth == 1.0f;
}

bool get_pos(sampler2DArray depth_sampler, ivec3 p, camera_data cam, out vec3 pos)
{
    bool valid = get_view_pos(depth_sampler, p, cam, pos);
    pos = (cam.view_inverse * vec4(pos, 1)).xyz;
    return valid;
}

vec3 get_world_pos(camera_data cam, vec3 view_pos)
{
    return (cam.view_inverse * vec4(view_pos, 1.0)).xyz;
}

struct Bilinear 
{ 
    vec2 origin;
    vec2 weights; 
};

Bilinear get_bilinear_filter( vec2 uv, vec2 tex_size )
{
    Bilinear result;
    result.origin = floor( uv * tex_size - 0.5 );
    result.weights = fract( uv * tex_size - 0.5 );
    return result;
}

vec4 get_bilinear_weights(Bilinear bilinear)
{
    vec4 bilinear_weights = {
        (1.0 - bilinear.weights.x) * (1.0 - bilinear.weights.y), (bilinear.weights.x) * (1.0 - bilinear.weights.y),
        (1.0 - bilinear.weights.x) * (bilinear.weights.y), (bilinear.weights.x) * (bilinear.weights.y)
    };

    return bilinear_weights;
}

vec4 bicubic_filter(sampler2DArray tex, vec3 uv)
{
    vec2 texture_size = vec2(textureSize(tex, 0).xy);
    vec4 size_params = vec4(1.0 / vec2(texture_size), vec2(texture_size));
    vec2 pos = size_params.zw * uv.xy;
    vec2 center_pos = floor(pos - 0.5) + 0.5;
    vec2 f = pos - center_pos;
    vec2 f2 = f * f;
    vec2 f3 = f * f2;

    float c = 0.5;
    vec2 w0 =        -c  * f3 +  2.0 * c        * f2 - c * f;
    vec2 w1 =  (2.0 - c) * f3 - (3.0 - c)       * f2         + 1.0;
    vec2 w2 = -(2.0 - c) * f3 + (3.0 - 2.0 * c) * f2 + c * f;
    vec2 w3 =         c  * f3 -               c * f2;

    vec2 w12 = w1 + w2;
    vec2 tc12 = size_params.xy * (center_pos + w2 / w12);
    vec4 center_color = texture(tex, vec3(tc12.x, tc12.y, uv.z));

    vec2 tc0 = size_params.xy * (center_pos - 1.0);
    vec2 tc3 = size_params.xy * (center_pos + 2.0);
    vec4 color = texture(tex, vec3(tc12.x, tc0.y, uv.z)) * (w12.x * w0.y) + 
        texture(tex, vec3(tc0.x, tc12.y, uv.z)) * (w0.x * w12.y) +
        center_color * (w12.x * w12.y) + 
        texture(tex, vec3(tc3.x, tc12.y, uv.z)) * (w3.x * w12.y) +
        texture(tex, vec3(tc12.x, tc3.y, uv.z)) * (w12.x * w3.y);

    float total_w =  (w12.x * w0.y) + (w0.x * w12.y) + (w12.x * w12.y) + (w3.x * w12.y) + (w12.x * w3.y);

    return color / total_w;
}

vec3 sample_gbuffer_normal(sampler2DArray normal_tex, Bilinear bilinear_filter, float layer)
{
    vec3 n00 = unpack_gbuffer_normal(texelFetch(normal_tex, ivec3(bilinear_filter.origin + vec2(0, 0), layer), 0).xy);
    vec3 n10 = unpack_gbuffer_normal(texelFetch(normal_tex, ivec3(bilinear_filter.origin + vec2(1, 0), layer), 0).xy);
    vec3 n01 = unpack_gbuffer_normal(texelFetch(normal_tex, ivec3(bilinear_filter.origin + vec2(0, 1), layer), 0).xy);
    vec3 n11 = unpack_gbuffer_normal(texelFetch(normal_tex, ivec3(bilinear_filter.origin + vec2(1, 1), layer), 0).xy);

    vec4 weights = get_bilinear_weights(bilinear_filter);
    vec3 normal = n00 * weights.x + n10 * weights.y + n01 * weights.z + n11 * weights.w;

    return normalize(normal);
}

vec4 sample_bilinear_with_custom_weights(sampler2DArray tex, Bilinear bilinear, float layer, vec4 weights)
{
    vec4 s00 = texelFetch(tex, ivec3(bilinear.origin + vec2(0, 0), layer), 0);
    vec4 s10 = texelFetch(tex, ivec3(bilinear.origin + vec2(1, 0), layer), 0);
    vec4 s01 = texelFetch(tex, ivec3(bilinear.origin + vec2(0, 1), layer), 0);
    vec4 s11 = texelFetch(tex, ivec3(bilinear.origin + vec2(1, 1), layer), 0);

    if (any(isnan(s00))) s00 = vec4(0.0);
    if (any(isnan(s10))) s10 = vec4(0.0);
    if (any(isnan(s01))) s01 = vec4(0.0);
    if (any(isnan(s11))) s11 = vec4(0.0);

    vec4 s = s00 * weights.x + s10 * weights.y + s01 * weights.z + s11 * weights.w;

    float w_sum = dot(weights, vec4(1.0));
    s = w_sum < 1e-4 ? vec4(0.0) : s / w_sum;

    return s;    
}

vec3 demodulate_specular(vec3 specular, vec3 V, vec3 N, float metallic, float roughness)
{
    float NoV = max(dot(N, V), 0.0);

    // Our input already has f0 demodulated from metallics, but not dielectrics
    // Therefore the f0 used for demodulation with the environment split sum approximation
    // is 1.0 for metallics and for dielectrics it depends on the ior which we no longer have access to
    // dielectric_f0 = 0.04 is a reasonable approximation for most common materials.
    float f0 = mix(0.04, 1.0, metallic);
    //vec3 specular_reflectance = specularGGXReflectanceApprox(f0, roughness, NoV);
    float specular_reflectance = environment_term_rtg(f0, NoV, roughness);
    return specular.rgb / specular_reflectance;
}

float get_plane_distance_weight(vec3 Xref, vec3 X, vec3 N, float frustum_size)
{
    const float plane_dist_sensitivity = 0.005;
    float plane_dist = abs(dot((X - Xref), N));
    return step(plane_dist / frustum_size, plane_dist_sensitivity);
}

//====================================================================================
// Configurable params
//====================================================================================

// Base blur radius in pixels, actually blur radius is scaled by frustum size
#define PREPASS_DIFFUSE_BLUR_RADIUS 30.0
#define PREPASS_SPECULAR_BLUR_RADIUS 30.0

#define TEMPORAL_ACCUMULATION_USE_BICUBIC_FILTER 1
#define TEMPORAL_ACCUMULATION_USE_SPEC_MIN_DIST_3X3 0

#define ATROUS_ITERATIONS 5
#define ATROUS_RADIUS 1 

// Randomly offset sample positions to mitigate "ringing"
#define ATROUS_RANDOM_OFFSET 0

#define DEMODULATION_USE_SPLIT_SUM_APPROXIMATION 1

//====================================================================================
// Toggle different passes on and off, useful for debugging
//====================================================================================

#define DENOISING_ENABLED 1

#if DENOISING_ENABLED == 1

#define HIT_DIST_RECONSTRUCTION_ENABLED 1

#define TEMPORAL_ACCUMULATION_ENABLED 1

#define DISOCCLUSION_FIX_ENABLED 1
#define DISOCCLUSION_FIX_USE_EDGE_STOPPERS 1

#define FIREFLY_SUPPRESSION_ENABLED 1

#define ATROUS_ENABLED 1

#endif

#define OUTPUT_DENOISED_DIFFUSE 0
#define OUTPUT_VARIANCE 1
#define OUTPUT_HIST_LENGTH 2
#define OUTPUT_UNFILTERED_VARIANCE 3
#define OUTPUT_REMODULATED_DENOISED_DIFFUSE 4
#define OUTPUT_DENOISED_SPECULAR 5
#define OUTPUT_REMODULATED_DENOISED_DIFFUSE_AND_SPECULAR 6
#define OUTPUT_DIFFUSE_HITDIST 7
#define OUTPUT_SPECULAR_VARIANCE 8
#define OUTPUT_UNFILTERED_SPECULAR_VARIANCE 9
#define OUTPUT_REMODULATED_DENOISED_SPECULAR 10
#define OUTPUT_RAW_INPUT 11
#define OUTPUT_SPECULAR_HIT_DIST 12

#define FINAL_OUTPUT 6

#define MAX_ACCUMULATED_FRAMES 30
//#define MAX_ACCUMULATED_FRAMES 100

#endif // SVGF_GLSL
