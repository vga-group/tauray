#ifndef RESTIR_GLSL
#define RESTIR_GLSL
#define USE_RAY_QUERIES
#extension GL_EXT_ray_query : enable

// This ReSTIR implementation follows the SIGGRAPH 2023 course "A Gentle
// Introduction to ReSTIR". I've tried to comment ReSTIR-specific math with
// names that correspond to what is used in the course notes.
//
// Also, I abbreviate "unbiased contribution weight" as UCW, so that's what that
// is.

#ifdef RESTIR_TEMPORAL
#define CALC_PREV_VERTEX_POS
#endif

#define SCENE_SET 1
#define SCENE_RASTER_SET 2
#define USE_EXPLICIT_GRADIENTS
#define CALC_TRIANGLE_CORNERS
#ifdef RAY_TRACING
#include "rt.glsl"
#else
#include "scene.glsl"
#endif
#include "scene_raster.glsl"

#define RESTIR_DI (MAX_BOUNCES == 1)

#if defined(USE_RANDOM_REPLAY_SHIFT)
#define RESTIR_HAS_RECONNECTION_DATA false
#else
#define RESTIR_HAS_RECONNECTION_DATA true
#endif

#if MAX_BOUNCES != 1 || !defined(USE_RECONNECTION_SHIFT)
#define RESTIR_HAS_SEEDS true
#else
#define RESTIR_HAS_SEEDS false
#endif

struct restir_config
{
    shadow_mapping_parameters sm_params;
    uvec2 display_size;
    float min_ray_dist;
    float max_ray_dist;
    //float regularization_gamma;
    float reconnection_scale;
    float max_confidence;

    float min_spatial_radius;
    float max_spatial_radius;
};

const uint NULL_INSTANCE_ID = 0xFFFFFFFF;
const uint POINT_LIGHT_INSTANCE_ID = 0xFFFFFFFF-1;
const uint DIRECTIONAL_LIGHT_INSTANCE_ID = 0xFFFFFFFF-2;
const uint ENVMAP_INSTANCE_ID = 0xFFFFFFFF-3;
// Both directional + envmap.
const uint MISS_INSTANCE_ID = 0xFFFFFFFF-4;
const uint UNCONNECTED_PATH_ID = MISS_INSTANCE_ID-1;

struct reconnection_vertex
{
    // Either triangle mesh index or type tag (see above INSTANCE_ID constants)
    uint instance_id; // Δ
    // e.g. triangle index, point light index etc.
    uint primitive_id; // Δ
    // e.g. barycentric coords
    vec2 hit_info; // λ_1, λ_2
    vec3 radiance_estimate; // L, emission if terminal
    vec3 incident_direction; // Direction of incoming radiance, ignored if terminal
};

struct restir_sample
{
    reconnection_vertex vertex; // ReVertex
    uint head_rng_seed; // Used to replay path up to reconnection vertex
    uint tail_rng_seed; // Used to replay path after reconnection vertex
    uint head_lobe; // Sampled lobe at last vertex before reconnection
    uint tail_lobe; // Sampled lobe at first vertex after reconnection
    uint head_length; // Number of vertices before reconnection
    uint tail_length; // Number of vertices after reconnection
    bool nee_terminal; // true if terminal vertex is NEE.
    float base_path_jacobian_part; // J
    float radiance_luminance; // Used for temporal gradients.
};

struct reservoir
{
    restir_sample output_sample; // Y
    float ucw; // W_Y
    float target_function_value;
    float sum_weight; // w_sum
    float confidence; // c
};

void init_restir_sample(out restir_sample s, uint path_seed)
{
    s.vertex.instance_id = UNCONNECTED_PATH_ID;
    s.vertex.primitive_id = 0;
    s.vertex.hit_info = vec2(0);
    s.vertex.radiance_estimate = vec3(0);
    s.vertex.incident_direction = vec3(0);
    s.head_rng_seed = path_seed;
    s.tail_rng_seed = path_seed;
    s.head_lobe = 0;
    s.tail_lobe = 0;
    s.head_length = 0;
    s.tail_length = 0;
    s.nee_terminal = false;
    s.base_path_jacobian_part = 1.0f;
    s.radiance_luminance = 0;
}

void init_reservoir(out reservoir r)
{
    // The initial output value is a null sample.
    // output_sample is invalid if the UCW is negative!
    r.ucw = -1.0f;
    r.target_function_value = 0;
    init_restir_sample(r.output_sample, 0);
    r.output_sample.vertex.instance_id = NULL_INSTANCE_ID;
    r.sum_weight = 0.0f;
    r.confidence = 0;
}

bool update_reservoir(
    float rand,
    inout reservoir r,
    float target_function_value, // p^
    float resampling_weight,// w_i
    float confidence // c
){
    r.sum_weight += resampling_weight;
    r.confidence += confidence;
    if(rand * r.sum_weight < resampling_weight)
    {
        r.target_function_value = target_function_value;
        return true;
    }
    return false;
}

// This is the function our ReSTIR is optimizing. You could change it to
// something else if you wanted to, but improvements are probably limited to
// better approximations of luminance. Maybe you could take nonlinear
// tonemapping into account?
float target_function(vec4 primary_bsdf, vec3 radiance)
{
    vec3 value = radiance;
#ifdef DEMODULATE_OUTPUT
    value *= primary_bsdf.rgb;
#endif
    return rgb_to_luminance(value);
}

// Output color can be derived differently from the target function, which is
// the case for demodulated output.
vec4 output_color_function(vec4 primary_bsdf, vec3 radiance)
{
#ifdef DEMODULATE_OUTPUT
    return vec4(primary_bsdf.rgb * radiance, primary_bsdf.a);
    //return vec4(radiance, primary_bsdf.a);
#else
    return vec4(radiance, 0);
#endif
}

struct intersection_info
{
    vertex_data vd;
    sampled_material mat;
    // Light sources are separated due to sampling concerns. This is needed for
    // implementing next event estimation.
    vec3 light;

    // These PDFs are used for MIS. The local PDF is separated from envmap_pdf
    // only because the final PDF calculation is done outside of the intersection
    // function and directional + envmap combo need some more complicated weighting.
    float local_pdf; // PDF of a tri, point or directional light on its own
    float envmap_pdf; // PDF of an envmap
};

#define MAX_CANDIDATE_ATTEMPTS 4

#endif
