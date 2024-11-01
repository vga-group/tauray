#ifndef TAURAY_RESTIR_STAGE_HH
#define TAURAY_RESTIR_STAGE_HH

#include "gbuffer.hh"
#include "stage.hh"
#include "texture.hh"
#include "rt_pipeline.hh"
#include "compute_pipeline.hh"
#include "sampler.hh"
#include "descriptor_set.hh"
#include "environment_map.hh"
#include "shadow_map.hh"
#include "rt_common.hh"
#include "timer.hh"

namespace tr
{

class scene_stage;

class restir_stage: public single_device_stage
{
public:
    enum shift_mapping_type
    {
        RECONNECTION_SHIFT = 0,
        RANDOM_REPLAY_SHIFT = 1,
        HYBRID_SHIFT = 2
    };

    struct options
    {
        // The default value is assuming a reconstructed / inexact position, as
        // caused by rasterization and depth-buffer reconstruction. If the
        // G-buffer originates from a ray tracer, you can set this much lower.
        float min_ray_dist = 1e-3;
        float max_ray_dist = 1e9;
        bool opaque_only = false;
        uint32_t max_bounces = 2; // If set to 1, this is equivalent to ReSTIR DI.
        float min_spatial_search_radius = 1.0f;
        // The search radius may be heuristically adjusted, but this is scale
        // and upper limit for it. No exceptions are made for the minimum,
        // though!
        float max_spatial_search_radius = 32.0f;

        // If the temporal reprojection fails, any other old pixels can also be
        // used in some circumstances. This parameter adjusts how far away they
        // can be searched from.
        float temporal_search_base_radius = 4.0f;
        float temporal_search_widening = 4.0f;
        // With 0, there's no temporal "search", it just tries to use the
        // reprojected pixel as-is. The search attempts have a small cost, but
        // also a very small benefit.
        uint32_t temporal_reuse_search_attempts = 0;

        // Allows assuming that the material for a reprojected point in the
        // previous frame is the same as in the current frame.
        // With reconnection & random replay shifts, this causes very slight,
        // likely unnoticeable bias. With hybrid shifts, this can cause
        // noticeable darkening during movement near texels with major
        // roughness differences.
        // Enabled by force if temporal material inputs are missing, as there
        // is no other option in that case.
        bool assume_unchanged_material = false;

        // Causes bias that could be noticeable in dynamic scenes. Allows
        // assuming that the radiance along a light path does not change between
        // frames.
        bool assume_unchanged_reconnection_radiance = false;

        // Skips checking for reconnection visibility in temporal reuse.
        bool assume_unchanged_temporal_visibility = false;

        // Only matters when assume_unchanged_temporal_visibility = true.
        // Causes very slight, likely unnoticeable bias.
        // Allows assuming that the acceleration structures of the current
        // frame are equivalent with those of the previous frame.
        bool assume_unchanged_acceleration_structures = false;

        // Improves quality at grazing angles, at the cost of a little bit of
        // performance.
        bool spatial_sample_oriented_disk = true;

        // Number of nearby pixels picked for spatial reuse.
        // Set to zero to disable spatial reuse.
        // More spatial samples = less noise but more time. There are no quality
        // downsides to taking more spatial samples.
        uint32_t spatial_samples = 1;

        // Number of canonical samples per frame.
        uint32_t canonical_samples = 1;

        // Number of ReSTIR passes to do. This is kind of "SPP" in that
        // increasing this yields superior quality and eventually converges to a
        // completely noise-free image. Multi-pass renders store the confidence
        // in the alpha channel.
        uint32_t passes = 1;

        // Whether to create a new canonical sample for each pass. It's faster
        // with false, but this makes 'passes' cause sample impoverishment,
        // which will make your image less stable unless max_confidence is low,
        // like 2-4. In which case your image will be noisier again.
        bool do_canonical_samples_for_passes = false;

        // Adjusts the amount of sample reuse. You probably shouldn't go above
        // 32, because it causes various correlation issues
        // (= "northern lights"). Also, images using too much temporal data
        // don't converge when accumulating.
        // 32767 is the absolute maximum until everything breaks down.
        float max_confidence = 16;

        // Toggle temporal reuse on/off. Temporal reuse is fairly cheap and
        // offers a big quality improvement, but also increases image
        // instability and sample impoverishment, so be careful with
        // max_confidence.
        bool temporal_reuse = true;

        // Relative scale of the reconnection boundary for hybrid shift mapping.
        // This is in world-space scale.
        float reconnection_scale = 2.0f;

        // The shift mapping type affects the expected graphics artifacts and
        // performance. You can the artifacts via setting max_confidence
        // lower, but that also increases noise.
        //
        // - RECONNECTION_SHIFT is good for direct lighting and okay for diffuse
        //   GI. It's also very fast.
        // - RANDOM_REPLAY_SHIFT is generally slightly worse quality and slower,
        //   but allows for higher max_confidence than RECONNECTION_SHIFT and
        //   works with specular surfaces as well.
        // - HYBRID_SHIFT is a combination of the previous two, where it delays
        //   reconnection further in that path. It's fairly good, but slower
        //   than RECONNECTION_SHIFT.
        shift_mapping_type shift_map = HYBRID_SHIFT;

        // Accumulate successive samples for a reference render. This
        // unfortunately can kinda clash with doing multiple passes.
        // Each accumulated sample is assumed to be at max_confidence. It'll
        // still converge just fine, but suboptimally.
        bool accumulate = false;

        // Enables shading each hit with all explicit lights for indirect
        // bounces, using their shadow maps. Currently, only bilinear
        // interpolation is supported the shadow maps.
        //
        // This option also disables direct light from explicit lights: you are
        // expected to render that separately using e.g. forward_stage, which
        // gives you more control over how the shadow maps are rendered.
        //
        // Requires that binning_stage does not bin explicit lights.
        bool shade_all_explicit_lights = false;

        // Filter for shade_all_explicit_lights.
        shadow_map_filter sm_filter = {0, 0, 0, 0};

        // Enables using ambient light and light probes in indirect bounces.
        // Should be used in conjunction with shade_all_explicit_lights.
        bool shade_fake_indirect = false;

        // Writes output to current gbuffer's demodulated colors if they are
        // present. Accumulation is not supported with demodulation.
        bool demodulated_output = false;

        // Whether to permute temporal samples or not. This increases noise,
        // but reduces temporal correlations, which can be very useful if you
        // intend to denoise the result.
        uint32_t temporal_permutation = 0;

        float regularization_gamma = 0.0f; // 0 disables path regularization

        light_sampling_weights sampling_weights;
        int camera_index = 0;

        bool expect_taa_jitter = false;
    };

    // TODO: Doesn't expect multi-view targets for now.
    restir_stage(
        device& dev,
        scene_stage& ss,
        gbuffer_target& current_buffers,
        gbuffer_target& previous_buffers,
        const options& opt
    );

    void reset_accumulation();

protected:
    void update(uint32_t frame_index) override;

private:
    void record_canonical_pass(vk::CommandBuffer cmd, uint32_t frame_index, int pass_index);
    void record_spatial_pass(vk::CommandBuffer cmd, uint32_t frame_index, bool final_pass, int pass_index);

    scene_stage* scene_data;

    // Generates one canonical path per frame.
    compute_pipeline canonical;
    push_descriptor_set canonical_set;

    // Merges the canonical path with temporal history.
    compute_pipeline temporal;
    push_descriptor_set temporal_set;

    // Traces rays for spatial reuse candidates & calculates MIS weights.
    compute_pipeline spatial_trace;
    push_descriptor_set spatial_trace_set;
    int selection_tile_size;

    // Gathers spatial samples and writes the final shade.
    compute_pipeline spatial_gather;
    push_descriptor_set spatial_gather_set;

    std::optional<texture> selection_data;
    // Present is spatial_samples > 1.
    // float mis
    // float half_jacobian
    std::optional<texture> spatial_mis_data;
    std::optional<texture> spatial_candidate_color;

    // Present if accumulation is used. Otherwise, this data is placed
    // temporarily in the gbuffer instead. Not supported with demodulation,
    // which disables accumulation anyway.
    std::optional<texture> cached_sample_color;

    // Even though textures are a bit clumsy, we pack our reservoir data there
    // to benefit from the locality-preserving curve that is likely used in
    // texture layouts. It's also slightly easier to drop certain parts out when
    // unneeded.
    struct reservoir_textures
    {
        // Always present.
        // Make no attempt at compressing the values of this buffer; you can
        // only cause subtle precision errors with negligible performance
        // benefit. Halfs are not enough for any of these. You have been warned.
        //
        // float target_function_value
        // float ucw
        // float base_path_jacobian_part
        // uint confidence_path_length:
        //     0..14: confidence
        //     15: nee_terminal
        //     16..17: head_lobe (0 = NEE/ALL, 1 = DIFFUSE, 2 = TRANSMISSION, 3 = REFLECTION)
        //     18..19: tail_lobe (0 = NEE/ALL, 1 = DIFFUSE, 2 = TRANSMISSION, 3 = REFLECTION)
        //     20..25: head length
        //     26..31: tail length
        //
        // head_lobe == 0 implicates NEE sample.
        std::optional<texture> ris_data;

        // Present unless shift_map == RANDOM_REPLAY_SHIFT
        // float hit_info_x (Barycoords, normals, envmap sample dir)
        // float hit_info_y (Barycoords, normals, envmap sample dir)
        // uint instance_id
        // uint primitive_id
        std::optional<texture> reconnection_data;

        // Present unless shift_map == RANDOM_REPLAY_SHIFT
        // float3 radiance_estimate
        // float luminance_estimate
        std::optional<texture> reconnection_radiance;

        // Present unless max_bounces == 1 and shift_map == RECONNECTION_SHIFT
        // uint head_rng_seed
        // uint tail_rng_seed
        // float incident_direction_x
        // float incident_direction_y
        std::optional<texture> rng_seeds;
    };
    reservoir_textures reservoir_data[2];
    int reservoir_data_parity;

    gbuffer_target current_buffers;
    gbuffer_target previous_buffers;

    sampler gbuf_sampler;

    options opt;
    unsigned accumulated_samples;
    uint64_t valid_history_frame;
    timer stage_timer;
    timer canonical_timer;
    timer temporal_timer;
    timer trace_timer;
    timer gather_timer;
};

}

#endif
