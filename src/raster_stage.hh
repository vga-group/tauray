#ifndef TAURAY_RASTER_STAGE_HH
#define TAURAY_RASTER_STAGE_HH
#include "context.hh"
#include "timer.hh"
#include "texture.hh"
#include "raster_pipeline.hh"
#include "sampler.hh"
#include "gbuffer.hh"
#include "gpu_buffer.hh"
#include "stage.hh"

namespace tr
{

class scene;
class shadow_map_renderer;
class sh_grid;
class raster_stage: public single_device_stage
{
public:
    struct options
    {
        // Must be a power of two.
        size_t max_samplers = 128;
        size_t max_3d_samplers = 128;
        bool clear_color = true;
        bool clear_depth = true;
        bool sample_shading = false;

        // Shadow filtering options.
        int pcf_samples = 64; // 0 => bilinear interpolation
        int omni_pcf_samples = 16; // 0 => bilinear interpolation
        int pcss_samples = 32; // 0 => disable PCSS
        // The minimum radius prevents PCSS from degrading to bilinear filtering
        // near shadow caster.
        float pcss_minimum_radius = 0.0f;

        // Spherical harmonics options
        bool use_probe_visibility = false;
        int sh_order = 2;

        // Required for some denoisers to drop albedo from transparent textures in the gbuffer
        bool force_alpha_to_coverage = false;
        // Output layout to avoid excess work in render pass
        vk::ImageLayout output_layout = vk::ImageLayout::eColorAttachmentOptimal;
    };

    raster_stage(
        device& dev,
        const std::vector<gbuffer_target>& output_array_targets,
        const options& opt
    );

    void set_scene(scene* s);
    scene* get_scene();

private:
    void record_command_buffers();

    std::vector<std::unique_ptr<raster_pipeline>> array_pipelines;
    std::vector<gbuffer_target> output_targets;
    options opt;

    sampler brdf_integration_sampler;
    scene* cur_scene;
    texture brdf_integration;
    texture noise_vector_2d;
    texture noise_vector_3d;
    timer raster_timer;
};

}

#endif
