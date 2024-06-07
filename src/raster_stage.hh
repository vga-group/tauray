#ifndef TAURAY_RASTER_STAGE_HH
#define TAURAY_RASTER_STAGE_HH
#include "context.hh"
#include "timer.hh"
#include "texture.hh"
#include "raster_pipeline.hh"
#include "sampler.hh"
#include "gbuffer.hh"
#include "gpu_buffer.hh"
#include "shadow_map.hh"
#include "stage.hh"

namespace tr
{

class scene_stage;
class shadow_map_renderer;
class sh_grid;
class raster_stage: public single_device_stage
{
public:
    struct options
    {
        bool clear_color = true;
        bool clear_depth = true;
        bool sample_shading = false;

        shadow_map_filter filter = {};

        // Spherical harmonics options
        bool use_probe_visibility = false;
        int sh_order = 2;
        bool estimate_indirect = true;

        // Required for some denoisers to drop albedo from transparent textures in the gbuffer
        bool force_alpha_to_coverage = false;
        // Output layout to avoid excess work in render pass
        vk::ImageLayout output_layout = vk::ImageLayout::eColorAttachmentOptimal;

        unsigned base_camera_index = 0;

        // Attempts to undo TAA jitter for textures, increasing their clarity.
        bool unjitter_textures = false;
    };

    raster_stage(
        device& dev,
        scene_stage& ss,
        const std::vector<gbuffer_target>& output_array_targets,
        const options& opt
    );

    raster_stage(
        device& dev,
        scene_stage& ss,
        gbuffer_target& output_target,
        const options& opt
    );

protected:
    void update(uint32_t frame_index) override;

private:
    std::vector<std::unique_ptr<raster_pipeline>> array_pipelines;
    std::vector<gbuffer_target> output_targets;
    options opt;

    uint32_t scene_state_counter;
    scene_stage* ss;
    timer raster_timer;
};

}

#endif
