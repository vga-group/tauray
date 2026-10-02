#ifndef TR_VOXELIZER_STAGE_HH
#define TR_VOXELIZER_STAGE_HH
#include "context.hh"
#include "raster_pipeline.hh"
#include "compute_pipeline.hh"
#include "timer.hh"
#include "scene_stage.hh"
#include "stage.hh"

namespace tr
{

class voxelizer_stage: public single_device_stage
{
public:
    struct options
    {
        int map_resolution = 128;
        aabb volume = aabb{vec3(-1), vec3(1)};

        bool track_history = false;
        bool reset_history_on_empty = false;
    };

    voxelizer_stage(
        device& dev,
        scene_stage& ss,
        const options& opt
    );

    // Voxel values:
    // 0   - has never been occupied
    // 1.0 - is currently occupied
    texture& get_map();

    // The voxel value is the number of frames the voxel has been occupied.
    texture& get_history_map();

    void reset();

private:
    void update(uint32_t frame_index) override;

    options opt;
    texture occupancy;
    texture mipped;
    std::optional<texture> mipped_history;

    scene_stage* ss;
    raster_pipeline raster;
    push_descriptor_set raster_desc;
    compute_pipeline clear;
    push_descriptor_set clear_desc;
    compute_pipeline gather;
    push_descriptor_set gather_desc;
    timer stage_timer;
    uint32_t history_counter;
};

}

#endif
