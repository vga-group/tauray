#ifndef TAURAY_AABB_SCENE_HH
#define TAURAY_AABB_SCENE_HH
#include "context.hh"
#include "timer.hh"
#include "acceleration_structure.hh"
#include "gpu_buffer.hh"

namespace tr
{

// This is a base class for implementing scenes of objects defined within AABBs;
// In other words, you should use this when the objects do not consist of
// triangle meshes. These are not supported in rasterization.
class aabb_scene
{
public:
    aabb_scene(
        context& ctx,
        const char* timer_name,
        size_t sbt_offset,
        size_t max_capacity = 1024
    );
    aabb_scene(const aabb_scene& s) = delete;
    aabb_scene(aabb_scene&& s) = delete;
    virtual ~aabb_scene() = default;

protected:
    size_t get_max_capacity() const;

    virtual size_t get_aabbs(vk::AabbPositionsKHR* aabb) = 0;

    // Must return true if the command buffers need to be re-recorded.
    void update_acceleration_structures(
        size_t device_index,
        uint32_t frame_index,
        bool& need_scene_reset,
        bool& command_buffers_outdated
    );

    void record_acceleration_structure_build(
        vk::CommandBuffer& cb,
        size_t device_index,
        uint32_t frame_index,
        bool update_only
    );

    void add_acceleration_structure_instances(
        vk::AccelerationStructureInstanceKHR* instances,
        size_t device_index,
        uint32_t frame_index,
        size_t& instance_index,
        size_t capacity
    ) const;

    void invalidate_acceleration_structures();

private:
    void init_acceleration_structures(const char* timer_name);

    context* ctx;
    size_t max_capacity;
    size_t sbt_offset;

    std::optional<bottom_level_acceleration_structure> blas;
    std::vector<gpu_buffer> aabb_buffer;
    struct as_update_data
    {
        bool scene_reset_needed = true;

        struct per_frame_data
        {
            bool command_buffers_outdated = true;
            unsigned aabb_count = 0;
        };
        per_frame_data per_frame[MAX_FRAMES_IN_FLIGHT];
        timer blas_update_timer;
    };
    std::vector<as_update_data> as_update;
};

}

#endif

