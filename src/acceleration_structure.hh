#ifndef TAURAY_ACCELERATION_STRUCTURE_HH
#define TAURAY_ACCELERATION_STRUCTURE_HH
#include "context.hh"
#include "transformable.hh"
#include "gpu_buffer.hh"
#include "vkm.hh"

namespace tr
{

class mesh;
class bottom_level_acceleration_structure
{
public:
    struct entry
    {
        // If nullptr, this entry is AABBs instead of a triangle mesh.
        const mesh* m = nullptr;
        size_t aabb_count = 0;
        // Pointer to AABB buffer
        gpu_buffer* aabb_buffer = nullptr;

        mat4 transform = mat4(1.0f);
        bool opaque = true;
    };

    bottom_level_acceleration_structure(
        device_mask dev,
        const std::vector<entry>& entries,
        bool backface_culled,
        bool dynamic,
        bool compact
    );

    void update_transforms(
        size_t frame_index,
        const std::vector<entry>& entries
    );

    void rebuild(
        device_id id,
        size_t frame_index,
        vk::CommandBuffer cb,
        const std::vector<entry>& entries,
        bool update = true
    );
    size_t get_updates_since_rebuild() const;
    vk::AccelerationStructureKHR get_blas_handle(device_id id) const;
    vk::DeviceAddress get_blas_address(device_id id) const;

    size_t get_geometry_count() const;
    bool is_backface_culled() const;

private:
    size_t updates_since_rebuild;
    size_t geometry_count;
    bool backface_culled;
    bool dynamic;
    bool compact;

    gpu_buffer transform_buffer;

    struct buffer_data
    {
        vkm<vk::AccelerationStructureKHR> blas;
        vkm<vk::Buffer> blas_buffer;
        vk::DeviceAddress blas_address;
        vkm<vk::Buffer> scratch_buffer;
        vk::DeviceAddress scratch_address;
    };
    per_device<buffer_data> buffers;
};

class top_level_acceleration_structure
{
public:
    top_level_acceleration_structure(
        device_mask dev,
        size_t capacity
    );

    gpu_buffer& get_instances_buffer();
    void rebuild(
        device_id id,
        vk::CommandBuffer cb,
        size_t instance_count,
        bool update
    );
    size_t get_updates_since_rebuild() const;
    void copy(
        device_id id,
        top_level_acceleration_structure& other,
        vk::CommandBuffer cmd
    );
    const vk::AccelerationStructureKHR* get_tlas_handle(device_id id) const;
    vk::DeviceAddress get_tlas_address(device_id id) const;

private:
    size_t updates_since_rebuild;
    size_t instance_count;
    size_t instance_capacity;
    bool require_rebuild;

    gpu_buffer instance_buffer;

    struct buffer_data
    {
        vkm<vk::AccelerationStructureKHR> tlas;
        vkm<vk::Buffer> tlas_buffer;
        vkm<vk::Buffer> scratch_buffer;
        vk::DeviceAddress tlas_address;
    };
    per_device<buffer_data> buffers;
};

}

#endif
