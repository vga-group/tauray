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
        const mesh* m = nullptr;
        mat4 transform = mat4(1);
        bool opaque = true;
    };

    bottom_level_acceleration_structure(
        context& ctx,
        const std::vector<entry>& entries,
        bool backface_culled,
        bool dynamic
    );

    void update_transforms(
        size_t frame_index,
        const std::vector<entry>& entries
    );

    void rebuild(
        size_t device_index,
        size_t frame_index,
        vk::CommandBuffer cb,
        const std::vector<entry>& entries,
        bool update = true
    );
    size_t get_updates_since_rebuild() const;
    vk::AccelerationStructureKHR get_blas_handle(size_t device_index) const;
    vk::DeviceAddress get_blas_address(size_t device_index) const;

    size_t get_geometry_count() const;

private:
    context* ctx;
    size_t updates_since_rebuild;
    size_t geometry_count;
    bool backface_culled;
    bool dynamic;

    // Per-device
    struct buffer_data
    {
        gpu_buffer transform_buffer;
        vkm<vk::AccelerationStructureKHR> blas;
        vkm<vk::Buffer> blas_buffer;
        vk::DeviceAddress blas_address;
        vkm<vk::Buffer> scratch_buffer;
        vk::DeviceAddress scratch_address;
    };
    std::vector<buffer_data> buffers;
};

class top_level_acceleration_structure
{
public:
    top_level_acceleration_structure(
        context& ctx,
        size_t capacity
    );

    gpu_buffer& get_instances_buffer(size_t device_index);
    void rebuild(
        size_t device_index,
        vk::CommandBuffer cb,
        size_t instance_count,
        bool update
    );
    size_t get_updates_since_rebuild() const;
    const vk::AccelerationStructureKHR* get_tlas_handle(size_t device_index) const;
    vk::DeviceAddress get_tlas_address(size_t device_index) const;

private:
    context* ctx;

    size_t updates_since_rebuild;
    size_t instance_count;
    size_t instance_capacity;
    bool require_rebuild;

    // Per-device
    struct buffer_data
    {
        vkm<vk::AccelerationStructureKHR> tlas;
        vkm<vk::Buffer> tlas_buffer;
        vkm<vk::Buffer> scratch_buffer;
        vk::DeviceAddress tlas_address;
        gpu_buffer instance_buffer;
    };
    std::vector<buffer_data> buffers;
};

}

#endif
