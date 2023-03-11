#ifndef TAURAY_MESH_SCENE_HH
#define TAURAY_MESH_SCENE_HH
#include "mesh.hh"
#include "acceleration_structure.hh"
#include "mesh_object.hh"
#include "timer.hh"
#include <unordered_map>

namespace tr
{

class mesh_scene
{
public:
    mesh_scene(context& ctx, size_t max_capacity = 1024);
    mesh_scene(const mesh_scene& s) = delete;
    mesh_scene(mesh_scene&& s) = delete;
    ~mesh_scene();

    void add(mesh_object& o);
    void remove(mesh_object& o);
    void clear_mesh_objects();
    const std::vector<mesh_object*>& get_mesh_objects() const;
    size_t get_instance_count() const;
    // This can be very slow!
    size_t get_sampler_count() const;

    struct instance
    {
        mat4 transform;
        mat4 prev_transform;
        mat4 normal_transform;
        const material* mat;
        const mesh* m;
        const mesh_object* o;
        uint64_t last_refresh_frame;
    };

    // Refresh will only occur once per frame, this will skip refreshing if it's
    // already been done on this frame! Unless force is true, in which case it
    // will update anyway.
    void refresh_instance_cache(bool force = false);
    const std::vector<instance>& get_instances() const;

    bool reserve_pre_transformed_vertices(size_t device_index, size_t max_vertex_count);
    void clear_pre_transformed_vertices(size_t device_index);
    vk::Buffer get_pre_transformed_vertices(size_t device_index);

    std::vector<vk::DescriptorBufferInfo> get_vertex_buffer_bindings(
        size_t device_index
    ) const;
    std::vector<vk::DescriptorBufferInfo> get_index_buffer_bindings(
        size_t device_index
    ) const;

    void refresh_dynamic_acceleration_structures(
        size_t device_index,
        size_t frame_index,
        vk::CommandBuffer cmd
    );

protected:
    template<typename F>
    void visit_animated(F&& f) const
    {
        for(mesh_object* o: objects)
        {
            if(!o->is_static())
                f(o);
        }
    }

    size_t get_max_capacity() const;

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

private:
    void invalidate_tlas();
    void ensure_blas();

    context* ctx;
    size_t max_capacity;
    std::vector<mesh_object*> objects;

    // We have to track scene & command buffer validity! So that's what this
    // struct is for.
    struct as_update_data
    {
        bool tlas_reset_needed = true;
        uint32_t pre_transformed_vertex_count = 0;
        vkm<vk::Buffer> pre_transformed_vertices;

        struct per_frame_data
        {
            bool command_buffers_outdated = true;
        };
        per_frame_data per_frame[MAX_FRAMES_IN_FLIGHT];
    };

    // For acceleration structures, instances are grouped by which ones go into
    // the same BLAS. If two groups share the same ID, they will have the same
    // acceleration structure as well, but are inserted as separate TLAS
    // instances still.
    struct instance_group
    {
        uint64_t id = 0;
        size_t size = 0;
        bool static_mesh = false;
        bool static_transformable = false;
    };

    std::vector<as_update_data> as_update;
    std::unordered_map<uint64_t, bottom_level_acceleration_structure> blas_cache;
    std::vector<instance> instance_cache;
    std::vector<instance_group> group_cache;
    uint64_t instance_cache_frame;
};

}

#endif
