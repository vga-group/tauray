#ifndef TAURAY_MESH_SCENE_HH
#define TAURAY_MESH_SCENE_HH
#include "mesh.hh"
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
    void invalidate_acceleration_structures();

    context* ctx;
    size_t max_capacity;
    std::vector<mesh_object*> objects;

    // The meshes contain the acceleration structures themselves, so we only
    // have to track scene & command buffer validity!
    struct acceleration_structure_data
    {
        bool scene_reset_needed = true;
        vkm<vk::Buffer> pre_transformed_vertices;

        struct per_frame_data
        {
            bool command_buffers_outdated = true;
        };
        per_frame_data per_frame[MAX_FRAMES_IN_FLIGHT];
    };
    std::vector<acceleration_structure_data> acceleration_structures;
    std::vector<instance> instance_cache;
    uint64_t instance_cache_frame;
};

}

#endif
