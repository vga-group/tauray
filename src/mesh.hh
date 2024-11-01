#ifndef TAURAY_MESH_HH
#define TAURAY_MESH_HH
#include "context.hh"
#include "animation.hh"
#include "material.hh"
#include "transformable.hh"
#include "gpu_buffer.hh"
#include "acceleration_structure.hh"
#include <optional>

namespace tr
{

class mesh
{
public:
    // All meshes are forced to have the same vertex attribs to avoid the need
    // for shader permutations ;) Also note strategic alignment.
    struct vertex
    {
        pvec3 pos;
        pvec3 normal;
        pvec2 uv;
        pvec4 tangent;
    };

    // Skeletal animation in Tauray works such that one mesh is the original
    // mesh, from which the animated meshes are continuously generated. Models
    // need to indicate the original mesh where possible. The joints are stored
    // in the related model (those animated mesh clones are per-model and
    // per-instance.)
    struct skin_data
    {
        puvec4 joints;
        pvec4 weights;
    };

    mesh(device_mask dev);
    mesh(
        device_mask dev,
        std::vector<vertex>&& vertices,
        std::vector<uint32_t>&& indices,
        std::vector<skin_data>&& skin = {}
    );
    // This constructor is for building animation copies of meshes.
    // They don't carry many of the buffers, as they aren't strictly necessary.
    mesh(mesh* animation_source);
    mesh(const context& other) = delete;
    mesh(mesh&& other) = default;

    // Why do we bother with some odd "ID"s? They're used to determine the
    // identity of a mesh when they are assigned into acceleration structures;
    // using pointers instead would not work, as the mesh can change without
    // notice. IDs are reassigned whenever refresh_buffers() is called.
    // The number of triangles in a mesh cannot change without the ID changing,
    // ensuring that acceleration structures with the same ID are at least
    // update-compatible.
    uint64_t get_id() const;

    std::vector<vertex>& get_vertices();
    const std::vector<vertex>& get_vertices() const;
    std::vector<uint32_t>& get_indices();
    const std::vector<uint32_t>& get_indices() const;
    std::vector<skin_data>& get_skin();
    const std::vector<skin_data>& get_skin() const;

    vk::Buffer get_vertex_buffer(device_id id) const;
    vk::Buffer get_prev_pos_buffer(device_id id) const;
    vk::Buffer get_index_buffer(device_id id) const;
    vk::Buffer get_skin_buffer(device_id id) const;

    bool is_skinned() const;
    mesh* get_animation_source() const;

    // If you modify vertices or indices after constructor call, use this to
    // reload the GPU buffer(s). If you give the command buffers, uploads are
    // recorded into them instead of temporary ones.
    void refresh_buffers();

    // Calculates new normals for existing vertices. Assumes that vertices and
    // indices are already filled out, but that normals and tangents are garbage.
    void calculate_normals();
    //
    // Calculates new tangents for existing vertices. Assumes that vertices and
    // indices are already filled out, but that tangents are garbage.
    void calculate_tangents();

    static std::vector<vk::VertexInputBindingDescription> get_bindings(bool animated = false);
    static std::vector<vk::VertexInputAttributeDescription> get_attributes(bool animated = false);

private:
    void init_buffers();

    static uint64_t id_counter;

    uint64_t id;
    std::vector<vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<skin_data> skin;
    mesh* animation_source;
    struct buffer_data
    {
        vkm<vk::Buffer> vertex_buffer;
        vkm<vk::Buffer> prev_pos_buffer; // Only allocated for animated meshes.
        vkm<vk::Buffer> index_buffer;
        vkm<vk::Buffer> skin_buffer;
    };
    per_device<buffer_data> buffers;
};

}

#endif
