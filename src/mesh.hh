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

    mesh(context& ctx);
    mesh(
        context& ctx,
        std::vector<vertex>&& vertices,
        std::vector<uint32_t>&& indices,
        std::vector<skin_data>&& skin = {}
    );
    // This constructor is for building animation copies of meshes.
    // They don't carry many of the buffers, as they aren't strictly necessary.
    mesh(mesh* animation_source);
    mesh(const context& other) = delete;
    mesh(mesh&& other) = default;

    std::vector<vertex>& get_vertices();
    const std::vector<vertex>& get_vertices() const;
    std::vector<uint32_t>& get_indices();
    const std::vector<uint32_t>& get_indices() const;
    std::vector<skin_data>& get_skin();
    const std::vector<skin_data>& get_skin() const;

    vk::Buffer get_vertex_buffer(size_t device_index) const;
    vk::Buffer get_index_buffer(size_t device_index) const;
    vk::Buffer get_skin_buffer(size_t device_index) const;

    vk::AccelerationStructureKHR get_blas(size_t device_index) const;
    vk::DeviceAddress get_blas_address(size_t device_index) const;

    enum blas_update_strategy
    {
        UPDATE_REBUILD, // Slow, but best acceleration structure quality
        UPDATE_FROM_ANIMATION_SOURCE, // Fast, bad quality but won't get progressively worse
        UPDATE_FROM_PREVIOUS // Fast, OK quality but eventually gets worse if applied repeatedly
    };

    // You only need to call this when you have modified the vertex buffer.
    void update_blas(
        vk::CommandBuffer cb,
        size_t device_index,
        blas_update_strategy strat = UPDATE_FROM_ANIMATION_SOURCE
    ) const;

    bool is_skinned() const;
    mesh* get_animation_source() const;

    // Setting certain meshes opaque can make rendering them faster, but
    // alpha effects will not work on them. The change will not take effect
    // until upload() is called.
    void set_opaque(bool opaque);
    bool get_opaque() const;

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

    static std::vector<vk::VertexInputBindingDescription> get_bindings();
    static std::vector<vk::VertexInputAttributeDescription> get_attributes();

private:
    void init_buffers();

    context* ctx;
    std::vector<vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<skin_data> skin;
    bool opaque;
    mesh* animation_source;
    // Per-device
    struct buffer_data
    {
        vkm<vk::Buffer> vertex_buffer;
        vkm<vk::Buffer> index_buffer;
        vkm<vk::Buffer> skin_buffer;
    };
    std::vector<buffer_data> buffers;
    mutable std::optional<bottom_level_acceleration_structure> blas;
};

}

#endif
