#ifndef TAURAY_MODEL_HH
#define TAURAY_MODEL_HH
#include "mesh.hh"
#include "material.hh"
#include <vector>

namespace tr
{

class model
{
public:
    model();
    model(const model& other);
    model(model&& other);
    model(const material& mat, mesh* m);

    model& operator=(const model& other);
    model& operator=(model&& other);

    struct vertex_group
    {
        material mat;
        mesh* m;
    };

    struct joint_data 
    {
        animated_node* node;
        mat4 inverse_bind_matrix;
    };

    void add_vertex_group(const material& mat, mesh* m);
    void clear_vertex_groups();

    bool is_skinned() const;

    size_t group_count() const;
    vertex_group& operator[](size_t i);
    const vertex_group& operator[](size_t i) const;

    using iterator = std::vector<vertex_group>::iterator;
    using const_iterator = std::vector<vertex_group>::const_iterator;

    iterator begin();
    const_iterator begin() const;
    const_iterator cbegin() const;

    iterator end();
    const_iterator end() const;
    const_iterator cend() const;

    std::vector<joint_data>& get_joints();
    const std::vector<joint_data>& get_joints() const;

    void init_joints_buffer(context& ctx);
    bool has_joints_buffer();
    const gpu_buffer& get_joint_buffer(size_t device_index) const;

    void update_joints(size_t device_index, uint32_t frame_index);
    void upload_joints(vk::CommandBuffer buf, size_t device_index, uint32_t frame_index);

private:

    std::vector<vertex_group> groups;
    std::vector<joint_data> joints;

    struct buffer_data
    {
        gpu_buffer joint_buffer;
    };
    std::vector<buffer_data> buffers;
}; 

}

#endif
