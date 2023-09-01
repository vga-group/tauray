#include "model.hh"

namespace
{
using namespace tr;

struct joint_data_gpu
{
    pmat4 joint_transform;
};

}

namespace tr
{

model::model(): shadow_terminator_offset(0.0f) {}
model::model(const model& other)
{
    operator=(other);
}

model::model(model&& other)
{
    operator=(std::move(other));
}

model::model(const material& mat, mesh* m)
: shadow_terminator_offset(0.0f)
{
    add_vertex_group(mat, m);
}

model& model::operator=(model&& other)
{
    groups = std::move(other.groups);
    joints = std::move(other.joints);
    joint_buffer = std::move(other.joint_buffer);
    shadow_terminator_offset = other.shadow_terminator_offset;
    return *this;
}

model& model::operator=(const model& other)
{
    groups = other.groups;
    joints = other.joints;
    joint_buffer.reset();
    shadow_terminator_offset = other.shadow_terminator_offset;

    return *this;
}

void model::add_vertex_group(const material& mat, mesh* m) { groups.push_back({mat, m}); }
void model::clear_vertex_groups() { groups.clear(); }

bool model::is_skinned() const
{
    for(const vertex_group& vg: groups)
    {
        if(vg.m->is_skinned())
            return true;
    }
    return false;
}

size_t model::group_count() const { return groups.size(); }
model::vertex_group& model::operator[](size_t i) { return groups[i]; }
const model::vertex_group& model::operator[](size_t i) const { return groups[i]; }

model::iterator model::begin() { return groups.begin(); }
model::const_iterator model::begin() const { return groups.begin(); }
model::const_iterator model::cbegin() const { return groups.cbegin(); }
model::iterator model::end() { return groups.end(); }
model::const_iterator model::end() const { return groups.end(); }
model::const_iterator model::cend() const { return groups.cend(); }

std::vector<model::joint_data>& model::get_joints()
{
    return joints;
}

const std::vector<model::joint_data>& model::get_joints() const
{
    return joints;
}

void model::init_joints_buffer(device_mask dev)
{
    if(joints.size() == 0) return;

    size_t joint_bytes = joints.size() * sizeof(joint_data_gpu);
    joint_buffer.emplace(
        dev, joint_bytes, vk::BufferUsageFlagBits::eStorageBuffer
    );
}

bool model::has_joints_buffer() const
{
    return joint_buffer.has_value();
}

const gpu_buffer& model::get_joint_buffer() const
{
    return joint_buffer.value();
}

void model::update_joints(uint32_t frame_index)
{
    if(!joint_buffer.has_value()) return;
    joint_buffer->foreach<joint_data_gpu>(
        frame_index, joints.size(),
        [&](joint_data_gpu& gpu_joint, size_t i){
            gpu_joint.joint_transform =
                joints[i].node->get_global_transform() *
                joints[i].inverse_bind_matrix;
        }
    );
}

void model::upload_joints(vk::CommandBuffer buf, device_id id, uint32_t frame_index)
{
    if(!joint_buffer.has_value()) return;
    joint_buffer->upload(id, frame_index, buf);
}

void model::set_shadow_terminator_offset(float offset)
{
    shadow_terminator_offset = offset;
}

float model::get_shadow_terminator_offset() const
{
    return shadow_terminator_offset;
}

}
