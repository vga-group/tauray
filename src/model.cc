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

model::model() {}
model::model(const model& other)
{
    operator=(other);
}

model::model(model&& other)
{
    operator=(std::move(other));
}

model::model(const material& mat, mesh* m)
{
    add_vertex_group(mat, m);
}

model& model::operator=(model&& other)
{
    groups = std::move(other.groups);
    joints = std::move(other.joints);
    buffers = std::move(other.buffers);
    return *this;
}

model& model::operator=(const model& other)
{
    groups = other.groups;
    joints = other.joints;
    buffers.clear();

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

void model::init_joints_buffer(context& ctx)
{
    if(joints.size() == 0) return;

    std::vector<device_data>& devices = ctx.get_devices();
    buffers.resize(devices.size());

    size_t joint_bytes = joints.size() * sizeof(joint_data_gpu);

    for(size_t i = 0; i < devices.size(); ++i)
    {
        buffers[i].joint_buffer = gpu_buffer(
            devices[i], joint_bytes, vk::BufferUsageFlagBits::eStorageBuffer
        );
    }
}

bool model::has_joints_buffer()
{
    return buffers.size() != 0;
}

const gpu_buffer& model::get_joint_buffer(size_t device_index) const
{
    return buffers[device_index].joint_buffer;
}

void model::update_joints(size_t device_index, uint32_t frame_index)
{
    if(buffers.size() == 0) return;
    buffers[device_index].joint_buffer.foreach<joint_data_gpu>(
        frame_index, joints.size(),
        [&](joint_data_gpu& gpu_joint, size_t i){
            gpu_joint.joint_transform =
                joints[i].node->get_global_transform() *
                joints[i].inverse_bind_matrix;
        }
    );
}

void model::upload_joints(vk::CommandBuffer buf, size_t device_index, uint32_t frame_index)
{
    if(buffers.size() == 0) return;
    buffers[device_index].joint_buffer.upload(frame_index, buf);
}

}
