#include "light_bvh.hh"
#include "mesh.hh"
#include "misc.hh"
#include "log.hh"
#include <glm/gtx/rotate_vector.hpp>
#include <cstring>

namespace
{
using namespace tr;

void cone_union(
    vec3 a_dir,
    float a_angle,
    vec3 b_dir,
    float b_angle,
    vec3& union_dir,
    float& union_angle
){
    float angle_between = acos(dot(a_dir, b_dir));

    if(min(angle_between + b_angle, float(M_PI)) <= a_angle)
    {
        union_dir = a_dir;
        union_angle = a_angle;
    }
    else if(min(angle_between + a_angle, float(M_PI)) <= b_angle)
    {
        union_dir = b_dir;
        union_angle = b_angle;
    }

    union_angle = min((a_angle + b_angle + angle_between) * 0.5f, float(M_PI));

    vec3 n = cross(a_dir, b_dir);
    if(dot(n, n) < 1e-6)
    {
        union_angle = M_PI;
        union_dir = normalize(a_dir+b_dir);
        return;
    }

    union_dir = rotate(a_dir, union_angle - a_angle, normalize(n));
}

float evaluate_cost(
    cpu_light_bounds& lb,
    aabb bounds,
    size_t axis
){
    float theta_w = min(lb.normal_variation_angle + lb.visibility_angle, float(M_PI));
    float sin_normal_variation_angle = sin(lb.normal_variation_angle);
    float cos_normal_variation_angle = cos(lb.normal_variation_angle);
    float m_omega = 2.0f * M_PI * (1 - cos_normal_variation_angle) +
        M_PI / 2.0f * (2.0f * theta_w * sin_normal_variation_angle
        - cos(lb.normal_variation_angle - 2 * theta_w)
        - 2.0f * lb.normal_variation_angle * sin_normal_variation_angle
        + cos_normal_variation_angle);

    vec3 size = bounds.max - bounds.min;
    vec3 lb_size = lb.max_bound - lb.min_bound;
    float Kr = vecmax(size)/size[axis];
    float surface_area = 2 * lb_size.x * lb_size.y + 2 * lb_size.x * lb_size.z + 2 * lb_size.y * lb_size.z;

    return lb.power * Kr * m_omega * surface_area;
}

}

namespace tr
{

cpu_light_bounds cpu_light_bounds::operator|(const cpu_light_bounds& other)
{
    if(power == 0) return other;
    else if(other.power == 0) return *this;

    cpu_light_bounds ret;

    ret.min_bound = min(min_bound, other.min_bound);
    ret.max_bound = max(max_bound, other.max_bound);
    ret.power = other.power + power;
    ::cone_union(primary_dir, normal_variation_angle, other.primary_dir, other.normal_variation_angle, ret.primary_dir, ret.normal_variation_angle);
    ret.visibility_angle = max(visibility_angle, other.visibility_angle);
    ret.double_sided = double_sided | other.double_sided;

    return ret;
}

void cpu_light_bvh::build(size_t triangle_count, const gpu_tri_light* triangles)
{
    nodes.clear();
    nodes.reserve(triangle_count*2);

    std::vector<cpu_light_bvh_node> leaves;
    leaves.reserve(triangle_count);
    for(size_t i = 0; i < triangle_count; ++i)
    {
        cpu_light_bvh_node leaf;
        gpu_tri_light tl = triangles[i];

        leaf.bounds.min_bound = min(min(tl.pos[0], tl.pos[1]), tl.pos[2]);
        leaf.bounds.max_bound = max(max(tl.pos[0], tl.pos[1]), tl.pos[2]);
        leaf.bounds.primary_dir = normalize(cross(tl.pos[1]-tl.pos[0], tl.pos[2]-tl.pos[0]));
        leaf.bounds.power = fabs(tl.power_estimate);
        leaf.bounds.normal_variation_angle = 0;
        leaf.bounds.visibility_angle = M_PI/2;
        leaf.bounds.double_sided = tl.power_estimate < 0;
        leaf.is_leaf = true;
        leaf.child_or_light_index = i;
        leaves.push_back(leaf);
    }

    bit_trail_table.resize(triangle_count);
    build_recursive(leaves.data(), leaves.size(), 0, 0);
}

#define SPLIT_BUCKET_COUNT 12
void cpu_light_bvh::build_recursive(cpu_light_bvh_node* begin, size_t count, size_t bit_index, uint32_t bit_trail)
{
    assert(bit_index < 32);
    if(count == 0) return; // WTF?
    if(count == 1)
    { // Leaf node
        nodes.push_back(*begin);
        bit_trail_table[begin->child_or_light_index] = bit_trail;
        return;
    }
    if(count == 2)
    {
        nodes.push_back(begin[0]);
        bit_trail_table[begin[0].child_or_light_index] = bit_trail;
        nodes.push_back(begin[1]);
        bit_trail_table[begin[1].child_or_light_index] = bit_trail|(1<<bit_index);
        return;
    }

    aabb bounds = {begin[0].bounds.min_bound, begin[0].bounds.max_bound};
    for(size_t i = 1; i < count; ++i)
    {
        cpu_light_bvh_node& n  = begin[i];
        // TODO: How about using centroids instead? May not matter in real use
        // cases.
        bounds.min = min(bounds.min, n.bounds.min_bound);
        bounds.max = max(bounds.max, n.bounds.max_bound);
    }

    vec3 size = bounds.max - bounds.min;

    struct split_bucket
    {
        cpu_light_bounds bounds = {
            vec3(0), vec3(0), vec3(0), 0,0,0,false
        };
        int count = 0;
    };
    split_bucket buckets[3][SPLIT_BUCKET_COUNT];
    vec3 inv_size = vec3(SPLIT_BUCKET_COUNT) / size;

    for(size_t i = 0; i < count; ++i)
    {
        cpu_light_bvh_node& n  = begin[i];
        vec3 centroid = (n.bounds.min_bound + n.bounds.max_bound)*0.5f;
        ivec3 index = (centroid - bounds.min) * inv_size;

        for(size_t axis = 0; axis < 3; ++axis)
        {
            split_bucket& b = buckets[axis][index[axis]];
            if(b.count == 0)
            {
                b.count = 1;
                b.bounds = n.bounds;
            }
            else
            {
                b.count++;
                b.bounds = b.bounds | n.bounds;
            }
        }
    }

    split_bucket bucket_ascending[3][SPLIT_BUCKET_COUNT];
    split_bucket bucket_descending[3][SPLIT_BUCKET_COUNT];

    for(size_t axis = 0; axis < 3; ++axis)
    {
        bucket_ascending[axis][0] = buckets[axis][0];
        bucket_descending[axis][SPLIT_BUCKET_COUNT-1] = buckets[axis][SPLIT_BUCKET_COUNT-1];
        for(size_t i = 1; i < SPLIT_BUCKET_COUNT; ++i)
        {
            bucket_ascending[axis][i].bounds = buckets[axis][i].bounds | bucket_ascending[axis][i-1].bounds;
            bucket_ascending[axis][i].count = buckets[axis][i].count + bucket_ascending[axis][i-1].count;
            size_t j = SPLIT_BUCKET_COUNT-1-i;
            bucket_descending[axis][j].bounds = buckets[axis][j].bounds | bucket_descending[axis][j+1].bounds;
            bucket_descending[axis][j].count = buckets[axis][j].count + bucket_descending[axis][j+1].count;
        }
    }

    float min_cost = FLT_MAX;
    int split_axis = 0;
    int split_bucket = 0;
    for(size_t axis = 0; axis < 3; ++axis)
    {
        for(size_t i = 0; i < SPLIT_BUCKET_COUNT-1; ++i)
        {
            float ascending_cost = evaluate_cost(
                bucket_ascending[axis][i].bounds,
                bounds,
                axis
            );
            float descending_cost = evaluate_cost(
                bucket_descending[axis][i+1].bounds,
                bounds,
                axis
            );
            float cost = ascending_cost + descending_cost;
            if(cost < min_cost)
            {
                min_cost = cost;
                split_axis = axis;
                split_bucket = i;
            }
        }
    }

    float split = float(split_bucket+1) * size[split_axis]/SPLIT_BUCKET_COUNT + bounds.min[split_axis];

    size_t split_count = std::partition(
        begin, begin + count,
        [&](const cpu_light_bvh_node& n) {
            return (n.bounds.min_bound[split_axis] + n.bounds.max_bound[split_axis])*0.5f < split;
        }
    ) - begin;

    cpu_light_bvh_node node;
    node.bounds = bucket_ascending[split_axis][split_bucket].bounds | bucket_descending[split_axis][split_bucket+1].bounds;
    node.is_leaf = false;
    size_t index = nodes.size();
    nodes.push_back(node);

    // Avoid the corner cases with empty splits.
    if(split_count == 0) split_count = 1;
    else if(split_count == count) split_count = count-1;

    build_recursive(begin, split_count, bit_index+1, bit_trail);
    size_t second_child_index = nodes.size();
    build_recursive(begin+split_count, count-split_count, bit_index+1, bit_trail|(1<<bit_index));

    nodes[index].child_or_light_index = second_child_index;
}

void cpu_light_bvh::build(size_t triangle_count, const gpu_buffer& triangles)
{
    device& dev = *triangles.get_device_mask().begin();

    dev.logical.waitIdle();

    vkm<vk::Buffer> dlbuf = create_download_buffer(dev, triangles.get_size());

    vk::CommandBuffer cmd = begin_command_buffer(dev);

    vk::BufferCopy copy = {0, 0, triangles.get_size()};
    cmd.copyBuffer(triangles[dev.id], *dlbuf, 1, &copy);

    end_command_buffer(dev, cmd);

    const gpu_tri_light* mapped = nullptr;
    vmaMapMemory(dev.allocator, dlbuf.get_allocation(), (void**)&mapped);

    build(triangle_count, mapped);

    vmaUnmapMemory(dev.allocator, dlbuf.get_allocation());
}

size_t cpu_light_bvh::get_gpu_bvh_size() const
{
    return sizeof(gpu_light_bvh) + sizeof(gpu_light_bvh_node) * nodes.size();
}

void cpu_light_bvh::get_gpu_bvh_data(gpu_light_bvh* bvh)
{
    vec3 min_bound = vec3(0);
    vec3 max_bound = vec3(0);
    if(nodes.size() != 0)
    {
        min_bound = nodes[0].bounds.min_bound;
        max_bound = nodes[0].bounds.max_bound;
    }
    bvh->min_bound = pvec4(min_bound, 0.0f);
    bvh->max_bound = pvec4(max_bound, 0.0f);

    for(size_t i = 0; i < nodes.size(); ++i)
    {
        gpu_light_bvh_node gpu_node;

        for(size_t j = 0; j < 3; ++j)
        {
            uint32_t qmin_bound = floor(clamp(65535 * (nodes[i].bounds.min_bound[j]-min_bound[j])/(max_bound[j]-min_bound[j]), 0.0f, 65535.0f));
            uint32_t qmax_bound = ceil(clamp(65535 * (nodes[i].bounds.max_bound[j]-min_bound[j])/(max_bound[j]-min_bound[j]), 0.0f, 65535.0f));
            gpu_node.bounds[j] = qmin_bound|(qmax_bound<<16);
        }
        gpu_node.primary_dir = packSnorm2x16(octahedral_encode(nodes[i].bounds.primary_dir));
        gpu_node.power = nodes[i].bounds.power;
        if(nodes[i].bounds.double_sided)
            gpu_node.power = -gpu_node.power;
        gpu_node.cos_normal_variation_angle = cos(nodes[i].bounds.normal_variation_angle);
        gpu_node.cos_visibility_angle = cos(nodes[i].bounds.visibility_angle);
        gpu_node.child_or_light_index = (nodes[i].is_leaf ? 0x80000000 : 0) | nodes[i].child_or_light_index;

        memcpy(bvh->nodes+i, &gpu_node, sizeof(gpu_light_bvh_node));
    }
}

size_t cpu_light_bvh::get_gpu_bit_trail_size() const
{
    return bit_trail_table.size() * sizeof(uint32_t);
}

void cpu_light_bvh::get_gpu_bit_trail_data(uint32_t* bit_trail)
{
    memcpy(bit_trail, bit_trail_table.data(), get_gpu_bit_trail_size());
}

}
