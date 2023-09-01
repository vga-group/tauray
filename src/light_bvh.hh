#ifndef TAURAY_LIGHT_BVH_HH
#define TAURAY_LIGHT_BVH_HH
#include "math.hh"
#include "light.hh"

// Implementation following PBRTv4 12.6.3
// The current crappy implementation just (mostly) builds the BVHs on the CPU,
// then uploads on the GPU. Hence, the builds are super slow and are not a
// representative implementation of the method for publications. The sampling
// results should be valid, though.
namespace tr
{

struct cpu_light_bounds
{
    vec3 min_bound;
    vec3 max_bound;
    vec3 primary_dir;
    float power;
    float normal_variation_angle; // theta_o
    float visibility_angle; // theta_e
    bool double_sided;

    cpu_light_bounds operator|(const cpu_light_bounds& other);
};

struct cpu_light_bvh_node
{
    cpu_light_bounds bounds;
    bool is_leaf;
    uint32_t child_or_light_index;
};

// Packed for cache reasons
struct gpu_light_bvh_node
{
    uint16_t min_bound[3]; // Quantized to BVH-global AABB (lower bound)
    uint16_t max_bound[3]; // Quantized to BVH-global AABB (upper bound)
    uint32_t primary_dir; // Octahedral encoding
    float power; // Negative marks double-sided.
    float cos_normal_variation_angle;
    float cos_visibility_angle;
    // Top bit indicates which one. 1 == light index (leaf), 0 == child.
    uint32_t child_or_light_index;
};

struct gpu_light_bvh
{
    pvec4 min_bound;
    pvec4 max_bound;
    gpu_light_bvh_node nodes[1]; // FAM
};

class gpu_buffer;
class cpu_light_bvh
{
public:
    // This is a CPU build, so it can be slow.
    void build(size_t triangle_count, const gpu_tri_light* triangles);
    void build(size_t triangle_count, const gpu_buffer& triangles);

    size_t get_gpu_bvh_size() const;
    void get_gpu_bvh_data(gpu_light_bvh* bvh);

    size_t get_gpu_bit_trail_size() const;
    void get_gpu_bit_trail_data(uint32_t* bit_trail);

private:
    vec3 min_bound;
    vec3 max_bound;
    std::vector<cpu_light_bvh_node> nodes;
    std::vector<uint32_t> bit_trail_table;

    void build_recursive(cpu_light_bvh_node* begin, size_t count, size_t bit_index, uint32_t bit_trail);
};

}

#endif
