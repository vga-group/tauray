#ifndef TAURAY_LIGHT_TREE_STAGE_HH
#define TAURAY_LIGHT_TREE_STAGE_HH

#include "scene_stage.hh"
#include "radix_sort.hh"

namespace tr
{

class light_tree_stage: public single_device_stage
{
public:
    struct options
    {
        // 3 is optimal due to bandwidth concerns (disregarding cache
        // behaviour, though!):
        // Bandwidth in traversal from root to leaf is proportional to
        // N / log(N). The minimum of this function is at 2.718. With integers,
        // 3 is the minimum.
        //
        // However, wider trees can bring cache benefits and more precision due
        // to each layer of hierarchy bringing in more approximation.
        // I can't think of a benefit to binary trees, though.
        uint32_t tree_width = 3;

        // If you want to lessen the amount of envmap samples taken, you can
        // lower this. That can be a good idea when the envmap is mostly or
        // fully occluded in most of the scene.
        float envmap_weight = 1.0f;

        // If you want to lessen the amount of directional light samples taken,
        // you can lower this. That can be a good idea when directional lights
        // are mostly or fully occluded in most of the scene.
        float directional_light_weight = 1.0f;

        // If exclude_explicit_lights == true, point, spot and directional 
        // lights will not be included in light sampling.
        bool exclude_explicit_lights = false;

        // AABB which contains all light sources.
        aabb light_aabb;
    };

    light_tree_stage(device& dev, scene_stage& s, const options& opt);
    light_tree_stage(light_tree_stage&&) = delete;
    ~light_tree_stage();

    descriptor_set& get_descriptors();

    void add_defines(std::map<std::string, std::string>& info);

    bool excludes_explicit_lights() const;

protected:
    void update(uint32_t frame_index) override;
    void reserve_buffers(uint32_t capacity);

private:
    options opt;
    scene_stage* scene_data;
    timer stage_timer;

    descriptor_set tree_set;
    descriptor_set extraction_set;
    push_descriptor_set reorder_set;
    descriptor_set tree_builder_set;

    uint32_t buffer_capacity;
    uint32_t tree_buffer_size;
    vkm<vk::Buffer> link_order;
    vkm<vk::Buffer> unsorted_link_buffer;
    vkm<vk::Buffer> sorted_link_buffer;
    vkm<vk::Buffer> trail_buffer;
    vkm<vk::Buffer> tree_buffer;
    gpu_buffer tree_params;

    compute_pipeline point_light_link_extraction;
    compute_pipeline tri_light_link_extraction;
    compute_pipeline sort_reorder;
    compute_pipeline tree_builder;

    radix_sort link_sorter;
};

}

#endif
