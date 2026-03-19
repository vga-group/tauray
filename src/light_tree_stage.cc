#include "light_tree_stage.hh"
#include "misc.hh"
#include "shadow_map.hh"
#include "environment_map.hh"
#include "light.hh"
#include "gpu_buffer.hh"

#define MORTON_BITS_PER_AXIS 10

namespace
{
using namespace tr;

struct gpu_light_link
{
    uint32_t instance_id;
    uint32_t primitive_id;
};

struct gpu_light_tree_node
{ // 32 bytes
    pvec4 center_radius;
    pvec4 radius_brightness;
};

struct point_light_extraction_push_constants
{
    pvec4 bounds_min;
    pvec4 bounds_inv_step;
    uint32_t output_offset;
    uint32_t key_shift;
    uint32_t point_light_count;
};

struct tri_light_extraction_push_constants
{
    pvec4 bounds_min;
    pvec4 bounds_inv_step;
    uint32_t output_offset;
    uint32_t key_shift;
    uint32_t triangle_count;
    uint32_t instance_id;
};

struct sort_reorder_push_constants 
{
    uint32_t entry_count;
    uint32_t tree_height;
    uint32_t tree_width;
    uint32_t branch_bits;
};

struct tree_builder_push_constants 
{
    uint32_t tree_width;
    uint32_t parent_offset;
    uint32_t child_offset;
    uint32_t parent_count;
    uint32_t padded_child_count;
    uint32_t child_count;
    uint32_t type_weight_flags;
};

struct gpu_tree_params 
{
    float tree_weight;
    float envmap_weight;
    float directional_light_weight;
    uint32_t leaf_count;
    uint32_t tree_start;
    uint32_t tree_width;
    uint32_t tree_height;
    uint32_t branch_bits;
    uint32_t branch_mask;
    uint32_t layer_offsets[32];
};

unsigned light_tree_layer_size(unsigned tree_width, unsigned leaf_count, unsigned layer)
{
    unsigned width = ipow(tree_width, layer+1);
    return (leaf_count + width - 1) / width * tree_width;
}

unsigned light_tree_layer_count(unsigned tree_width, unsigned leaf_count)
{
    unsigned width = 0;
    unsigned count = 0;
    while(leaf_count > width)
    {
        count++;
        if(width == 0) width = tree_width;
        else width *= tree_width;
    }
    return count;
}

unsigned light_tree_packed_size(unsigned tree_width, unsigned leaf_count)
{
    unsigned size = 0;

    unsigned layer_count = light_tree_layer_count(tree_width, leaf_count);
    for(unsigned i = 0; i < layer_count; ++i)
    {
        unsigned layer_size = light_tree_layer_size(tree_width, leaf_count, i);
        size += layer_size;
    }
    return size;
}

// Number of bits needed to store one branch index.
unsigned light_tree_branch_bits(unsigned tree_width)
{
    return ilog2(next_power_of_two(tree_width));
}

}

namespace tr
{

light_tree_stage::light_tree_stage(device&dev, scene_stage& s, const options& opt):
    single_device_stage(dev),
    opt(opt),
    scene_data(&s),
    stage_timer(dev, "light tree"),
    tree_set(dev),
    extraction_set(dev),
    reorder_set(dev),
    tree_builder_set(dev),
    buffer_capacity(0),
    tree_buffer_size(0),
    tree_params(dev, sizeof(gpu_tree_params), vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer),
    point_light_link_extraction(dev),
    tri_light_link_extraction(dev),
    sort_reorder(dev),
    tree_builder(dev),
    link_sorter(dev)
{
    extraction_set.add("light_links", {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    extraction_set.add("leaf_data", {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    extraction_set.add("sort_order", {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    extraction_set.add("point_lights", {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});

    reorder_set.add("unsorted_light_links", {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    reorder_set.add("unsorted_leaf_data", {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    reorder_set.add("sort_order", {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    reorder_set.add("sorted_light_links", {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    reorder_set.add("sorted_leaf_data", {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    reorder_set.add("trail_data", {5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});

    tree_builder_set.add("nodes", {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    tree_builder_set.add("type_weights", {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});

    tree_set.add("light_tree_links", {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    tree_set.add("light_tree", {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    tree_set.add("light_tree_trail", {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});
    tree_set.add("light_tree_params", {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr});

    std::map<std::string, std::string> defines;
    s.get_defines(defines);

    point_light_link_extraction.init(
        {"shader/light_tree_point_light_extraction.comp", defines},
        {&extraction_set}
    );

    tri_light_link_extraction.init(
        {"shader/light_tree_tri_light_extraction.comp", defines},
        {&extraction_set, &s.get_descriptors()}
    );

    sort_reorder.init(
        {"shader/light_tree_sort_reorder.comp", defines},
        {&reorder_set}
    );

    tree_builder.init(
        {"shader/light_tree_builder.comp", defines},
        {&tree_builder_set}
    );

    // Make sure that the buffers at least exist.
    reserve_buffers(1);
}

light_tree_stage::~light_tree_stage()
{
}

descriptor_set& light_tree_stage::get_descriptors()
{
    return tree_set;
}

void light_tree_stage::add_defines(std::map<std::string, std::string>& info)
{
    info["LIGHT_TREE_WIDTH"] = std::to_string(opt.tree_width);
}

bool light_tree_stage::excludes_explicit_lights() const
{
    return opt.exclude_explicit_lights;
}

void light_tree_stage::update(uint32_t frame_index)
{
    uint32_t point_light_count = scene_data->get_point_light_count();
    if(opt.exclude_explicit_lights) point_light_count = 0;
    uint32_t tri_light_count = scene_data->get_tri_light_count();
    uint32_t total_entries = point_light_count + tri_light_count;

    reserve_buffers(total_entries);

    scene* current_scene = scene_data->get_scene();

    clear_commands();
    vk::CommandBuffer cmd = begin_compute(true);
    stage_timer.begin(cmd, dev->id, frame_index);

    unsigned layer_count = light_tree_layer_count(opt.tree_width, total_entries);

    gpu_tree_params params;

    params.tree_weight = 1.0f;

    params.envmap_weight = 0;
    params.directional_light_weight = 0;
    for(unsigned i = 0; i < layer_count; ++i)
        params.layer_offsets[i] = light_tree_layer_size(opt.tree_width, total_entries, layer_count-1-i);

    environment_map* envmap = scene_data->get_environment_map();
    if(envmap)
        params.envmap_weight = envmap->get_average_luminance();

    if(!opt.exclude_explicit_lights)
    {
        current_scene->foreach([&](transformable&, directional_light& l) {
            params.directional_light_weight += luminance(l.get_color());
        });
    }

    params.envmap_weight *= opt.envmap_weight;
    params.directional_light_weight *= opt.directional_light_weight;

    params.leaf_count = total_entries;
    params.tree_start = tree_buffer_size - light_tree_packed_size(opt.tree_width, total_entries);
    params.tree_width = opt.tree_width;
    params.tree_height = layer_count;
    params.branch_bits = light_tree_branch_bits(opt.tree_width);
    params.branch_mask = (1<<params.branch_bits)-1;
    tree_params.update(frame_index, &params);
    tree_params.upload(dev->id, frame_index, cmd);

    //==========================================================================
    // EXTRACTION
    //==========================================================================
    uint32_t extraction_offset = 0;

    // Point light extraction
    if(point_light_count > 0)
    {
        point_light_link_extraction.bind(cmd);
        point_light_link_extraction.set_descriptors(cmd, extraction_set, 0, 0);

        point_light_extraction_push_constants pc;

        pc.bounds_min = vec4(opt.light_aabb.min, 0);
        pc.bounds_inv_step = vec4(float(1 << MORTON_BITS_PER_AXIS) / (opt.light_aabb.max - opt.light_aabb.min), 0);
        pc.output_offset = extraction_offset;
        pc.key_shift = 32 - 3 * MORTON_BITS_PER_AXIS;
        pc.point_light_count = point_light_count;

        point_light_link_extraction.push_constants(cmd, pc);
        cmd.dispatch((point_light_count+255u)/256u, 1, 1);

        extraction_offset += point_light_count;
    }

    // Triangle light extraction
    if(tri_light_count > 0)
    {
        tri_light_link_extraction.bind(cmd);
        tri_light_link_extraction.set_descriptors(cmd, extraction_set, 0, 0);
        tri_light_link_extraction.set_descriptors(cmd, scene_data->get_descriptors(), 0, 1);

        tri_light_extraction_push_constants tri_pc;
        tri_pc.bounds_min = vec4(opt.light_aabb.min, 0);
        tri_pc.bounds_inv_step = vec4(float(1 << MORTON_BITS_PER_AXIS) / (opt.light_aabb.max - opt.light_aabb.min), 0);
        tri_pc.key_shift = 32 - 3 * MORTON_BITS_PER_AXIS;

        auto& instances = scene_data->get_instances();
        for(size_t i = 0; i < instances.size(); ++i)
        {
            const material* mat = instances[i].mat;
            if(mat->emission_factor == vec3(0))
                continue;

            size_t tri_count = instances[i].m->get_indices().size()/3;

            tri_pc.output_offset = extraction_offset;
            tri_pc.triangle_count = tri_count;
            tri_pc.instance_id = i;

            tri_light_link_extraction.push_constants(cmd, tri_pc);
            cmd.dispatch((tri_count+255u)/256u, 1, 1);

            extraction_offset += tri_count;
        }
    }

    assert(extraction_offset == total_entries);

    //==========================================================================
    // SORTING
    //==========================================================================
    unsigned bottom_layer_size = light_tree_layer_size(opt.tree_width, total_entries, 0);
    unsigned bottom_layer_offset = tree_buffer_size - bottom_layer_size;

    if(total_entries > 0)
    {
        vk::DescriptorBufferInfo key_info = link_sorter.sort(
            cmd,
            link_order,
            total_entries,
            min(3u*MORTON_BITS_PER_AXIS, 32u)
        );

        reorder_set.set_buffer(dev->id, "unsorted_light_links", {{unsorted_link_buffer, 0, VK_WHOLE_SIZE}});
        reorder_set.set_buffer(dev->id, "unsorted_leaf_data", {{tree_buffer, 0, VK_WHOLE_SIZE}});
        reorder_set.set_buffer(dev->id, "sort_order", {{key_info.buffer, (uint32_t)key_info.offset, VK_WHOLE_SIZE}});
        reorder_set.set_buffer(dev->id, "sorted_light_links", {{sorted_link_buffer, 0, VK_WHOLE_SIZE}});
        reorder_set.set_buffer(dev->id, "sorted_leaf_data", {{tree_buffer, bottom_layer_offset * sizeof(gpu_light_tree_node), VK_WHOLE_SIZE}});
        reorder_set.set_buffer(dev->id, "trail_data", {{trail_buffer, 0, VK_WHOLE_SIZE}});

        sort_reorder_push_constants reorder_pc;
        reorder_pc.entry_count = total_entries;
        reorder_pc.tree_height = layer_count;
        reorder_pc.tree_width = opt.tree_width;
        reorder_pc.branch_bits = light_tree_branch_bits(opt.tree_width);

        sort_reorder.bind(cmd);
        sort_reorder.push_constants(cmd, reorder_pc);
        sort_reorder.push_descriptors(cmd, reorder_set, 0);
        cmd.dispatch((total_entries+255u)/256u, 1, 1);

        vk::BufferMemoryBarrier buffer_barriers[2] = {
            vk::BufferMemoryBarrier{
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                *tree_buffer,
                0,
                VK_WHOLE_SIZE
            },
            vk::BufferMemoryBarrier{
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                *sorted_link_buffer,
                0,
                VK_WHOLE_SIZE
            }
        };
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {},
            {},
            buffer_barriers,
            {}
        );
    }

    //==========================================================================
    // TREE BUILDING
    //==========================================================================
    tree_builder.bind(cmd);
    tree_builder.set_descriptors(cmd, tree_builder_set, 0, 0);

    unsigned prev_layer_offset = bottom_layer_offset;
    unsigned prev_layer_size = bottom_layer_size;
    unsigned legitimate_child_count = total_entries;
    for(unsigned layer = 1; layer < layer_count; ++layer)
    {
        unsigned layer_size = light_tree_layer_size(opt.tree_width, total_entries, layer);
        unsigned layer_offset = prev_layer_offset - layer_size;

        // Build layer
        tree_builder_push_constants pc;
        pc.tree_width = opt.tree_width;
        pc.parent_offset = layer_offset;
        pc.child_offset = prev_layer_offset;
        pc.parent_count = layer_size;
        pc.padded_child_count = prev_layer_size;
        pc.child_count = layer == 1 ? total_entries : pc.padded_child_count;
        pc.type_weight_flags = 0;

        tree_builder.push_constants(cmd, pc);
        cmd.dispatch((layer_size+255u)/256u, 1, 1);

        prev_layer_offset = layer_offset;
        prev_layer_size = layer_size;

        legitimate_child_count = (legitimate_child_count+opt.tree_width-1)/opt.tree_width;

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, {
                {
                    vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite,
                    vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite,
                    {}, {}, *tree_buffer, 0, VK_WHOLE_SIZE
                },
            }, {}
        );
    }

    //==========================================================================
    // REFERENCE BRIGHTNESS
    //==========================================================================
    // The reference brightness is used for selecting between the light tree
    // and directional lights and envmap lights.
    // It represents the approximate brightness of all lights in the light tree.
    // It's precalculated for speed, assuming that all surfaces in the scene are
    // inside the top-most light AABB.
    //
    // Also, in case you are wondering about the dispatch size: it's only done
    // as a shader to avoid routing data through the CPU. There's zero
    // parallelism here.

    {
        tree_builder_push_constants pc;
        pc.tree_width = opt.tree_width;
        pc.parent_offset = 0; // Does not matter.
        pc.child_offset = prev_layer_offset;
        pc.parent_count = 1;
        pc.padded_child_count = min(prev_layer_size, total_entries);
        pc.child_count = pc.padded_child_count;
        pc.type_weight_flags = 1;

        tree_builder.push_constants(cmd, pc);
        cmd.dispatch(1,1,1);
    }

    stage_timer.end(cmd, dev->id, frame_index);
    end_compute(cmd, frame_index);
}

void light_tree_stage::reserve_buffers(uint32_t capacity)
{
    if(capacity <= buffer_capacity)
        return;

    uint32_t next_capacity = buffer_capacity * 3 / 2;
    if(capacity > next_capacity) buffer_capacity = capacity;
    else buffer_capacity = next_capacity;

    link_order = link_sorter.create_keyval_buffer(buffer_capacity);
    unsorted_link_buffer = create_buffer(
        *dev,
        {
            {},
            buffer_capacity * sizeof(gpu_light_link),
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::SharingMode::eExclusive
        },
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
    );
    sorted_link_buffer = create_buffer(
        *dev,
        {
            {},
            buffer_capacity * sizeof(gpu_light_link),
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::SharingMode::eExclusive
        },
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
    );
    trail_buffer = create_buffer(
        *dev,
        {
            {},
            buffer_capacity * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::SharingMode::eExclusive
        },
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
    );
    tree_buffer_size = max(
        light_tree_packed_size(opt.tree_width, buffer_capacity),
        buffer_capacity * 2
    );
    tree_buffer = create_buffer(
        *dev,
        {
            {},
            tree_buffer_size * sizeof(gpu_light_tree_node),
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::SharingMode::eExclusive
        },
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
    );

    extraction_set.reset(dev->id, 1);

    extraction_set.set_buffer(dev->id, 0, "light_links", {{unsorted_link_buffer, 0, VK_WHOLE_SIZE}});
    extraction_set.set_buffer(dev->id, 0, "leaf_data", {{tree_buffer, 0, VK_WHOLE_SIZE}});
    extraction_set.set_buffer(dev->id, 0, "sort_order", {{link_order, 0, VK_WHOLE_SIZE}});
    extraction_set.set_buffer(0, "point_lights", scene_data->get_point_lights_buffer());

    tree_builder_set.reset(dev->id, 1);
    tree_builder_set.set_buffer(dev->id, 0, "nodes", {{tree_buffer, 0, VK_WHOLE_SIZE}});
    tree_builder_set.set_buffer(0, "type_weights", tree_params);

    tree_set.reset(dev->id, 1);
    tree_set.set_buffer(dev->id, 0, "light_tree_links", {{sorted_link_buffer, 0, VK_WHOLE_SIZE}});
    tree_set.set_buffer(dev->id, 0, "light_tree", {{tree_buffer, 0, VK_WHOLE_SIZE}});
    tree_set.set_buffer(dev->id, 0, "light_tree_trail", {{trail_buffer, 0, VK_WHOLE_SIZE}});
    tree_set.set_buffer(0, "light_tree_params", tree_params);
}

}
