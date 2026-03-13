#include "voxelizer.hh"
#include "misc.hh"

namespace
{
using namespace tr;

struct clear_push_constant_buffer
{
    int clear_mips;
};

struct raster_push_constant_buffer
{
    pvec4 offset;
    pvec4 scale;
    uint instance_id;
};

struct gather_push_constant_buffer
{
    pivec4 mip_size;
    int mip_index;
};

}

namespace tr
{

voxelizer_stage::voxelizer_stage(
    device& dev,
    scene_stage& ss,
    const options& opt
):  single_device_stage(dev),
    opt(opt),
    occupancy(
        dev,
        ivec3(opt.map_resolution),
        vk::Format::eR32Uint,
        0, nullptr,
        vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eStorage),
    mipped(
        dev,
        ivec3(opt.map_resolution),
        vk::Format::eR8Unorm,
        0, nullptr,
        vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eStorage, vk::ImageLayout::eGeneral, true),
    ss(&ss),
    raster(dev),
    raster_desc(dev),
    clear(dev),
    clear_desc(dev),
    gather(dev),
    gather_desc(dev),
    stage_timer(dev, "voxelization"),
    history_counter(0)
{
    {
        raster_pipeline::pipeline_state ps;

        ps.output_size = uvec2(opt.map_resolution);
        ps.viewport = uvec4(0,0,opt.map_resolution, opt.map_resolution),
        ps.vertex_bindings = mesh::get_bindings();
        ps.vertex_attributes = {mesh::get_attributes()[0]};
        ps.src = {
            {"shader/voxelizer.vert"},
            {"shader/voxelizer.frag"},
            {"shader/voxelizer.geom"}
        };
        ps.conservative_rasterization = true;

        raster_desc.add(ps.src);
        ps.layout = {&raster_desc, &ss.get_descriptors()};

        raster.init(ps);
    }

    {
        shader_source src("shader/voxelizer_clear.comp");
        clear_desc.add(src);
        clear.init(src, {&clear_desc});
    }

    {
        shader_source src("shader/voxelizer_gather.comp");
        gather_desc.add(src);
        gather.init(src, {&gather_desc});
    }
}

texture& voxelizer_stage::get_map()
{
    return mipped;
}

void voxelizer_stage::reset()
{
    history_counter = 0;
}

void voxelizer_stage::update(uint32_t frame_index)
{
    clear_commands();

    // Record command buffer
    vk::CommandBuffer cb = begin_graphics();
    stage_timer.begin(cb, dev->id, frame_index);

    {
        clear.bind(cb);

        clear_desc.set_image(dev->id, "occupancy", {{{}, occupancy.get_image_view(dev->id), vk::ImageLayout::eGeneral}});
        clear_desc.set_image(dev->id, "mip0", {{{}, mipped.get_image_view(dev->id), vk::ImageLayout::eGeneral}});
        clear.push_descriptors(cb, clear_desc, 0);

        clear_push_constant_buffer pc;
        // Clear mips only on first frame
        pc.clear_mips = history_counter == 0 ? 1 : 0;
        clear.push_constants(cb, pc);

        uvec3 wg = uvec3(opt.map_resolution+3)/4u;
        cb.dispatch(wg.x, wg.y, wg.z);

        full_barrier(cb);
    }

    {
        raster.begin_render_pass(cb, frame_index);
        raster.bind(cb);
        raster_desc.set_image(dev->id, "occupancy", {{{}, occupancy.get_image_view(dev->id), vk::ImageLayout::eGeneral}});
        raster.push_descriptors(cb, raster_desc, 0);
        raster.set_descriptors(cb, ss->get_descriptors(), 0, 1);

        const std::vector<scene_stage::instance>& instances = ss->get_instances();
        raster_push_constant_buffer pc;
        pc.scale = pvec4(2.0f / (opt.volume.max - opt.volume.min), 0.0f);
        pc.offset = -pvec4((opt.volume.max + opt.volume.min) * 0.5f, 0.0f) * pc.scale;
        for(size_t i = 0; i < instances.size(); ++i)
        {
            const scene_stage::instance& inst = instances[i];
            const mesh* m = inst.m;
            vk::Buffer vertex_buffers[] = {m->get_vertex_buffer(dev->id)};
            vk::DeviceSize offsets[] = {0};
            cb.bindVertexBuffers(0, 1, vertex_buffers, offsets);
            cb.bindIndexBuffer(
                m->get_index_buffer(dev->id),
                0, vk::IndexType::eUint32
            );
            pc.instance_id = i;

            raster.push_constants(cb, pc);

            cb.drawIndexed(m->get_indices().size(), 1, 0, 0, 0);
        }
        raster.end_render_pass(cb);
    }

    gather.bind(cb);
    for (int i = 0; (opt.map_resolution >> i) > 0; ++i)
    {
        full_barrier(cb);

        ivec3 res = ivec3(opt.map_resolution >> i);

        gather_desc.set_image(dev->id, "src_occupancy", {{{}, occupancy.get_image_view(dev->id), vk::ImageLayout::eGeneral}});
        gather_desc.set_image(dev->id, "prev_mip", {{{}, mipped.get_mip_image_view(dev->id, max(i-1, 0)), vk::ImageLayout::eGeneral}});
        gather_desc.set_image(dev->id, "cur_mip", {{{}, mipped.get_mip_image_view(dev->id, i), vk::ImageLayout::eGeneral}});
        gather.push_descriptors(cb, gather_desc, 0);

        gather_push_constant_buffer pc;
        pc.mip_size = ivec4(res, 0);
        pc.mip_index = i;
        gather.push_constants(cb, pc);

        uvec3 wg = uvec3(res+3)/4u;
        cb.dispatch(wg.x, wg.y, wg.z);
    }

    stage_timer.end(cb, dev->id, frame_index);
    end_graphics(cb, frame_index);
    history_counter++;
}

}
