#include "dshgi_client.hh"
#include "misc.hh"
#include <czmq.h>

namespace
{
using namespace tr;
struct blend_info
{
    float ratio;
};

struct push_constant_buffer
{
    pivec3 size;
    uint32_t index;
};

}

namespace tr
{

// Does the actual updating of the grid data.
class dshgi_client_stage: public single_device_stage
{
public:
    dshgi_client_stage(
        device& dev,
        scene_stage& ss,
        dshgi_client& client
    ):  single_device_stage(dev),
        client(&client), ss(&ss),
        stage_timer(dev, "sh_grids_from_server")
    {
    }

    ~dshgi_client_stage()
    {
        unmap_all();
    }

protected:
    void update(uint32_t frame_index) override
    {
        light_scene* cur_scene = ss->get_scene();

        if(ss->check_update(scene_stage::LIGHT, scene_state_counter))
        {
            clear_commands();
            unmap_all();

            const std::vector<sh_grid*>& grids = cur_scene->get_sh_grids();
            for(sh_grid* grid: grids)
            {
                grid_data& d = data[grid];
                d.last_update = std::chrono::steady_clock::now();
                d.size = grid->get_required_bytes();
                d.staging_buffer = create_staging_buffer(*dev, d.size);
                d.progress = 1.0f;
                d.frames_since_update = 0;
                d.mem = nullptr;
                vmaMapMemory(dev->allocator, d.staging_buffer.get_allocation(), &d.mem);
            }

            if(grids.size() > 0)
            {
                comp.reset(new compute_pipeline(
                    *dev,
                    compute_pipeline::params{
                        {"shader/sh_grid_blend.comp"}, {},
                        (uint32_t)(grids.size()*MAX_FRAMES_IN_FLIGHT)
                    }
                ));

                blend_infos = gpu_buffer(
                    *dev, sizeof(blend_info) * grids.size(),
                    vk::BufferUsageFlagBits::eUniformBuffer
                );
            }

            size_t set_index = 0;
            for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
            {
                // Record command buffer
                vk::CommandBuffer cb = begin_compute();
                stage_timer.begin(cb, dev->index, i);
                blend_infos.upload(dev->index, i, cb);

                size_t j = 0;
                for(sh_grid* grid: grids)
                {
                    grid_data& d = data.at(grid);

                    texture& new_tex = client->sh_grid_upload_textures.at(grid);
                    texture& tmp_tex = client->sh_grid_tmp_textures.at(grid);
                    texture& out_tex = client->sh_grid_blended_textures.at(grid);

                    uvec3 dim = new_tex.get_dimensions();

                    // Upload new texture
                    transition_image_layout(
                        cb,
                        new_tex.get_image(dev->index),
                        new_tex.get_format(),
                        vk::ImageLayout::eUndefined,
                        vk::ImageLayout::eTransferDstOptimal,
                        0, 1
                    );

                    cb.copyBufferToImage(
                        d.staging_buffer,
                        new_tex.get_image(dev->index),
                        vk::ImageLayout::eTransferDstOptimal,
                        vk::BufferImageCopy{
                            0, 0, 0,
                            {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                            {0,0,0},
                            {dim.x, dim.y, dim.z}
                        }
                    );

                    transition_image_layout(
                        cb,
                        new_tex.get_image(dev->index),
                        new_tex.get_format(),
                        vk::ImageLayout::eTransferDstOptimal,
                        vk::ImageLayout::eGeneral,
                        0, 1
                    );

                    transition_image_layout(
                        cb,
                        out_tex.get_image(dev->index),
                        out_tex.get_format(),
                        vk::ImageLayout::eUndefined,
                        vk::ImageLayout::eGeneral,
                        0, 1
                    );

                    // Blend with temporary texture
                    comp->update_descriptor_set({
                        {"input_sh", {{}, new_tex.get_image_view(dev->index), vk::ImageLayout::eGeneral}},
                        {"inout_sh", {{}, tmp_tex.get_image_view(dev->index), vk::ImageLayout::eGeneral}},
                        {"output_sh", {{}, out_tex.get_image_view(dev->index), vk::ImageLayout::eGeneral}},
                        {"info", {blend_infos[dev->index], j*sizeof(blend_info), sizeof(blend_info)}}
                    }, set_index);

                    comp->bind(cb, set_index);

                    push_constant_buffer control;
                    control.size = dim;
                    control.index = j;

                    comp->push_constants(cb, control);

                    uvec3 wg = (dim+3u)/4u;
                    cb.dispatch(wg.x, wg.y, wg.z);

                    transition_image_layout(
                        cb,
                        out_tex.get_image(dev->index),
                        out_tex.get_format(),
                        vk::ImageLayout::eGeneral,
                        vk::ImageLayout::eShaderReadOnlyOptimal,
                        0, 1, 0, 1, false, true
                    );

                    j++;
                    set_index++;
                }

                stage_timer.end(cb, dev->index, i);
                end_compute(cb, i);
            }
        }

        auto now = std::chrono::steady_clock::now();
        for(auto& gd: client->local_grids)
        {
            grid_data& d = data.at(&gd.grid);
            if(gd.data_updated)
            {
                d.last_duration = now - d.last_update;
                d.last_update = now;
                d.progress = 1.0f/max(d.frames_since_update, 1u);
                d.frames_since_update = 0;
                if(d.mem && gd.data.size() == d.size)
                    memcpy(d.mem, gd.data.data(), d.size);
                gd.data_updated = false;
            }
        }

        const std::vector<sh_grid*>& grids = cur_scene->get_sh_grids();
        blend_infos.map<blend_info>(frame_index, [&](blend_info* bi){
            for(size_t i = 0; i < grids.size(); ++i)
            {
                grid_data& d = data.at(grids[i]);
                auto duration_now = now - d.last_update;
                float progress =
                    std::chrono::duration_cast<std::chrono::duration<double>>(duration_now)/
                    std::chrono::duration_cast<std::chrono::duration<double>>(d.last_duration);
                if(d.frames_since_update == 0)
                {
                    progress += d.progress;
                    d.progress = 0.0f;
                }
                blend_info& info = bi[i];
                if(progress >= 0.99 || d.progress >= 0.99)
                {
                    info.ratio = 1.0f;
                    d.progress = 1.0f;
                }
                else
                {
                    info.ratio = (progress-d.progress)/(1-d.progress);
                    d.progress = progress;
                }
                d.frames_since_update++;
            }
        });
    }

private:
    void unmap_all()
    {
        for(auto& pair: data)
        {
            vmaUnmapMemory(
                dev->allocator,
                pair.second.staging_buffer.get_allocation()
            );
        }
        data.clear();
    }
    std::unique_ptr<compute_pipeline> comp;
    dshgi_client* client;
    scene_stage* ss;
    uint32_t scene_state_counter = 0;
    timer stage_timer;

    struct grid_data
    {
        size_t size;
        vkm<vk::Buffer> staging_buffer;
        std::chrono::steady_clock::time_point last_update;
        std::chrono::steady_clock::duration last_duration;
        float progress;
        uint32_t frames_since_update;
        void* mem;
    };
    gpu_buffer blend_infos;
    std::unordered_map<sh_grid*, grid_data> data;
};

dshgi_client::dshgi_client(context& ctx, scene_stage& ss, const options& opt)
:   ctx(&ctx), opt(opt), ss(&ss),
    remote_timestamp(0), new_remote_timestamp(false), exit_receiver(false),
    receiver_thread(receiver_worker, this)
{
    sh_refresher.reset(new dshgi_client_stage(ctx.get_display_device(), ss, *this));
}

dshgi_client::~dshgi_client()
{
    exit_receiver = true;
    receiver_thread.join();
}

bool dshgi_client::refresh()
{
    bool reset = false;
    std::unique_lock lk(remote_grids_mutex);

    // If the list changed, just go for a full reset.
    if(local_grids.size() != remote_grids.size())
    {
        local_grids.resize(remote_grids.size());
        for(sh_grid_data& gd: local_grids)
        {
            gd.topo_changed = true;
            gd.data_updated = true;
        }
        reset = true;
        sh_grid_upload_textures.clear();
        sh_grid_tmp_textures.clear();
        sh_grid_blended_textures.clear();
    }

    scene* cur_scene = ss->get_scene();
    cur_scene->clear_sh_grids();
    for(sh_grid_data& lg: local_grids)
        cur_scene->add(lg.grid);
    ss->set_sh_grid_textures(&sh_grid_blended_textures);

    for(size_t i = 0; i < local_grids.size(); ++i)
    {
        sh_grid_data& lg = local_grids[i];
        sh_grid_data& rg = remote_grids[i];
        lg.topo_changed |= rg.topo_changed;
        rg.topo_changed = false;
        if(lg.topo_changed)
        {
            reset = true;
            lg.grid = rg.grid;
        }
        lg.data_updated |= rg.data_updated;
        rg.data_updated = false;
        if(lg.data_updated)
        {
            lg.data.resize(rg.data.size());
            memcpy(lg.data.data(), rg.data.data(), lg.data.size());
        }

        if(lg.topo_changed)
        {
            sh_grid_upload_textures.emplace(
                &lg.grid, lg.grid.create_texture(ctx->get_display_device())
            );
            sh_grid_tmp_textures.emplace(
                &lg.grid, lg.grid.create_texture(ctx->get_display_device())
            );
            sh_grid_blended_textures.emplace(
                &lg.grid, lg.grid.create_texture(ctx->get_display_device())
            );
        }

        lg.topo_changed = false;
    }

    if(new_remote_timestamp)
    {
        // Hardcoded: if we're behind the remote animation timestamp, jump to it.
        if(cur_scene->get_total_ticks() < remote_timestamp)
        {
            cur_scene->update(remote_timestamp - cur_scene->get_total_ticks());
        }
        // If we're a full second ahead the remote timestamp, time to rewind.
        else if(cur_scene->get_total_ticks() > remote_timestamp + 1000000)
        {
            cur_scene->set_animation_time(remote_timestamp);
        }
        new_remote_timestamp = false;
    }

    return reset;
}

dependencies dshgi_client::render(dependencies deps)
{
    return sh_refresher->run(deps);
}

void dshgi_client::receiver_worker(dshgi_client* s)
{
    zsys_handler_set(nullptr);
    zsock_t* socket = zsock_new(ZMQ_SUB);
    zsock_set_subscribe(socket, "sh_grid ");
    zsock_set_subscribe(socket, "sh_grid_count ");
    zsock_set_subscribe(socket, "timestamp ");
    int err = zsock_connect(socket, "tcp://%s", s->opt.server_address.c_str());
    if(err < 0)
        throw std::runtime_error(strerror(errno));

    zpoller_t* poller = zpoller_new(socket, nullptr);

    for(;;)
    {
        while(zpoller_wait(poller, 100) == nullptr)
        {
            if(s->exit_receiver)
                break;
        }
        if(s->exit_receiver)
            break;
        zmsg_t* msg = zmsg_recv(socket);
        zframe_t* frame = zmsg_first(msg);
        if(zframe_streq(frame, "sh_grid "))
        {
            uint32_t index = 0;
            frame = zmsg_next(msg);
            memcpy(&index, zframe_data(frame), sizeof(index));
            int32_t order = 0;
            frame = zmsg_next(msg);
            memcpy(&order, zframe_data(frame), sizeof(order));
            float radius = 0;
            frame = zmsg_next(msg);
            memcpy(&radius, zframe_data(frame), sizeof(radius));
            mat4 transform = mat4(1);
            frame = zmsg_next(msg);
            memcpy(&transform, zframe_data(frame), sizeof(transform));
            puvec3 res = puvec3(1);
            frame = zmsg_next(msg);
            memcpy(&res, zframe_data(frame), sizeof(res));
            VkFormat fmt;
            frame = zmsg_next(msg);
            memcpy(&fmt, zframe_data(frame), sizeof(fmt));

            //printf("sh_grid %u %d %f %ux%ux%u\n", index, order, radius, res.x, res.y, res.z);

            frame = zmsg_next(msg);

            std::unique_lock lk(s->remote_grids_mutex);
            if(index >= s->remote_grids.size())
                s->remote_grids.resize(index+1);

            sh_grid_data& gd = s->remote_grids[index];
            if(gd.grid.get_order() != order)
            {
                gd.grid.set_order(order);
                gd.topo_changed = true;
            }
            gd.grid.set_radius(order);
            gd.grid.set_transform(transform);
            if(gd.grid.get_resolution() != uvec3(res))
            {
                gd.grid.set_resolution(res);
                gd.topo_changed = true;
            }
            gd.data_updated = true;
            gd.data.resize(zframe_size(frame));
            memcpy(gd.data.data(), zframe_data(frame), gd.data.size());
        }
        if(zframe_streq(frame, "sh_grid_count "))
        {
            uint32_t count = 0;
            frame = zmsg_next(msg);
            memcpy(&count, zframe_data(frame), sizeof(count));
            //printf("sh_grid_count %u\n", count);
            std::unique_lock lk(s->remote_grids_mutex);
            // Only scale down here!
            if(count < s->remote_grids.size())
                s->remote_grids.resize(count);
        }
        if(zframe_streq(frame, "timestamp "))
        {
            time_ticks timestamp = 0;
            frame = zmsg_next(msg);
            memcpy(&timestamp, zframe_data(frame), sizeof(timestamp));
            //printf("timestamp %lu\n", timestamp);
            std::unique_lock lk(s->remote_grids_mutex);
            s->remote_timestamp = timestamp;
            s->new_remote_timestamp = true;
        }
        zmsg_destroy(&msg);
    }
    zpoller_destroy(&poller);
    zsock_destroy(&socket);
}

}
