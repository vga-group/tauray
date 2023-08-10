#include "dshgi_server.hh"
#include "sh_grid.hh"
#include "misc.hh"
#include "log.hh"
#include <thread>
#include <iostream>
#include <czmq.h>

namespace tr
{

class sh_grid_to_cpu_stage: public single_device_stage
{
public:
    sh_grid_to_cpu_stage(device& dev, scene_stage& ss)
    : single_device_stage(dev), ss(&ss), stage_timer(dev, "sh_grid_to_cpu")
    {
    }

    ~sh_grid_to_cpu_stage()
    {
        for(auto& pair: data)
        {
            vmaUnmapMemory(
                dev->allocator,
                pair.second.staging_buffer.get_allocation()
            );
        }
    }

    void set_renderer(sh_renderer* ren)
    {
        this->ren = ren;
    }

    void get_memory(sh_grid* sh, size_t& size, void*& mem)
    {
        grid_data& d = data.at(sh);
        size = d.size;
        mem = d.mem;
    }

private:
    void update(uint32_t) override
    {
        if(!ss->check_update(scene_stage::LIGHT, scene_state_counter))
            return;

        clear_commands();
        data.clear();

        scene* s = ss->get_scene();
        const std::vector<sh_grid*>& grids = s->get_sh_grids();

        for(sh_grid* grid: grids)
        {
            grid_data& d = data[grid];
            d.size = grid->get_required_bytes();
            d.staging_buffer = create_download_buffer(*dev, d.size);
            vmaMapMemory(dev->allocator, d.staging_buffer.get_allocation(), &d.mem);
        }

        for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            // Record command buffer
            vk::CommandBuffer cb = begin_compute();
            stage_timer.begin(cb, dev->index, i);

            for(sh_grid* grid: grids)
            {
                grid_data& d = data.at(grid);

                texture& tex = ren->get_sh_grid_texture(grid);

                uvec3 dim = tex.get_dimensions();
                transition_image_layout(
                    cb,
                    tex.get_image(dev->index),
                    tex.get_format(),
                    vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::ImageLayout::eTransferSrcOptimal,
                    0, 1, 0, 1, true
                );

                cb.copyImageToBuffer(
                    tex.get_image(dev->index),
                    vk::ImageLayout::eTransferSrcOptimal,
                    d.staging_buffer,
                    vk::BufferImageCopy{
                        0, 0, 0,
                        {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                        {0,0,0},
                        {dim.x, dim.y, dim.z}
                    }
                );
            }

            stage_timer.end(cb, dev->index, i);
            end_compute(cb, i);
        }
    }

    scene_stage* ss;
    timer stage_timer;
    uint32_t scene_state_counter = 0;
    sh_renderer* ren;

    struct grid_data
    {
        size_t size;
        vkm<vk::Buffer> staging_buffer;
        void* mem;
    };
    std::unordered_map<sh_grid*, grid_data> data;
};

dshgi_server::dshgi_server(context& ctx, const options& opt)
:   ctx(&ctx), opt(opt), exit_sender(false), subscriber_count(0),
    sender_thread(sender_worker, this)
{
    scene_update.reset(new scene_stage(ctx.get_display_device(), {}));
    sh_grid_to_cpu.reset(new sh_grid_to_cpu_stage(ctx.get_display_device(), *scene_update));

    sh.emplace(ctx, *scene_update, opt.sh);

    sender_semaphore = create_timeline_semaphore(ctx.get_display_device());
}

dshgi_server::~dshgi_server()
{
    exit_sender = true;
    frame_queue_cv.notify_one();
    sender_thread.join();
}

void dshgi_server::set_scene(scene* s)
{
    scene_update->set_scene(s);
}

void dshgi_server::render()
{
    if(subscriber_count != 0)
    {
        dependencies deps(ctx->begin_frame());
        device& d = ctx->get_display_device();

        deps = scene_update->run(deps);
        uint64_t signal_value = ctx->get_frame_counter()-1;
        if(signal_value != 0)
        {
            // Make sure we don't overwrite the SH probes while the server is
            // sending them!
            deps.add({d.index, sender_semaphore, signal_value});
        }

        deps = sh->render(deps);
        {
            std::unique_lock lk(frame_queue_mutex);
            frame_queue.emplace_back(deps);
        }
        frame_queue_cv.notify_one();
        deps = sh_grid_to_cpu->run(deps);

        ctx->end_frame(deps);
    }
    else std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void dshgi_server::sender_worker(dshgi_server* s)
{
    zsys_handler_set(nullptr);
    zsock_t* socket = zsock_new(ZMQ_XPUB);
    int rate_limit = 1000000; // 1 Gbps rate limit for now.
    zsock_set_rate(socket, rate_limit);
    // TODO: Try PGM or NORM with multiple clients, they have multicast so they
    // may be faster!
    int err = zsock_bind(socket, "tcp://*:%d", (int)s->opt.port_number);
    if(err < 0)
        throw std::runtime_error(strerror(errno));

    device& dev = s->ctx->get_display_device();
    zpoller_t* poller = zpoller_new(socket, nullptr);

    auto check_recv = [&](){
        // Let's check for subscribers every now and then.
        while(zpoller_wait(poller, 0) != nullptr)
        {
            zmsg_t* msg = zmsg_recv(socket);
            zframe_t* frame = zmsg_first(msg);
            uint8_t tag = *(uint8_t*)zframe_data(frame);
            frame = zmsg_next(msg);
            if(tag != 0) s->subscriber_count++;
            else s->subscriber_count--;
            TR_LOG("Client count: ", s->subscriber_count);
            zmsg_destroy(&msg);
        }
    };

    for(;;)
    {
        dependencies deps;
        {
            std::unique_lock<std::mutex> lk(s->frame_queue_mutex);
            check_recv();
            while(!s->frame_queue_cv.wait_for(
                lk,
                std::chrono::milliseconds(100),
                [s]{return !s->frame_queue.empty() || s->exit_sender;}
            )){
                check_recv();
            }
            if(s->exit_sender)
                break;

            deps = std::move(s->frame_queue.front());
            s->frame_queue.erase(s->frame_queue.begin());
        }
        deps.wait(dev);

        const std::vector<sh_grid*>& grids = s->cur_scene->get_sh_grids();

        { // Send animation timestamp
            zmsg_t* msg = zmsg_new();
            zmsg_addstr(msg, "timestamp ");
            time_ticks timestamp = s->cur_scene->get_total_ticks();
            zmsg_addmem(msg, &timestamp, sizeof(timestamp));
            zmsg_send(&msg, socket);
        }

        // The total number of grids is also sent. The intention here is that
        // the client doesn't actually have to know anything about the locations
        // or number of SH grids ahead-of-time.
        {
            zmsg_t* msg = zmsg_new();
            zmsg_addstr(msg, "sh_grid_count ");
            uint32_t count = grids.size();
            zmsg_addmem(msg, &count, sizeof(count));
            zmsg_send(&msg, socket);
        }

        for(size_t i = 0; i < grids.size(); ++i)
        {
            sh_grid* grid = grids[i];
            size_t size = 0;
            void* mem = nullptr;
            s->sh_grid_to_cpu->get_memory(grids[i], size, mem);

            zmsg_t* msg = zmsg_new();
            zmsg_addstr(msg, "sh_grid ");

            uint32_t index = i;
            zmsg_addmem(msg, &index, sizeof(index));

            int32_t order = grid->get_order();
            zmsg_addmem(msg, &order, sizeof(order));

            float radius = grid->get_radius();
            zmsg_addmem(msg, &radius, sizeof(radius));

            mat4 transform = grid->get_global_transform();
            zmsg_addmem(msg, &transform, sizeof(transform));

            puvec3 res = grid->get_resolution();
            zmsg_addmem(msg, &res, sizeof(res));

            VkFormat fmt = (VkFormat)s->sh->get_sh_grid_texture(grid).get_format();
            zmsg_addmem(msg, &fmt, sizeof(fmt));

            zmsg_addmem(msg, mem, size);

            zmsg_send(&msg, socket);
        }

        dev.dev.signalSemaphore({*s->sender_semaphore, deps.value(dev.index, 0)});
    }
    zpoller_destroy(&poller);
    zsock_destroy(&socket);
}

}
