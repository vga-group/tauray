#ifndef TAURAY_RT_RENDERER_HH
#define TAURAY_RT_RENDERER_HH
#ifdef WIN32
#include "windows.h"
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#endif
#include "context.hh"
#include "texture.hh"
#include "path_tracer_stage.hh"
#include "direct_stage.hh"
#include "whitted_stage.hh"
#include "raster_stage.hh"
#include "feature_stage.hh"
#include "stitch_stage.hh"
#include "scene_update_stage.hh"
#include "skinning_stage.hh"
#include "renderer.hh"
#include "post_processing_renderer.hh"
#include <variant>

namespace tr
{

class scene;

template<typename Pipeline>
class rt_renderer: public renderer
{
public:
    struct options: Pipeline::options
    {
        scene_update_stage::options scene_options = {};
        post_processing_renderer::options post_process = {};
        size_t active_viewport_count = 1;
        bool accumulate = false;
    };

    rt_renderer(context& ctx, const options& opt);
    rt_renderer(const rt_renderer& other) = delete;
    rt_renderer(rt_renderer&& other) = delete;
    ~rt_renderer();

    void set_scene(scene* s) override;
    void reset_accumulation(bool reset_sample_counter = true) override;
    void render() override;
    void set_device_workloads(const std::vector<double>& ratios) override;

private:
    void init_resources();

    context* ctx;
    options opt;
    post_processing_renderer post_processing;
    bool use_raster_gbuffer = true;
    unsigned accumulated_frames = 0;

    struct transfer_buffer
    {
        void* host_ptr = nullptr;
        vk::Buffer gpu_to_cpu;
        vk::DeviceMemory gpu_to_cpu_mem;
        vk::Buffer cpu_to_gpu;
        vk::DeviceMemory cpu_to_gpu_mem;
    };

    struct per_frame_data
    {
        // One per gbuffer entry
        std::vector<transfer_buffer> transfer_buffers;

        vkm<vk::Semaphore> gpu_to_cpu_sem_copy;

        vkm<vk::CommandBuffer> gpu_to_cpu_cb;
        vkm<vk::Semaphore> gpu_to_cpu_sem;
#ifdef WIN32
        HANDLE sem_handle;
#else
        int sem_fd;
#endif
        vkm<vk::CommandBuffer> cpu_to_gpu_cb;
        vkm<vk::Semaphore> cpu_to_gpu_sem;
    };

    timer gpu_to_cpu_timer;
    gbuffer_texture gbuffer;

    struct per_device_data
    {
        gbuffer_texture gbuffer_copy;
        timer cpu_to_gpu_timer;

        std::vector<per_frame_data> per_frame;

        std::unique_ptr<Pipeline> ray_tracer;
        std::unique_ptr<skinning_stage> skinning;
        std::unique_ptr<scene_update_stage> scene_update;

        distribution_params dist;
        dependencies last_frame_deps;
    };
    std::vector<per_device_data> per_device;
    std::unique_ptr<stitch_stage> stitch;
    std::unique_ptr<raster_stage> gbuffer_rasterizer;

    void reset_transfer_command_buffers(
        uint32_t frame_index,
        per_device_data& r,
        per_frame_data& f,
        uvec2 transfer_size,
        device_data& primary,
        device_data& secondary
    );
};

using path_tracer_renderer = rt_renderer<path_tracer_stage>;
using whitted_renderer = rt_renderer<whitted_stage>;
using feature_renderer = rt_renderer<feature_stage>;
using direct_renderer = rt_renderer<direct_stage>;

}

#endif
