#include "rt_renderer.hh"
#include "scene.hh"
#include "misc.hh"
#ifdef WIN32
#include <vulkan/vulkan_win32.h>
#endif

namespace
{
using namespace tr;

template<typename Opt>
post_processing_renderer::options get_pp_opt(
    const Opt& opt
){
    post_processing_renderer::options pp_opt = opt.post_process;
    pp_opt.active_viewport_count = opt.active_viewport_count;
    return pp_opt;
}

}

namespace tr
{

template<typename Pipeline>
rt_renderer<Pipeline>::rt_renderer(context& ctx, const options& opt)
:   ctx(&ctx), opt(opt),
    post_processing(ctx.get_display_device(), ctx.get_size(), get_pp_opt(opt))
{
    this->opt.distribution.size = ctx.get_size();

    if(
        (opt.projection == camera::PERSPECTIVE ||
         opt.projection == camera::ORTHOGRAPHIC)
    ) use_raster_gbuffer = true;

    std::vector<device_data>& devices = ctx.get_devices();
    per_device.resize(devices.size());
    for(size_t i = 0; i < per_device.size(); ++i)
    {
        init_device_resources(i);
    }

    init_primary_device_resources();
}

template<typename Pipeline>
rt_renderer<Pipeline>::~rt_renderer()
{
    std::vector<device_data>& devices = ctx->get_devices();
    device_data& display_device = ctx->get_display_device();
    ctx->sync();

    // Ensure each pipeline is deleted before the assets they may use
    stitch.reset();

    for(size_t i = 0; i < per_device.size(); ++i)
    {
        device_data& dev = devices[i];
        per_device_data& d = per_device[i];
        d.ray_tracer.reset();
        for(auto& f: d.per_frame)
        {
            for(auto& tb: f.transfer_buffers)
            {
                if(!tb.host_ptr) continue;
                release_host_buffer(tb.host_ptr);
                destroy_host_allocated_buffer(
                    dev, tb.gpu_to_cpu, tb.gpu_to_cpu_mem
                );
                destroy_host_allocated_buffer(
                    display_device, tb.cpu_to_gpu, tb.cpu_to_gpu_mem
                );
            }
        }
    }
    ctx->sync();
}

template<typename Pipeline>
void rt_renderer<Pipeline>::set_scene(scene* s)
{
    opt.projection = s->get_camera(0)->get_projection_type();
    if(s)
    {
        for(size_t i = 0; i < per_device.size(); ++i)
        {
            per_device[i].skinning->set_scene(s);
            per_device[i].scene_update->set_scene(s);
            if(per_device[i].ray_tracer)
                per_device[i].ray_tracer->set_scene(s);
        }

        if(gbuffer_rasterizer)
            gbuffer_rasterizer->set_scene(s);

        post_processing.set_scene(s);
    }
}

template<typename Pipeline>
void rt_renderer<Pipeline>::reset_accumulation(bool reset_sample_counter)
{
    for(size_t i = 0; i < per_device.size(); ++i)
    {
        if(per_device[i].ray_tracer)
        {
            per_device[i].ray_tracer->reset_accumulated_samples();
            if(reset_sample_counter)
                per_device[i].ray_tracer->reset_sample_counter();
        }
    }
    accumulated_frames = 0;
    if(stitch)
        stitch->set_blend_ratio(1.0f);
}

template<typename Pipeline>
void rt_renderer<Pipeline>::render()
{
    dependencies display_deps(ctx->begin_frame());
    uint32_t swapchain_index, frame_index;
    ctx->get_indices(swapchain_index, frame_index);
    uint64_t frame_counter = ctx->get_frame_counter();

    device_data& display_device = ctx->get_display_device();
    std::vector<device_data>& devices = ctx->get_devices();

    for(size_t i = 0; i < devices.size(); ++i)
    {
        dependencies device_deps = per_device[i].skinning->run({});
        device_data& dev = devices[i];
        for(int sample = 0; sample < (opt.accumulate ? opt.samples_per_pixel : 1); ++sample)
        {
            device_deps = per_device[i].scene_update->run(device_deps);
            if(i == ctx->get_display_device().index && sample == 0)
                device_deps.concat(post_processing.get_gbuffer_write_dependencies());

            device_deps = per_device[i].ray_tracer->run(device_deps);
        }

        if(i != display_device.index)
        {
            per_frame_data& f = per_device[i].per_frame[frame_index];
            vk::TimelineSemaphoreSubmitInfo timeline_info = device_deps.get_timeline_info();
            vk::SubmitInfo submit_info = device_deps.get_submit_info(timeline_info);
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = f.gpu_to_cpu_cb.get();
            submit_info.signalSemaphoreCount = 1;
            submit_info.pSignalSemaphores = f.gpu_to_cpu_sem.get();

            vk::PipelineStageFlags wait_stage =
                vk::PipelineStageFlagBits::eTopOfPipe;
            dev.graphics_queue.submit(submit_info, {});

            timeline_info.waitSemaphoreValueCount = 0;
            timeline_info.pWaitSemaphoreValues = nullptr;
            timeline_info.signalSemaphoreValueCount = 1;
            timeline_info.pSignalSemaphoreValues = &frame_counter;
            submit_info.pWaitDstStageMask = &wait_stage;
            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = f.gpu_to_cpu_sem_copy.get();
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = f.cpu_to_gpu_cb.get();
            submit_info.signalSemaphoreCount = 1;
            submit_info.pSignalSemaphores = f.cpu_to_gpu_sem.get();

            display_device.graphics_queue.submit(submit_info, {});
            display_deps.add({f.cpu_to_gpu_sem, frame_counter});
        }
        else
        {
            if(gbuffer_rasterizer)
                device_deps = gbuffer_rasterizer->run(device_deps);
            display_deps.concat(device_deps);
        }
    }

    display_deps.concat(post_processing.get_gbuffer_write_dependencies());

    if(stitch)
    {
        stitch->refresh_params();
        display_deps = stitch->run(display_deps);
        // Reset temporary blending from stitching
        stitch->set_blend_ratio(1.0f);
    }

    display_deps = post_processing.render(display_deps);
    ctx->end_frame(display_deps);
    accumulated_frames++;
}

template<typename Pipeline>
void rt_renderer<Pipeline>::set_device_workloads(const std::vector<double>& ratios)
{
    assert(ratios.size() == per_device.size());
    if(
        opt.distribution.strategy == DISTRIBUTION_SCANLINE ||
        opt.distribution.strategy == DISTRIBUTION_DUPLICATE
    ) return;

    device_data& display_device = ctx->get_display_device();

    double cumulative = 0;
    for(size_t i = 0; i < per_device.size(); ++i)
    {
        per_device_data& r = per_device[i];

        double ratio = clamp(ratios[i], 0.0, 1.0 - cumulative);
        r.dist = get_device_distribution_params(
            ctx->get_size(),
            opt.distribution.strategy,
            cumulative,
            ratio,
            i,
            ctx->get_devices().size(),
            i == ctx->get_display_device().index
        );
        cumulative += ratio;
        per_device[i].ray_tracer->reset_distribution_params(r.dist);
        // Only the primary device renders in-place, so it's the only device
        // that can actually accumulate samples normally when workload ratio
        // changes.
        if(i != display_device.index)
            per_device[i].ray_tracer->reset_accumulated_samples();
        uvec2 target_size = get_distribution_target_size(r.dist);
        for(size_t j = 0; j < MAX_FRAMES_IN_FLIGHT; ++j)
        {
            reset_transfer_command_buffers(
                j, per_device[i], per_device[i].per_frame[j],
                target_size, ctx->get_display_device(), ctx->get_devices()[i]
            );
        }
    }

    std::vector<distribution_params> dist;
    for(per_device_data& r: per_device)
        dist.push_back(r.dist);

    // Temporarily blend non-primary GPU accumulation from stitching stage
    // instead.
    if(opt.accumulate)
        stitch->set_blend_ratio(1.0f/(accumulated_frames+1));
    stitch->set_distribution_params(dist);
}

template<typename Pipeline>
void rt_renderer<Pipeline>::init_device_resources(size_t device_index)
{
    device_data& d = ctx->get_devices()[device_index];
    bool is_display_device = d.index == ctx->get_display_device().index;
    double even_workload_ratio = 1.0/ctx->get_devices().size();

    per_device_data& r = per_device[device_index];
    r.per_frame.resize(MAX_FRAMES_IN_FLIGHT);
    r.skinning.reset(new skinning_stage(d, opt.max_meshes));
    r.scene_update.reset(new scene_update_stage(d, opt.scene_options));
    r.dist = get_device_distribution_params(
        ctx->get_size(),
        opt.distribution.strategy,
        even_workload_ratio * device_index,
        even_workload_ratio,
        device_index,
        ctx->get_devices().size(),
        is_display_device
    );

    typename Pipeline::options rt_opt = opt;
    rt_opt.distribution = r.dist;
    rt_opt.active_viewport_count = opt.active_viewport_count;
    uvec2 max_target_size = get_distribution_target_max_size(rt_opt.distribution);
    uvec2 target_size = get_distribution_target_size(rt_opt.distribution);

    unsigned color_format_size = sizeof(uint16_t)*4;
    gbuffer_spec spec, copy_spec;
    spec.color_present = true;
    spec.color_format = vk::Format::eR16G16B16A16Sfloat;
    if constexpr(std::is_same_v<Pipeline, path_tracer_stage>)
    {
        color_format_size = sizeof(uint32_t)*4;
        spec.color_format = vk::Format::eR32G32B32A32Sfloat;
        rt_opt.samples_per_pixel = opt.accumulate ? 1 : opt.samples_per_pixel;
    }
    post_processing.set_gbuffer_spec(spec);

    // Disable raster G-Buffer when nothing rasterizable is needed.
    if(
        spec.present_count()
        - spec.color_present
        - spec.direct_present
        - spec.diffuse_present == 0
    ) use_raster_gbuffer = false;

    vk::ImageUsageFlags img_usage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc;
    if(use_raster_gbuffer) img_usage |= vk::ImageUsageFlagBits::eColorAttachment;

    spec.set_all_usage(img_usage);
    spec.color_usage = vk::ImageUsageFlagBits::eStorage|
        vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc;

    if(use_raster_gbuffer)
    {
        spec.depth_present = true;
        spec.depth_usage = vk::ImageUsageFlagBits::eDepthStencilAttachment|
            vk::ImageUsageFlagBits::eTransferSrc;
        copy_spec.color_present = spec.color_present;
        copy_spec.color_format = spec.color_format;
        copy_spec.diffuse_present = spec.diffuse_present;
        copy_spec.diffuse_format = spec.diffuse_format;
        copy_spec.direct_present = spec.direct_present;
        copy_spec.direct_format = spec.direct_format;
    }
    else
        copy_spec = spec;

    copy_spec.set_all_usage(
        vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eStorage
    );

    if(!is_display_device)
    {
        r.gpu_to_cpu_timer = timer(d, "distribution frame to host");
        r.cpu_to_gpu_timer = timer(
            ctx->get_display_device(), "distribution frame from host"
        );
    }

    r.gbuffer.reset(d, max_target_size, ctx->get_display_count());
    r.gbuffer.add(spec);

    if(!is_display_device)
    {
        device_data& display_device = ctx->get_display_device();
        r.gbuffer_copy.reset(display_device, max_target_size, ctx->get_display_count());
        r.gbuffer_copy.add(copy_spec);
    }

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        auto& f = r.per_frame[i];

        if(!is_display_device)
        {
            device_data& display_device = ctx->get_display_device();

            size_t sz = max_target_size.x*max_target_size.y*color_format_size*ctx->get_display_count();
            for(size_t j = 0; j < r.gbuffer_copy.entry_count(); ++j)
            {
                transfer_buffer tb;
                tb.host_ptr = allocate_host_buffer({&display_device, &d}, sz);
                create_host_allocated_buffer(
                    d, tb.gpu_to_cpu, tb.gpu_to_cpu_mem, sz, tb.host_ptr
                );
                create_host_allocated_buffer(
                    display_device, tb.cpu_to_gpu, tb.cpu_to_gpu_mem, sz,
                    tb.host_ptr
                );
                f.transfer_buffers.push_back(tb);
            }

#ifdef WIN32
            static PFN_vkImportSemaphoreWin32HandleKHR vkImportSemaphoreWin32HandleKHR =
                PFN_vkImportSemaphoreWin32HandleKHR(vkGetInstanceProcAddr(ctx->get_vulkan_instance(), "vkImportSemaphoreWin32HandleKHR"));
            static PFN_vkGetSemaphoreWin32HandleKHR vkGetSemaphoreWin32HandleKHR =
                PFN_vkGetSemaphoreWin32HandleKHR(vkGetInstanceProcAddr(ctx->get_vulkan_instance(), "vkGetSemaphoreWin32HandleKHR"));
            vk::SemaphoreCreateInfo sem_info;
            vk::ExportSemaphoreCreateInfo esem_info(vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32);
            sem_info.pNext = &esem_info;
            f.gpu_to_cpu_sem = vkm(d, d.dev.createSemaphore(sem_info));
            VkSemaphoreGetWin32HandleInfoKHR win32info {
                VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
                nullptr,
                *f.gpu_to_cpu_sem,
                VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
            };
            vkGetSemaphoreWin32HandleKHR(d.dev, &win32info, &f.sem_handle);

            f.gpu_to_cpu_sem_copy = create_binary_semaphore(display_device);

            VkImportSemaphoreWin32HandleInfoKHR win32ImportSemaphoreInfo{
                VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
                nullptr,
                *f.gpu_to_cpu_sem_copy,
                0,
                VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
                &f.sem_handle,
                nullptr
            };

            vkImportSemaphoreWin32HandleKHR(d.dev, &win32ImportSemaphoreInfo);

#else
            vk::SemaphoreCreateInfo sem_info;
            vk::ExportSemaphoreCreateInfo esem_info(
                vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd
            );
            sem_info.pNext = &esem_info;
            f.gpu_to_cpu_sem = vkm(d, d.dev.createSemaphore(sem_info));
            f.sem_fd = d.dev.getSemaphoreFdKHR({f.gpu_to_cpu_sem});

            f.gpu_to_cpu_sem_copy = create_binary_semaphore(display_device);

            display_device.dev.importSemaphoreFdKHR({
                f.gpu_to_cpu_sem_copy, {},
                vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd,
                f.sem_fd
            });
#endif
            f.cpu_to_gpu_sem = create_timeline_semaphore(display_device);
            reset_transfer_command_buffers(
                i, r, f, target_size, ctx->get_display_device(), d
            );
        }
    }

    gbuffer_target transfer_target = r.gbuffer.get_array_target();
    if(use_raster_gbuffer)
    {
        gbuffer_target limited_target;
        limited_target.color = transfer_target.color;
        limited_target.diffuse = transfer_target.diffuse;
        limited_target.direct = transfer_target.direct;
        transfer_target = limited_target;
    }
    transfer_target.set_layout(is_display_device ?
        vk::ImageLayout::eGeneral : vk::ImageLayout::eTransferSrcOptimal
    );

    r.ray_tracer.reset(new Pipeline(
        d,
        get_distribution_render_size(rt_opt.distribution),
        transfer_target,
        rt_opt
    ));
}

template<typename Pipeline>
void rt_renderer<Pipeline>::init_primary_device_resources()
{
    size_t device_count = ctx->get_devices().size();
    if(device_count > 1)
    { // If multi-device, use parallel implementation
        std::vector<gbuffer_target> dimgs;
        for(size_t i = 0; i < per_device.size(); ++i)
        {
            gbuffer_target dimg;
            if(i == ctx->get_display_device().index)
            {
                dimg = per_device[i].gbuffer.get_array_target();
            }
            else
            {
                dimg = per_device[i].gbuffer_copy.get_array_target();
            }

            if(use_raster_gbuffer)
            {
                gbuffer_target limited_target;
                limited_target.color = dimg.color;
                limited_target.diffuse = dimg.diffuse;
                limited_target.direct = dimg.direct;
                dimg = limited_target;
            }
            dimgs.push_back(dimg);
        }

        std::vector<distribution_params> dist;
        for(per_device_data& r: per_device)
            dist.push_back(r.dist);

        stitch.reset(new stitch_stage(
            ctx->get_display_device(),
            ctx->get_size(),
            dimgs,
            dist,
            {
                opt.distribution.strategy,
                this->opt.active_viewport_count
            }
        ));
    }

    gbuffer_texture& gbuf_tex = per_device[ctx->get_display_device().index].gbuffer;
    if(use_raster_gbuffer)
    {
        raster_stage::options raster_opt;
        raster_opt.max_samplers = opt.max_samplers;
        raster_opt.max_3d_samplers = 0;
        raster_opt.pcf_samples = 0;
        raster_opt.pcss_samples = 0;
        raster_opt.output_layout = vk::ImageLayout::eGeneral;
        raster_opt.force_alpha_to_coverage = opt.post_process.bmfr || opt.post_process.svgf_denoiser ? true : false;

        // We take out the things that the path tracer should produce.
        std::vector<gbuffer_target> gbuffer_block_targets;
        for(size_t i = 0; i < gbuf_tex.get_multiview_block_count(); ++i)
        {
            gbuffer_target mv_target = gbuf_tex.get_multiview_block_target(i);
            mv_target.color = render_target();
            mv_target.diffuse = render_target();
            mv_target.direct = render_target();
            gbuffer_block_targets.push_back(mv_target);
        }

        if(gbuffer_block_targets[0].entry_count() != 0)
        {
            gbuffer_rasterizer.reset(new raster_stage(
                ctx->get_display_device(), gbuffer_block_targets, raster_opt
            ));
        }
    }

    gbuffer_target pp_target = gbuf_tex.get_array_target();
    pp_target.set_layout(vk::ImageLayout::eGeneral);
    post_processing.set_display(pp_target);
}

template<typename Pipeline>
void rt_renderer<Pipeline>::reset_transfer_command_buffers(
    uint32_t frame_index,
    per_device_data& r,
    per_frame_data& f,
    uvec2 transfer_size,
    device_data& primary,
    device_data& secondary
){
    f.gpu_to_cpu_cb = create_graphics_command_buffer(secondary);

    f.gpu_to_cpu_cb->begin(vk::CommandBufferBeginInfo{});
    r.gpu_to_cpu_timer.begin(f.gpu_to_cpu_cb, frame_index);

    f.cpu_to_gpu_cb = create_graphics_command_buffer(primary);

    f.cpu_to_gpu_cb->begin(vk::CommandBufferBeginInfo{});
    r.cpu_to_gpu_timer.begin(f.cpu_to_gpu_cb, frame_index);

    gbuffer_target target = r.gbuffer.get_array_target();
    gbuffer_target target_copy = r.gbuffer_copy.get_array_target();

    size_t j = 0;
    for(size_t i = 0; i < MAX_GBUFFER_ENTRIES; ++i)
    {
        if(!target_copy[i])
            continue;

        if(transfer_size.x * transfer_size.y == 0)
        {
            j++;
            continue;
        }

        // GPU -> CPU
        vk::BufferImageCopy region(
            0, 0, 0,
            {vk::ImageAspectFlagBits::eColor, 0, 0, (uint32_t)opt.active_viewport_count},
            {0,0,0}, {transfer_size.x, transfer_size.y, 1}
        );

        f.gpu_to_cpu_cb->copyImageToBuffer(
            target[i][frame_index].image, vk::ImageLayout::eTransferSrcOptimal,
            f.transfer_buffers[j].gpu_to_cpu, 1, &region
        );

        // CPU -> GPU
        vk::ImageMemoryBarrier img_barrier(
            {}, vk::AccessFlagBits::eTransferWrite,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            target_copy[i][frame_index].image,
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS}
        );

        f.cpu_to_gpu_cb->pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer,
            {},
            {}, {},
            img_barrier
        );

        f.cpu_to_gpu_cb->copyBufferToImage(
            f.transfer_buffers[j].cpu_to_gpu, target_copy[i][frame_index].image,
            vk::ImageLayout::eTransferDstOptimal, 1, &region
        );

        img_barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        img_barrier.dstAccessMask = {};
        img_barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        img_barrier.newLayout = vk::ImageLayout::eGeneral;
        f.cpu_to_gpu_cb->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eBottomOfPipe,
            {}, {}, {}, img_barrier
        );

        ++j;
    }

    r.gpu_to_cpu_timer.end(f.gpu_to_cpu_cb, frame_index);
    f.gpu_to_cpu_cb->end();
    r.cpu_to_gpu_timer.end(f.cpu_to_gpu_cb, frame_index);
    f.cpu_to_gpu_cb->end();
}

template class rt_renderer<path_tracer_stage>;
template class rt_renderer<whitted_stage>;
template class rt_renderer<feature_stage>;

}
