#include "tauray.hh"
#include "options.hh"
#include "window.hh"
#include "headless.hh"
#include "openxr.hh"
#include "looking_glass.hh"
#include "frame_server.hh"
#include "server_context.hh"
#include "raster_renderer.hh"
#include "dshgi_renderer.hh"
#include "dshgi_server.hh"
#include "frame_client.hh"
#include "rt_renderer.hh"
#include "scene.hh"
#include "camera.hh"
#include "texture.hh"
#include "environment_map.hh"
#include "sampler.hh"
#include "material.hh"
#include "gltf.hh"
#include "assimp.hh"
#include "misc.hh"
#include "load_balancer.hh"
#include <chrono>
#include <iostream>
#include <thread>
#include <numeric>
#include <filesystem>

namespace fs = std::filesystem;

namespace tr
{

struct throttler
{
    throttler(float throttle_fps)
    {
        if(throttle_fps != 0)
        {
            active = true;
            throttle_time = std::chrono::duration_cast<decltype(throttle_time)>(
                std::chrono::duration<float>(1.0/throttle_fps)
            );
            time = std::chrono::high_resolution_clock::now();
        }
        else active = false;
    }

    void step()
    {
        if(active)
        {
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = stop - time;
            if(duration < throttle_time)
                std::this_thread::sleep_for(throttle_time-duration);
            time = std::chrono::high_resolution_clock::now();
        }
    }
    bool active;
    std::chrono::high_resolution_clock::duration throttle_time;
    std::chrono::high_resolution_clock::time_point time;
};

void set_camera_params(const options& opt, scene_data& s)
{
    if(auto proj = opt.force_projection)
    {
        switch(*proj)
        {
        case camera::PERSPECTIVE:
            s.s->get_camera()->perspective(90.0f, 1.0f, 0.1f, 100.0f);
            break;
        case camera::ORTHOGRAPHIC:
            s.s->get_camera()->ortho(
                -1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 100.0f
            );
            break;
        case camera::EQUIRECTANGULAR:
            s.s->get_camera()->equirectangular(360, 180);
            break;
        default:
            break;
        }
    }

    s.s->get_camera()->set_aspect(
        opt.aspect_ratio > 0 ?
            opt.aspect_ratio : opt.width/(float)opt.height
    );
    if(opt.fov)
        s.s->get_camera()->set_fov(opt.fov);

    if(opt.camera_clip_range.near > 0)
        s.s->get_camera()->set_near(opt.camera_clip_range.near);
    if(opt.camera_clip_range.far > 0)
        s.s->get_camera()->set_far(opt.camera_clip_range.far);

    if(opt.depth_of_field.f_stop != 0)
        s.s->get_camera()->set_focus(
            opt.depth_of_field.f_stop,
            opt.depth_of_field.distance,
            opt.depth_of_field.sides,
            opt.depth_of_field.angle,
            opt.depth_of_field.sensor_size
        );
}

scene_data load_scenes(context& ctx, const options& opt)
{
    // The frame client does not need scene data :D
    if(opt.display == options::display_type::FRAME_CLIENT)
        return {};

    device_mask dev = device_mask::all(ctx);

    std::unique_ptr<environment_map> sky;
    if(opt.envmap.size())
        sky.reset(new environment_map(dev, opt.envmap));

    std::vector<scene_graph> scenes;
    for(const std::string& path: opt.scene_paths)
    {
        scene_graph sg_temp;

        fs::path fsp(path);
        if(fsp.extension() == ".gltf" || fsp.extension() == ".glb")
        {
            sg_temp = load_gltf(
                dev, path, opt.force_single_sided, opt.force_double_sided
            );
        }
        else
        {
            sg_temp = load_assimp(dev, path);
        }
        scenes.emplace_back(std::move(sg_temp));
        scene_graph& sg = scenes.back();

        for(auto& pair: sg.sh_grids)
            pair.second.set_order(opt.sh_order);

        if(opt.alpha_to_transmittance)
        {
            for(auto& pair: sg.models)
            for(auto& vg: pair.second)
            {
                if(vg.mat.albedo_factor.a < 1.0f)
                {
                    vg.mat.transmittance = (1.0f - vg.mat.albedo_factor.a);
                    vg.mat.albedo_factor.a = 1.0f;
                }
            }
        }
        else if(opt.transmittance_to_alpha >= 0.0f)
        {
            for(auto& pair: sg.models)
            for(auto& vg: pair.second)
            {
                vg.mat.albedo_factor *= mix(
                    1.0f, opt.transmittance_to_alpha, vg.mat.transmittance
                );
            }
        }

        if(opt.up_axis == 0)
        {
            sg.apply_transform(mat4(
                0,1,0,0,
                0,0,1,0,
                1,0,0,0,
                0,0,0,1
            ));
        }
        else if(opt.up_axis == 2)
        {
            sg.apply_transform(mat4(
                0,0,1,0,
                1,0,0,0,
                0,1,0,0,
                0,0,0,1
            ));
        }
    }

    scene_data s{
        std::move(sky),
        std::move(scenes),
        std::make_unique<scene>()
    };
    s.s->set_environment_map(s.sky.get());
    s.s->set_ambient(opt.ambient);
    for(scene_graph& sg: s.scenes)
    {
        sg.to_scene(*s.s);

        if(opt.camera != "")
        {
            auto it = sg.cameras.find(opt.camera);
            if(it != sg.cameras.end())
            {
                s.s->set_camera(&it->second);
            }
            else
            {
                // Blender's camera export is really annoying.
                it = sg.cameras.find(opt.camera + "_Orientation");
                if(it != sg.cameras.end())
                    s.s->set_camera(&it->second);
            }
        }
        else if(sg.cameras.size() != 0 && s.s->get_camera() == nullptr)
        {
            s.s->set_camera(&sg.cameras.begin()->second);
        }
    }

    if(s.s->get_camera() == nullptr)
    {
        if(opt.camera != "")
            throw std::runtime_error(
                "Failed to find a camera named " + opt.camera + "."
            );
    }
    else
    {
        set_camera_params(opt, s);
    }

    if(opt.animation_flag)
        s.s->play(opt.animation, !opt.replay, opt.animation == "");

    return s;
}

context* create_context(const options& opt)
{
    // The frame client does not need a context :D
    if(opt.display == options::display_type::FRAME_CLIENT)
        return nullptr;

    context::options ctx_opt;
    if(auto rtype = std::get_if<options::basic_pipeline_type>(&opt.renderer))
    {
        if(*rtype == options::RASTER || *rtype == options::DSHGI_CLIENT)
            ctx_opt.disable_ray_tracing = true;
    }
#if _WIN32
    // WORKAROUND: Multi-device rendering on Windows is currently not supported
    // due to problems encountered related to multi threading and freezing
    // during semaphore signal operations
    ctx_opt.physical_device_indices = { -1 };
#else
    ctx_opt.physical_device_indices = opt.devices;
#endif
    ctx_opt.max_timestamps = 128;
    ctx_opt.enable_vulkan_validation = opt.validation;
    ctx_opt.fake_device_multiplier = opt.fake_devices;

    if(opt.renderer == options::DSHGI_SERVER)
    {
        return new server_context(ctx_opt);
    }
    else if(opt.headless != "" || opt.headful)
    {
        headless::options hd_opt;
        (context::options&)hd_opt = ctx_opt;
        hd_opt.size = uvec2(opt.width, opt.height);
        hd_opt.output_prefix = opt.headless;
        hd_opt.output_compression = opt.compression;
        hd_opt.output_format = opt.format;
        hd_opt.output_file_type = opt.filetype;
        hd_opt.viewer = opt.headful;
        hd_opt.viewer_fullscreen = opt.fullscreen;
        hd_opt.display_count =
            opt.headful ? 1 : opt.camera_grid.w * opt.camera_grid.h;
        hd_opt.single_frame = !opt.animation_flag && !opt.frames;
        hd_opt.first_frame_index = opt.skip_frames;
        hd_opt.skip_nan_check =
            (std::holds_alternative<feature_stage::feature>(opt.renderer) &&
             isnan(opt.default_value)) ||
            (opt.spatial_reprojection.size() != 0 &&
             opt.spatial_reprojection.size() < hd_opt.display_count);
        return new headless(hd_opt);
    }
    else if(opt.display == options::display_type::OPENXR)
    {
        openxr::options xr_opt;
        (context::options&)xr_opt = ctx_opt;
        xr_opt.size = uvec2(opt.width, opt.height);
        xr_opt.fullscreen = opt.fullscreen;
        xr_opt.hdr_display = opt.hdr;
        return new openxr(xr_opt);
    }
    else if(opt.display == options::display_type::LOOKING_GLASS)
    {
        looking_glass::options lkg_opt;
        (context::options&)lkg_opt = ctx_opt;
        lkg_opt.vsync = opt.vsync;
        lkg_opt.viewport_size = uvec2(opt.width, opt.height);
        lkg_opt.viewport_count = opt.lkg_params.viewports;
        lkg_opt.mid_plane_dist = opt.lkg_params.midplane;
        lkg_opt.depthiness = opt.lkg_params.depth;
        lkg_opt.relative_view_distance = opt.lkg_params.relative_dist;
        if(opt.lkg_calibration.display_index >= 0)
        {
            looking_glass::options::calibration_data calib;
            memcpy(&calib, &opt.lkg_calibration, sizeof(calib));
            lkg_opt.calibration_override.emplace(calib);
        }
        return new looking_glass(lkg_opt);
    }
    else if(opt.display == options::display_type::FRAME_SERVER)
    {
        frame_server::options fs_opt;
        (context::options&)fs_opt = ctx_opt;
        fs_opt.size = uvec2(opt.width, opt.height);
        fs_opt.port_number = opt.port;
        return new frame_server(fs_opt);
    }
    else
    {
        window::options win_opt;
        (context::options&)win_opt = ctx_opt;
        win_opt.size = uvec2(opt.width, opt.height);
        win_opt.fullscreen = opt.fullscreen;
        win_opt.vsync = opt.vsync;
        win_opt.hdr_display = opt.hdr;
        return new window(win_opt);
    }
}

renderer* create_renderer(context& ctx, options& opt, scene& s)
{
    tonemap_stage::options tonemap;
    tonemap.tonemap_operator = opt.tonemap;
    tonemap.exposure = opt.exposure;
    tonemap.gamma = opt.gamma;
    tonemap.alpha_grid_background = opt.headless == "";
    tonemap.post_resolve = opt.tonemap_post_resolve;

    bool use_shadow_terminator_fix = false;
    bool has_tri_lights = false;
    bool has_point_lights = s.get_point_lights().size() + s.get_spotlights().size() > 0;
    bool has_directional_lights = s.get_directional_lights().size() > 0;
    for(const mesh_object* o: s.get_mesh_objects())
    {
        if(o->get_shadow_terminator_offset() > 0.0f)
            use_shadow_terminator_fix = true;
        const model* m = o->get_model();
        if(m) for(const model::vertex_group& vg: *m)
        {
            if(vg.mat.emission_factor != vec3(0))
                has_tri_lights = true;
        }
    }

    scene_stage::options scene_options;
    scene_options.max_instances = s.get_instance_count();
    scene_options.max_lights = s.get_point_lights().size() + s.get_spotlights().size();
    scene_options.gather_emissive_triangles = has_tri_lights && opt.sample_emissive_triangles > 0;
    scene_options.pre_transform_vertices = opt.pre_transform_vertices;
    scene_options.group_strategy = opt.as_strategy;

    taa_stage::options taa;
    taa.blending_ratio = 1.0f - 1.0f/opt.taa.sequence_length;

    rt_camera_stage::options rc_opt;
    rc_opt.projection = s.get_camera()->get_projection_type();
    rc_opt.max_instances = s.get_instance_count();
    rc_opt.max_samplers = s.get_sampler_count();
    rc_opt.min_ray_dist = opt.min_ray_dist;
    rc_opt.max_ray_depth = opt.max_ray_depth;
    rc_opt.samples_per_pass = min(opt.samples_per_pass, opt.samples_per_pixel);
    // Round sample count to next multiple of samples_per_pass
    rc_opt.samples_per_pixel =
        ((opt.samples_per_pixel + rc_opt.samples_per_pass - 1)
         / rc_opt.samples_per_pass) * rc_opt.samples_per_pass;
    rc_opt.rng_seed = opt.rng_seed;
    rc_opt.local_sampler = opt.sampler;
    rc_opt.transparent_background = opt.transparent_background;
    rc_opt.pre_transformed_vertices = opt.pre_transform_vertices;

    if(opt.progress)
    {
        rc_opt.max_passes_per_command_buffer = max(
            rc_opt.samples_per_pixel / rc_opt.samples_per_pass / 100,
            1
        );
    }

    light_sampling_weights sampling_weights;
    sampling_weights.point_lights = has_point_lights ? opt.sample_point_lights : 0.0f;
    sampling_weights.directional_lights = has_directional_lights ? opt.sample_directional_lights : 0.0f;
    sampling_weights.envmap = s.get_environment_map() ? opt.sample_envmap : 0.0f;
    sampling_weights.emissive_triangles = has_tri_lights ? opt.sample_emissive_triangles : 0.0f;

    s.auto_shadow_maps(
        opt.shadow_map_resolution,
        vec3(
            opt.shadow_map_radius,
            opt.shadow_map_radius,
            opt.shadow_map_depth
        ),
        vec2(opt.shadow_map_bias/5.0f, opt.shadow_map_bias),
        opt.shadow_map_cascades,
        opt.shadow_map_resolution,
        0.01f,
        vec2(0.005, opt.shadow_map_bias*2)
    );

    if(auto rtype = std::get_if<feature_stage::feature>(&opt.renderer))
    {
        feature_renderer::options rt_opt;
        (rt_camera_stage::options&)rt_opt = rc_opt;
        rt_opt.default_value = vec4(opt.default_value);
        rt_opt.feat = *rtype;
        rt_opt.post_process.tonemap = tonemap;
        rt_opt.scene_options = scene_options;
        return new feature_renderer(ctx, rt_opt);
    }
    else if(auto rtype = std::get_if<options::basic_pipeline_type>(&opt.renderer))
    {
        switch(*rtype)
        {
        case options::PATH_TRACER:
            {
                path_tracer_renderer::options rt_opt;
                (rt_camera_stage::options&)rt_opt = rc_opt;
                rt_opt.use_shadow_terminator_fix =
                    opt.shadow_terminator_fix && use_shadow_terminator_fix;
                rt_opt.use_white_albedo_on_first_bounce =
                    opt.use_white_albedo_on_first_bounce;
                rt_opt.film = opt.film;
                rt_opt.mis_mode = opt.multiple_importance_sampling;
                rt_opt.film_radius = opt.film_radius;
                rt_opt.russian_roulette_delta = opt.russian_roulette;
                rt_opt.indirect_clamping = opt.indirect_clamping;
                rt_opt.regularization_gamma = opt.regularization;
                rt_opt.sampling_weights = sampling_weights;
                rt_opt.bounce_mode = opt.bounce_mode;
                rt_opt.tri_light_mode = opt.tri_light_mode;
                rt_opt.post_process.tonemap = tonemap;
                rt_opt.depth_of_field = opt.depth_of_field.f_stop != 0;
                if(opt.temporal_reprojection > 0.0f)
                    rt_opt.post_process.temporal_reprojection =
                        temporal_reprojection_stage::options{opt.temporal_reprojection, {}};
                if(opt.spatial_reprojection.size() > 0)
                    rt_opt.post_process.spatial_reprojection =
                        spatial_reprojection_stage::options{};
                if(opt.taa.sequence_length != 0)
                    rt_opt.post_process.taa = taa;
                rt_opt.hide_lights = opt.hide_lights;
                rt_opt.active_viewport_count =
                    opt.spatial_reprojection.size() == 0 ?
                    ctx.get_display_count() :
                    opt.spatial_reprojection.size();
                rt_opt.accumulate = opt.accumulation;
                rt_opt.post_process.tonemap.reorder = get_viewport_reorder_mask(
                    opt.spatial_reprojection,
                    ctx.get_display_count()
                );
                if (opt.denoiser == options::denoiser_type::SVGF)
                {
                    svgf_stage::options svgf_opt{};
                    svgf_opt.atrous_diffuse_iters = opt.svgf_params.atrous_diffuse_iter;
                    svgf_opt.atrous_spec_iters = opt.svgf_params.atrous_spec_iter;
                    svgf_opt.atrous_kernel_radius = opt.svgf_params.atrous_kernel_radius;
                    svgf_opt.sigma_l = opt.svgf_params.sigma_l;
                    svgf_opt.sigma_n = opt.svgf_params.sigma_n;
                    svgf_opt.sigma_z = opt.svgf_params.sigma_z;
                    svgf_opt.temporal_alpha_color = opt.svgf_params.min_alpha_color;
                    svgf_opt.temporal_alpha_moments = opt.svgf_params.min_alpha_moments;
                    rt_opt.post_process.svgf_denoiser = svgf_opt;
                }
                else if (opt.denoiser == options::denoiser_type::BMFR)
                    rt_opt.post_process.bmfr = bmfr_stage::options{ bmfr_stage::bmfr_settings::DIFFUSE_ONLY };
                rt_opt.scene_options = scene_options;
                rt_opt.distribution.strategy = opt.distribution_strategy;
                if(ctx.get_devices().size() == 1)
                    rt_opt.distribution.strategy = DISTRIBUTION_DUPLICATE;
                return new path_tracer_renderer(ctx, rt_opt);
            }
        case options::DIRECT:
            {
                direct_renderer::options rt_opt;
                (rt_camera_stage::options&)rt_opt = rc_opt;
                rt_opt.film = opt.film;
                rt_opt.film_radius = opt.film_radius;
                rt_opt.sampling_weights = sampling_weights;
                rt_opt.bounce_mode = opt.bounce_mode;
                rt_opt.tri_light_mode = opt.tri_light_mode;
                rt_opt.post_process.tonemap = tonemap;
                if(opt.temporal_reprojection > 0.0f)
                    rt_opt.post_process.temporal_reprojection =
                        temporal_reprojection_stage::options{opt.temporal_reprojection, {}};
                if(opt.spatial_reprojection.size() > 0)
                    rt_opt.post_process.spatial_reprojection =
                        spatial_reprojection_stage::options{};
                if(opt.taa.sequence_length != 0)
                    rt_opt.post_process.taa = taa;
                rt_opt.active_viewport_count =
                    opt.spatial_reprojection.size() == 0 ?
                    ctx.get_display_count() :
                    opt.spatial_reprojection.size();
                rt_opt.accumulate = opt.accumulation;
                rt_opt.post_process.tonemap.reorder = get_viewport_reorder_mask(
                    opt.spatial_reprojection,
                    ctx.get_display_count()
                );
                if(opt.denoiser == options::denoiser_type::SVGF)
                {
                    svgf_stage::options svgf_opt{};
                    svgf_opt.atrous_diffuse_iters = opt.svgf_params.atrous_diffuse_iter;
                    svgf_opt.atrous_spec_iters = opt.svgf_params.atrous_spec_iter;
                    svgf_opt.atrous_kernel_radius = opt.svgf_params.atrous_kernel_radius;
                    svgf_opt.sigma_l = opt.svgf_params.sigma_l;
                    svgf_opt.sigma_n = opt.svgf_params.sigma_n;
                    svgf_opt.sigma_z = opt.svgf_params.sigma_z;
                    svgf_opt.temporal_alpha_color = opt.svgf_params.min_alpha_color;
                    svgf_opt.temporal_alpha_moments = opt.svgf_params.min_alpha_moments;
                    rt_opt.post_process.svgf_denoiser = svgf_opt;
                }
                else if(opt.denoiser == options::denoiser_type::BMFR)
                    rt_opt.post_process.bmfr = bmfr_stage::options{ bmfr_stage::bmfr_settings::DIFFUSE_ONLY };
                rt_opt.scene_options = scene_options;
                rt_opt.distribution.strategy = opt.distribution_strategy;
                if(ctx.get_devices().size() == 1)
                    rt_opt.distribution.strategy = DISTRIBUTION_DUPLICATE;
                return new direct_renderer(ctx, rt_opt);
            }
        case options::WHITTED:
            {
                whitted_renderer::options rt_opt;
                (rt_camera_stage::options&)rt_opt = rc_opt;
                rt_opt.post_process.tonemap = tonemap;
                rt_opt.scene_options = scene_options;
                if(opt.taa.sequence_length != 0)
                    rt_opt.post_process.taa = taa;
                return new whitted_renderer(ctx, rt_opt);
            }
        case options::RASTER:
            {
                raster_renderer::options rr_opt;
                rr_opt.max_samplers = s.get_sampler_count();
                rr_opt.msaa_samples = opt.samples_per_pixel;
                rr_opt.sample_shading = opt.sample_shading;
                if(opt.taa.sequence_length != 0)
                    rr_opt.post_process.taa = taa;
                rr_opt.post_process.tonemap = tonemap;
                rr_opt.pcf_samples = min(opt.pcf, 64);
                rr_opt.omni_pcf_samples = min(opt.pcf, 64);
                rr_opt.pcss_samples = min(opt.pcss, 64);
                rr_opt.pcss_minimum_radius = opt.pcss_minimum_radius;
                rr_opt.z_pre_pass = opt.use_z_pre_pass;
                rr_opt.scene_options = scene_options;
                return new raster_renderer(ctx, rr_opt);
            }
        case options::DSHGI:
            {
                dshgi_renderer::options dr_opt;
                sh_renderer::options sh;
                (rt_stage::options&)sh = rc_opt;
                sh.samples_per_probe = opt.samples_per_probe;
                sh.film = opt.film;
                sh.film_radius = opt.film_radius;
                sh.mis_mode = opt.multiple_importance_sampling;
                sh.russian_roulette_delta = opt.russian_roulette;
                sh.temporal_ratio = opt.dshgi_temporal_ratio;
                sh.indirect_clamping = opt.indirect_clamping;
                sh.regularization_gamma = opt.regularization;
                sh.sampling_weights = sampling_weights;
                dr_opt.sh_source = sh;
                dr_opt.sh_order = opt.sh_order;
                dr_opt.use_probe_visibility = opt.use_probe_visibility;
                if(opt.taa.sequence_length != 0)
                    dr_opt.post_process.taa = taa;
                dr_opt.post_process.tonemap = tonemap;
                dr_opt.max_samplers = s.get_sampler_count();
                dr_opt.msaa_samples = opt.samples_per_pixel;
                dr_opt.sample_shading = opt.sample_shading;
                dr_opt.pcf_samples = min(opt.pcf, 64);
                dr_opt.omni_pcf_samples = min(opt.pcf, 64);
                dr_opt.pcss_samples = min(opt.pcss, 64);
                dr_opt.pcss_minimum_radius = opt.pcss_minimum_radius;
                dr_opt.z_pre_pass = opt.use_z_pre_pass;
                dr_opt.scene_options = scene_options;
                dr_opt.scene_options.alloc_sh_grids = true;
                return new dshgi_renderer(ctx, dr_opt);
            }
        case options::DSHGI_SERVER:
            {
                dshgi_server::options dr_opt;
                (rt_stage::options&)dr_opt.sh = rc_opt;
                dr_opt.sh.samples_per_probe = opt.samples_per_probe;
                dr_opt.sh.film = opt.film;
                dr_opt.sh.film_radius = opt.film_radius;
                dr_opt.sh.mis_mode = opt.multiple_importance_sampling;
                dr_opt.sh.russian_roulette_delta = opt.russian_roulette;
                dr_opt.sh.temporal_ratio = opt.dshgi_temporal_ratio;
                dr_opt.sh.indirect_clamping = opt.indirect_clamping;
                dr_opt.sh.regularization_gamma = opt.regularization;
                dr_opt.sh.sampling_weights = sampling_weights;
                dr_opt.port_number = opt.port;
                return new dshgi_server(ctx, dr_opt);
            }
        case options::DSHGI_CLIENT:
            {
                dshgi_renderer::options dr_opt;
                dshgi_client::options client;
                client.server_address = opt.connect;
                dr_opt.sh_source = client;
                dr_opt.sh_order = opt.sh_order;
                dr_opt.use_probe_visibility = opt.use_probe_visibility;
                dr_opt.post_process.tonemap = tonemap;
                if(opt.taa.sequence_length != 0)
                    dr_opt.post_process.taa = taa;
                dr_opt.max_samplers = s.get_sampler_count();
                dr_opt.msaa_samples = opt.samples_per_pixel;
                dr_opt.sample_shading = opt.sample_shading;
                dr_opt.pcf_samples = min(opt.pcf, 64);
                dr_opt.omni_pcf_samples = min(opt.pcf/2, 64);
                dr_opt.pcss_samples = min(opt.pcss, 64);
                dr_opt.pcss_minimum_radius = opt.pcss_minimum_radius;
                dr_opt.z_pre_pass = opt.use_z_pre_pass;
                dr_opt.scene_options = scene_options;
                dr_opt.scene_options.alloc_sh_grids = true;
                return new dshgi_renderer(ctx, dr_opt);
            }
        };
    }
    return nullptr;
}

std::vector<camera> generate_cameras(camera& tracked, options& opt)
{
    if(
        opt.camera_grid.w * opt.camera_grid.h <= 1 &&
        opt.camera_offset == vec3(0)
    ) return {};

    std::vector<camera> res;
    float width = (opt.camera_grid.w-1)*opt.camera_grid.x;
    float height = (opt.camera_grid.h-1)*opt.camera_grid.y;

    vec2 fov = vec2(tracked.get_hfov(), tracked.get_vfov());
    vec2 tfov = tan(glm::radians(fov) * 0.5f);

    quat grid_rotation = tr::angleAxis(
        glm::radians(opt.camera_grid_roll),
        vec3(0.0f, 0.0f, 1.0f)
    );

    for(int y = 0; y < opt.camera_grid.h; ++y)
    for(int x = 0; x < opt.camera_grid.w; ++x)
    {
        camera cam(&tracked);
        cam.copy_projection(tracked);
        vec3 grid_pos = grid_rotation * vec3(
            -width*0.5f + x*opt.camera_grid.x,
            height*0.5f - y*opt.camera_grid.y,
            0
        );
        vec2 pan = -vec2(grid_pos.x, grid_pos.y)/(tfov * opt.camera_recentering_distance);
        cam.set_position(grid_pos + opt.camera_offset);
        cam.set_pan(pan);
        res.push_back(cam);
    }
    return res;
}

void show_stats(scene_data& sd, options& opt)
{
    if(!opt.scene_stats)
        return;

    std::cout << "\nScene statistics: \n";

    std::set<const mesh*> meshes;
    for(const mesh_object* mo: sd.s->get_mesh_objects())
    {
        if(!mo || !mo->get_model()) continue;
        for(const model::vertex_group& vg: *mo->get_model())
            meshes.insert(vg.m);
    }
    std::cout << "Number of unique meshes = " << meshes.size() << std::endl;
    std::cout << "Number of mesh instances = " << sd.s->get_instance_count() << std::endl;
    //std::cout << "Number of BLASes (depends on settings) = " << sd.s->get_blas_group_count() << std::endl;

    //Calculating the number of triangles and dynamic objects
    uint32_t triangle_count = 0;
    uint32_t dyn_obj_count = 0;
    for(auto& mesh_obj: sd.s->get_mesh_objects())
    {
        for (auto& group: *mesh_obj->get_model())
        {
            triangle_count += group.m->get_indices().size() / 3;
        }
        dyn_obj_count += mesh_obj->is_static() ? 0:1;
    }
    std::cout << "Number of triangles = " << triangle_count << std::endl;

    uint32_t objects_count = sd.s->get_mesh_objects().size();
    std::cout << "\nNumber of objects = " <<  objects_count << std::endl;
    std::cout << "Static objects = " << objects_count - dyn_obj_count << std::endl;
    std::cout << "Dynamic objects = " << dyn_obj_count << std::endl;

    std::cout << "\nNumber of textures = " << sd.s->get_sampler_count() << std::endl;
    std::cout << "Number of point lights = " << sd.s->get_point_lights().size() << std::endl;
    std::cout << "Number of spot lights = " << sd.s->get_spotlights().size() << std::endl;

    opt.scene_stats = false;
}

void interactive_viewer(context& ctx, scene_data& sd, options& opt)
{
    scene& s = *sd.s;
    load_balancer lb(ctx, opt.workload);
    camera cam;
    if(s.get_camera())
    {
        cam = *s.get_camera();
        cam.set_parent(nullptr, true);
        cam.stop();
    }
    else
    {
        cam.set_position(vec3(0,0,2));
        cam.perspective(90, opt.width/(float)opt.height, 0.1f, 300.0f);
    }
    std::vector<camera> cameras = generate_cameras(cam, opt);
    if(cameras.size() == 0) s.set_camera(&cam);
    else
    {
        s.set_camera(&cameras[0]);
        s.add_control_node(cam);
    }

    std::unique_ptr<renderer> rr;

    float speed = 1.0f;
    vec3 euler = cam.get_orientation_euler();
    float pitch = euler.x;
    float yaw = euler.y;
    float roll = euler.z;
    float sensitivity = 0.2;
    bool paused = false;
    int camera_index = 0;
    throttler throttle(opt.throttle);

    if(openxr* xr = dynamic_cast<openxr*>(&ctx))
    {
        xr->setup_xr_surroundings(s, &cam);
        sensitivity = 0;
    }

    if(looking_glass* lkg = dynamic_cast<looking_glass*>(&ctx))
    {
        cameras.clear();
        lkg->setup_cameras(s, &cam);
    }

    s.reorder_cameras_by_active(opt.spatial_reprojection);
    s.set_camera_jitter(get_camera_jitter_sequence(opt.taa.sequence_length, ctx.get_size()));

    std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    float delta = 0.0f;
    bool focused = true;
    bool camera_locked = false;
    bool recreate_renderer = true;
    bool crash_on_exception = true;
    bool camera_moved = false;

    ivec3 camera_movement = ivec3(0);
    std::string command_line;
    while(opt.running)
    {
        camera_moved = false;
        if(nonblock_getline(command_line))
        {
            if(parse_command(command_line.c_str(), opt))
            {
                set_camera_params(opt, sd);
                recreate_renderer = true;
                camera_moved = true;
            }
        }

        if(recreate_renderer)
        {
            try
            {
                s.set_camera_jitter(get_camera_jitter_sequence(opt.taa.sequence_length, ctx.get_size()));
                rr.reset(create_renderer(ctx, opt, s));
                rr->set_scene(&s);
                ctx.set_displaying(false);
                for(int i = 0; i < opt.warmup_frames; ++i)
                    if(!opt.skip_render) rr->render();
                ctx.set_displaying(true);
            }
            catch(std::runtime_error& err)
            {
                if(crash_on_exception) throw err;
                else TR_ERR(err.what());
            }
            show_stats(sd, opt);

            recreate_renderer = false;
        }

        SDL_Event event;
        while(SDL_PollEvent(&event)) switch(event.type)
        {
        case SDL_QUIT:
            opt.running = false;
            break;
        case SDL_KEYDOWN:
        case SDL_KEYUP:
            if(event.type == SDL_KEYDOWN)
            {
                if(event.key.keysym.sym == SDLK_ESCAPE) opt.running = false;
                if(event.key.keysym.sym == SDLK_RETURN) paused = !paused;
                if(event.key.keysym.sym == SDLK_PAGEUP)
                {
                    camera_index++;
                    camera_moved = true;
                }
                if(event.key.keysym.sym == SDLK_PAGEDOWN)
                {
                    camera_index--;
                    camera_moved = true;
                }
                if(event.key.keysym.sym == SDLK_t && !opt.timing)
                    ctx.get_timing().print_last_trace(opt.trace);
                if(event.key.keysym.sym == SDLK_0)
                {
                    // Full camera reset, for when you get lost ;)
                    cam.set_global_position();
                    cam.set_global_orientation();
                    camera_moved = true;
                }
                if(event.key.keysym.sym == SDLK_F1)
                {
                    camera_locked = !camera_locked;
                    SDL_SetWindowGrab(SDL_GetWindowFromID(event.key.windowID), (SDL_bool)!camera_locked);
                    SDL_SetRelativeMouseMode((SDL_bool)!camera_locked);
                }
                if(event.key.keysym.sym == SDLK_F5)
                {
                    shader_source::clear_binary_cache();
                    rr.reset();
                    recreate_renderer = true;
                    crash_on_exception = false;
                }
            }
            if(event.key.repeat == SDL_FALSE)
            {
                int direction = event.type == SDL_KEYDOWN ? 1 : -1;
                if(event.key.keysym.scancode == SDL_SCANCODE_W)
                    camera_movement.z -= direction;
                if(event.key.keysym.scancode == SDL_SCANCODE_S)
                    camera_movement.z += direction;
                if(event.key.keysym.scancode == SDL_SCANCODE_A)
                    camera_movement.x -= direction;
                if(event.key.keysym.scancode == SDL_SCANCODE_D)
                    camera_movement.x += direction;
                if(event.key.keysym.scancode == SDL_SCANCODE_LSHIFT)
                    camera_movement.y -= direction;
                if(event.key.keysym.scancode == SDL_SCANCODE_SPACE)
                    camera_movement.y += direction;
            }
            break;
        case SDL_MOUSEWHEEL:
            if(event.wheel.y != 0)
                speed *= pow(1.1, event.wheel.y);
            break;
        case SDL_MOUSEMOTION:
            if(focused && !camera_locked)
            {
                pitch = std::clamp(
                    pitch-event.motion.yrel*sensitivity, -90.0f, 90.0f
                );
                yaw -= event.motion.xrel*sensitivity;
                roll = 0;
                camera_moved = true;
            }
            break;
        case SDL_WINDOWEVENT:
            if(event.window.event == SDL_WINDOWEVENT_FOCUS_LOST)
                focused = false;
            if(event.window.event == SDL_WINDOWEVENT_FOCUS_GAINED)
                focused = true;
            break;
        }

        if(ctx.init_frame())
            break;

        if(cameras.size() != 0)
        {
            while(camera_index < 0) camera_index += cameras.size();
            camera_index %= cameras.size();
            s.set_camera(&cameras[camera_index]);
        }

        if(!camera_locked)
        {
            camera_movement = clamp(camera_movement, ivec3(-1), ivec3(1));
            if(camera_movement != ivec3(0))
                camera_moved = true;
            cam.translate_local(vec3(camera_movement)*delta*speed);
            cam.set_orientation(pitch, yaw, roll);
        }

        if(camera_moved || !opt.accumulation)
        {
            // With this commented line, sample counter restarts whenever the
            // camera moves. This makes the noise pattern look still when
            // moving, which may be unwanted, but could provide lower noise
            // with some samplers in the future.
            //if(rr) rr->reset_accumulation(opt.accumulation);
            if(rr) rr->reset_accumulation(false);
        }

        s.update(paused || !opt.animation_flag ? 0 : delta * 1000000);

        try
        {
            if(rr) rr->render();
            else
            {
                ctx.end_frame(ctx.begin_frame());
            }
        }
        catch(vk::OutOfDateKHRError& e)
        {
            rr.reset();
            if(window* win = dynamic_cast<window*>(&ctx))
                win->recreate_swapchains();
            else if(openxr* xr = dynamic_cast<openxr*>(&ctx))
                xr->recreate_swapchains();
            else if(looking_glass* lkg = dynamic_cast<looking_glass*>(&ctx))
                lkg->recreate_swapchains();
            else break;
        }
        if(opt.timing) ctx.get_timing().print_last_trace(opt.trace);

        throttle.step();
        if(rr) lb.update(*rr);

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end-start;
        delta = elapsed.count();
        start = end;
    }

    // Ensure everything is finished before going to destructors.
    ctx.sync();

    // TODO: This hack prevents SteamVR from freezing on exit. This isn't just
    // a Tauray bug, it seems every Vulkan+Linux+Nvidia combo causes that...
    // Remove it once SteamVR isn't busted anymore.
    if(opt.display == options::display_type::OPENXR)
        abort();
}

void replay_viewer(context& ctx, scene_data& sd, options& opt)
{
    scene& s = *sd.s;
    load_balancer lb(ctx, opt.workload);
    camera default_cam;
    if(s.get_camera() == nullptr)
    {
        TR_WARN(
            "Warning: no camera is defined in the scene, so a default camera "
            "setup is used. You probably do not want this in replay mode."
        );
        default_cam.set_position(vec3(0,0,2));
        default_cam.perspective(90, opt.width/(float)opt.height, 0.1f, 300.0f);
        s.set_camera(&default_cam);
    }
    camera_log clog(s.get_camera());

    std::vector<camera_log> camera_logs;
    std::vector<camera> cameras = generate_cameras(*s.get_camera(), opt);
    if(cameras.size() == 0)
    {
        if(openxr* xr = dynamic_cast<openxr*>(&ctx))
            xr->setup_xr_surroundings(s, s.get_camera());
        if(looking_glass* lkg = dynamic_cast<looking_glass*>(&ctx))
            lkg->setup_cameras(s, s.get_camera());
        for(camera* cam: s.get_cameras())
            camera_logs.emplace_back(cam);
    }
    else
    {
        s.add_control_node(*s.get_camera());
        s.clear_cameras();
        for(camera& cam: cameras)
        {
            s.add(cam);
            camera_logs.emplace_back(&cam);
        }
    }
    s.reorder_cameras_by_active(opt.spatial_reprojection);
    s.set_camera_jitter(get_camera_jitter_sequence(opt.taa.sequence_length, ctx.get_size()));

    std::unique_ptr<renderer> rr;

    // Ticks in microseconds per update.
    time_ticks update_dt = round(1000000.0/opt.framerate);

    size_t frame_count = opt.frames ? opt.frames : -1;
    bool is_animated = s.is_playing();
    if(!opt.frames && !is_animated) frame_count = 1;

    if(opt.progress && frame_count != size_t(-1))
    {
        progress_tracker::options popt;
        popt.expected_frame_count = frame_count;
        ctx.get_progress_tracker().begin(popt);
    }

    for(size_t i = 0; i < frame_count; ++i)
    {
        if(!opt.frames && is_animated && !s.is_playing())
            break;

        if(!rr)
        {
            rr.reset(create_renderer(ctx, opt, s));
            rr->set_scene(&s);
            lb.update(*rr);
            ctx.set_displaying(false);
            for(int i = 0; i < opt.warmup_frames; ++i)
            {
                if(!opt.skip_render)
                {
                    s.update(0, true);
                    rr->render();
                    lb.update(*rr);
                }
            }
            ctx.set_displaying(true);
        }

        if(ctx.init_frame())
            break;

        // First frame should not update time.
        time_ticks dt = i == 0 ? 0 : update_dt;
        s.update(dt, true);
        for(camera_log& clog: camera_logs)
            clog.frame(dt);

        try
        {
            if(!opt.skip_render && (int)i >= opt.skip_frames)
            {
                rr->reset_accumulation();
                rr->render();
                if(opt.timing) ctx.get_timing().print_last_trace(opt.trace);
            }
        }
        catch(vk::OutOfDateKHRError& e)
        {
            rr.reset();
            if(window* win = dynamic_cast<window*>(&ctx))
                win->recreate_swapchains();
            if(openxr* xr = dynamic_cast<openxr*>(&ctx))
                xr->recreate_swapchains();
            else break;
        }

        lb.update(*rr);
    }

    if(opt.camera_log != "")
    {
        for(size_t i = 0; i < camera_logs.size(); ++i)
        {
            std::string filename = opt.camera_log;
            if(camera_logs.size() != 1)
                filename += std::to_string(i);
            camera_logs[i].write(filename+".json");
        }
    }

    // Ensure everything is finished before going to destructors.
    ctx.get_timing().wait_all_frames(opt.timing, opt.trace);
}

void headless_server(context& ctx, scene_data& sd, options& opt)
{
    scene& s = *sd.s;
    std::unique_ptr<renderer> rr(create_renderer(ctx, opt, s));
    rr->set_scene(&s);
    ctx.set_displaying(true);

    throttler throttle(opt.throttle);
    std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    float delta = 0.0f;
    while(opt.running)
    {
        if(ctx.init_frame())
            break;

        s.update(delta * 1000000, true);

        rr->reset_accumulation();

        rr->render();

        throttle.step();

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end-start;
        delta = elapsed.count();
        start = end;
    }

    // Ensure everything is finished before going to destructors.
    ctx.sync();
    TR_LOG("Server shutting down.");
}

void run(context& ctx, scene_data& sd, options& opt)
{
    if(opt.display == options::display_type::FRAME_CLIENT)
    {
        frame_client(opt);
    }
    else if(opt.renderer == options::DSHGI_SERVER)
    {
        headless_server(ctx, sd, opt);
    }
    else if(opt.replay)
    {
        replay_viewer(ctx, sd, opt);
    }
    else
    {
        interactive_viewer(ctx, sd, opt);
    }
}

}
