#include "tauray.hh"
#include "openxr.hh"
#include "window.hh"
#include "gltf.hh"
#include "misc.hh"
#include <iostream>
#include <fstream>

using namespace tr;

struct demo_scene_data
{
    scene_assets mesh_stream;
    std::vector<scene_assets> static_scene;
    std::unique_ptr<scene> shadow_realm;
    std::unique_ptr<scene> real_world;

    std::vector<entity> frame_entities;
};

demo_scene_data load_demo_scenes(context& ctx, const options& opt)
{
    device_mask dev = device_mask::all(ctx);
    demo_scene_data data;
    data.shadow_realm.reset(new scene);
    data.real_world.reset(new scene);

    if(opt.scene_paths.size() > 0)
    {
        const std::string& path = opt.scene_paths[0];
        data.mesh_stream = load_gltf(
            dev, *data.shadow_realm, path, opt.force_single_sided, opt.force_double_sided
        );

        data.shadow_realm->foreach([&](entity id, tr::model& model, tr::transformable& t){
            data.frame_entities.push_back(id);

            t.set_static(false);
            t.set_scaling(vec3(0.0014));
            t.set_orientation(90, vec3(0,1,0));
            t.set_position(vec3(0.45,0.2,6));

            // Shadow realm is for the transient people.
            for(auto& vg: model)
                vg.mat.transient = true;
        });
    }

    for(size_t i = 1; i < opt.scene_paths.size(); ++i)
    {
        const std::string& path = opt.scene_paths[i];
        fs::path fsp(path);
        data.static_scene.emplace_back(load_gltf(
            dev, *data.real_world, path, opt.force_single_sided, opt.force_double_sided
        ));
    }

    if(opt.envmap.size())
    {
        entity id = data.real_world->add();
        data.real_world->emplace<environment_map>(id, dev, opt.envmap);
    }

    camera cam;
    cam.perspective(90, opt.width/(float)opt.height, 0.1f, 300.0f);
    if(opt.camera_clip_range.near > 0)
        cam.set_near(opt.camera_clip_range.near);
    if(opt.camera_clip_range.far > 0)
        cam.set_far(opt.camera_clip_range.far);
    cam.set_aspect(opt.aspect_ratio > 0 ? opt.aspect_ratio : opt.width/(float)opt.height);
    if(opt.fov)
        cam.set_fov(opt.fov);

    data.real_world->add(
        std::move(cam),
        transformable(vec3(0,2,0)),
        camera_metadata{true, 0, true}
    );

    if(opt.animation_flag)
        play(*data.real_world, opt.animation, !opt.replay, opt.animation == "");

    return data;
}

void run_demo(context& ctx, demo_scene_data& sd, options& opt)
{
    scene& s = *sd.real_world;

    entity cam_id = INVALID_ENTITY;
    s.foreach([&](entity id, camera_metadata& md){
        if(md.enabled) cam_id = id;
    });

    transformable* cam = s.get<transformable>(cam_id);
    std::unique_ptr<renderer> rr;

    float speed = 1.0f;
    vec3 euler = cam->get_orientation_euler();
    float pitch = euler.x;
    float yaw = euler.y;
    float roll = euler.z;
    float sensitivity = 0.2;
    bool paused = false;
    int camera_index = 0;

    if(openxr* xr = dynamic_cast<openxr*>(&ctx))
    {
        xr->setup_xr_surroundings(s, cam);
        sensitivity = 0;
    }

    s.foreach([&](camera_metadata& md){
        md.actively_rendered = opt.spatial_reprojection.count(md.index);
    });
    set_camera_jitter(s, get_camera_jitter_sequence(opt.taa.sequence_length, ctx.get_size()));

    std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    float delta = 0.0f;
    double total_time = 0.0f;
    bool focused = true;
    bool camera_locked = false;
    bool recreate_renderer = true;
    bool crash_on_exception = true;
    bool camera_moved = false;

    int last_update_frame = 0;
    entity last_anim_entity = INVALID_ENTITY;
    constexpr float frame_duration = 1.0f/25.0f;

    ivec3 camera_movement = ivec3(0);
    std::string command_line;
    while(opt.running)
    {
        camera_moved = false;
        if(nonblock_getline(command_line))
        {
            if(parse_command(command_line.c_str(), opt))
            {
                recreate_renderer = true;
                camera_moved = true;
            }
        }

        const int frame_index = (int)(total_time / frame_duration) % sd.frame_entities.size();
        if(frame_index != last_update_frame)
        {
            entity new_ent = sd.frame_entities[frame_index];

            if(last_anim_entity != INVALID_ENTITY)
            {
                s.remove(last_anim_entity);
                last_anim_entity = INVALID_ENTITY;
            }
            last_anim_entity = s.copy(*sd.shadow_realm, new_ent);
            last_update_frame = frame_index;

            if(rr) rr->set_scene(&s);
        }

        if(recreate_renderer)
        {
            try
            {
                set_camera_jitter(s, get_camera_jitter_sequence(opt.taa.sequence_length, ctx.get_size()));
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
                    cam->set_global_position();
                    cam->set_global_orientation();
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

        if(!camera_locked)
        {
            camera_movement = clamp(camera_movement, ivec3(-1), ivec3(1));
            if(camera_movement != ivec3(0))
                camera_moved = true;
            cam->translate_local(vec3(camera_movement)*delta*speed);
            cam->set_orientation(pitch, yaw, roll);
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

        update(s, paused || !opt.animation_flag ? 0 : delta * 1000000);

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
            else break;
        }
        if(opt.timing) ctx.get_timing().print_last_trace(opt.trace);

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end-start;
        delta = elapsed.count();
        if(!paused)
            total_time += delta;
        start = end;
    }

    // Ensure everything is finished before going to destructors.
    ctx.sync();
}

int main(int, char** argv) try
{
    std::ios_base::sync_with_stdio(false);

    tr::options opt;
    tr::parse_command_line_options(argv, opt);

    // Initialize log timer.
    tr::get_initial_time();
    std::optional<std::ofstream> timing_output_file;

    if(opt.silent)
    {
        tr::enabled_log_types[(uint32_t)tr::log_type::GENERAL] = false;
        tr::enabled_log_types[(uint32_t)tr::log_type::WARNING] = false;
    }

    if(opt.timing_output.size() != 0)
    {
        timing_output_file.emplace(opt.timing_output, std::ios::binary|std::ios::trunc);
        tr::log_output_streams[(uint32_t)tr::log_type::TIMING] = &timing_output_file.value();
    }

    std::unique_ptr<tr::context> ctx(tr::create_context(opt));

    demo_scene_data sd = load_demo_scenes(*ctx, opt);

    run_demo(*ctx, sd, opt);

    return 0;
}
catch (std::runtime_error& e)
{
    // Can't use TR_ERR here, because the logger may not yet be initialized or
    // it's output file may already be closed.
    if (strlen(e.what())) std::cerr << e.what() << "\n" << std::endl;
    return 1;
}
