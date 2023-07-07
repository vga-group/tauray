#include "sh_renderer.hh"
#include "sh_grid.hh"
#include "scene.hh"

namespace tr
{

sh_renderer::sh_renderer(context& ctx, const options& opt)
: ctx(&ctx), opt(opt)
{
}

sh_renderer::~sh_renderer()
{
}

void sh_renderer::set_scene(scene* s)
{
    cur_scene = s;
    per_grid.clear();

    device& dev = ctx->get_display_device();

    for(sh_grid* s: cur_scene->get_sh_grids())
    {
        sh_grid_targets.emplace(
            s, s->create_target_texture(dev, opt.samples_per_probe)
        );
        sh_grid_textures.emplace(s, s->create_texture(dev));
    }

    cur_scene->set_sh_grid_textures(&sh_grid_textures);
}

texture& sh_renderer::get_sh_grid_texture(sh_grid* grid)
{
    return sh_grid_textures.at(grid);
}

dependencies sh_renderer::render(dependencies deps)
{
    // This has to be done here, unfortunately. The sh_renderer::set_scene
    // must be called _before_ scene_update_stage::set_scene, but the rest of
    // the stages need to be built _after_ scene_update_stage::set_scene.
    if(per_grid.size() == 0)
    {
        device& dev = ctx->get_display_device();
        int grid_index = 0;
        for(sh_grid* s:  cur_scene->get_sh_grids())
        {
            texture* output_grids = &sh_grid_targets.at(s);
            texture* compact_grids = &sh_grid_textures.at(s);

            sh_path_tracer_stage::options sh_opt = opt;
            sh_opt.sh_grid_index = grid_index++;
            sh_opt.sh_order = s->get_order();

            s->get_target_sampling_info(
                ctx->get_display_device(),
                sh_opt.samples_per_probe,
                sh_opt.samples_per_invocation
            );

            per_grid_data& p = per_grid.emplace_back();
            p.pt.reset(new sh_path_tracer_stage(
                dev, *output_grids, vk::ImageLayout::eGeneral, sh_opt
            ));
            p.pt->set_scene(cur_scene);

            p.compact.reset(new sh_compact_stage(
                dev, *output_grids, *compact_grids
            ));
        }
    }

    for(auto& p: per_grid)
    {
        deps = p.pt->run(deps);
        deps = p.compact->run(deps);
    }

    return deps;
}

}

