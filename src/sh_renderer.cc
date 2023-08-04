#include "sh_renderer.hh"
#include "sh_grid.hh"
#include "scene_stage.hh"

namespace tr
{

sh_renderer::sh_renderer(
    context& ctx,
    scene_stage& ss,
    const options& opt
): ctx(&ctx), opt(opt), ss(&ss)
{

    device& dev = ctx.get_display_device();

    // TODO: Move the actual textures over to scene_stage? That way, you could
    // change the grids while the program is running, and it would avoid the
    // set_sh_grid_textures weirdness.
    per_grid.clear();
    int grid_index = 0;
    for(sh_grid* s: ss.get_scene()->get_sh_grids())
    {
        sh_grid_targets.emplace(
            s, s->create_target_texture(dev, opt.samples_per_probe)
        );
        sh_grid_textures.emplace(s, s->create_texture(dev));

        texture* output_grids = &sh_grid_targets.at(s);
        texture* compact_grids = &sh_grid_textures.at(s);

        sh_path_tracer_stage::options sh_opt = opt;
        sh_opt.sh_grid_index = grid_index++;
        sh_opt.sh_order = s->get_order();

        s->get_target_sampling_info(
            ctx.get_display_device(),
            sh_opt.samples_per_probe,
            sh_opt.samples_per_invocation
        );

        per_grid_data& p = per_grid.emplace_back();
        p.pt.reset(new sh_path_tracer_stage(
            dev, ss, *output_grids, vk::ImageLayout::eGeneral, sh_opt
        ));

        p.compact.reset(new sh_compact_stage(
            dev, *output_grids, *compact_grids
        ));
    }

    ss.set_sh_grid_textures(&sh_grid_textures);
}

sh_renderer::~sh_renderer()
{
}

texture& sh_renderer::get_sh_grid_texture(sh_grid* grid)
{
    return sh_grid_textures.at(grid);
}

dependencies sh_renderer::render(dependencies deps)
{
    for(auto& p: per_grid)
    {
        deps = p.pt->run(deps);
        deps = p.compact->run(deps);
    }

    return deps;
}

}

