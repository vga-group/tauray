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
}

sh_renderer::~sh_renderer()
{
}

void sh_renderer::update_grids()
{
    device& dev = ctx->get_display_device();

    per_grid.clear();
    ss->get_scene()->foreach([&](entity id, sh_grid& s){
        sh_grid_targets.emplace(
            &s, s.create_target_texture(dev, opt.samples_per_probe)
        );

        texture* output_grids = &sh_grid_targets.at(&s);
        const texture* compact_grids = &ss->get_sh_grid_textures().at(&s);

        sh_path_tracer_stage::options sh_opt = opt;
        sh_opt.sh_grid_id = id;
        sh_opt.sh_order = s.get_order();

        s.get_target_sampling_info(
            ctx->get_display_device(),
            sh_opt.samples_per_probe,
            sh_opt.samples_per_invocation
        );

        per_grid_data& p = per_grid.emplace_back();
        p.pt.reset(new sh_path_tracer_stage(
            dev, *ss, *output_grids, vk::ImageLayout::eGeneral, sh_opt
        ));

        p.compact.reset(new sh_compact_stage(
            dev, *output_grids, *const_cast<texture*>(compact_grids)
        ));
    });
}

dependencies sh_renderer::render(dependencies deps)
{
    if(ss->check_update(scene_stage::LIGHT, scene_state_counter))
        update_grids();

    for(auto& p: per_grid)
    {
        deps = p.pt->run(deps);
        deps = p.compact->run(deps);
    }

    return deps;
}

}

