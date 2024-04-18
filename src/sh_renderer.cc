#include "sh_renderer.hh"
#include "sh_grid.hh"
#include "scene_stage.hh"
#include "log.hh"

namespace tr
{

sh_renderer::sh_renderer(
    device_mask dev,
    scene_stage& ss,
    const options& opt
): dev(dev), opt(opt), ss(&ss)
{
}

sh_renderer::~sh_renderer()
{
}

void sh_renderer::update_grids()
{
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
            dev,
            sh_opt.samples_per_probe,
            sh_opt.samples_per_invocation
        );

        per_grid_data& p = per_grid.emplace_back();
        for(device& d: dev)
        {
            p.pt.emplace_back(new sh_path_tracer_stage(
                d, *ss, *output_grids, vk::ImageLayout::eGeneral, sh_opt
            ));

            p.compact.emplace_back(new sh_compact_stage(
                d, *output_grids, *const_cast<texture*>(compact_grids)
            ));
        }
    });
}

dependencies sh_renderer::render(dependencies deps)
{
    if(ss->check_update(scene_stage::LIGHT, scene_state_counter))
        update_grids();

    for(auto& p: per_grid)
    {
        for(auto& s: p.pt) deps = s->run(deps);
        for(auto& s: p.compact) deps = s->run(deps);
    }

    return deps;
}

}

