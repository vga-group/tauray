#include "dshgi_renderer.hh"
#include "sh_grid.hh"
#include "scene.hh"

namespace tr
{

dshgi_renderer::dshgi_renderer(context& ctx, const options& opt)
: raster_renderer(ctx, opt), opt(opt)
{
    if(auto rtype = std::get_if<sh_renderer::options>(&opt.sh_source))
        sh.reset(new sh_renderer(ctx.get_display_device(), *scene_update, *rtype));
    else if(auto rtype = std::get_if<dshgi_client::options>(&opt.sh_source))
        client.reset(new dshgi_client(ctx, *scene_update, *rtype));
}

dshgi_renderer::~dshgi_renderer()
{
}

void dshgi_renderer::render()
{
    if(client && client->refresh())
        set_scene(cur_scene);

    dependencies deps(ctx->begin_frame());

    deps = scene_update->run(deps);
    if(sh) deps = sh->render(deps);
    else if(client) deps = client->render(deps);
    deps = render_core(deps);

    ctx->end_frame(deps);
}

}
