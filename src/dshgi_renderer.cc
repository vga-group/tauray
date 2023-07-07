#include "dshgi_renderer.hh"
#include "sh_grid.hh"
#include "scene.hh"

namespace tr
{

dshgi_renderer::dshgi_renderer(context& ctx, const options& opt)
: raster_renderer(ctx, opt), opt(opt)
{
    if(auto rtype = std::get_if<sh_renderer::options>(&opt.sh_source))
        sh.reset(new sh_renderer(ctx, *rtype));
    else if(auto rtype = std::get_if<dshgi_client::options>(&opt.sh_source))
        client.reset(new dshgi_client(ctx, *rtype));
}

dshgi_renderer::~dshgi_renderer()
{
}

void dshgi_renderer::set_scene(scene* s)
{
    if(sh) sh->set_scene(s);
    if(client) client->set_scene(s);
    raster_renderer::set_scene(s);
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
