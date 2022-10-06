#ifndef TAURAY_DSHGI_RENDERER_HH
#define TAURAY_DSHGI_RENDERER_HH
#include "context.hh"
#include "texture.hh"
#include "raster_renderer.hh"
#include "sh_renderer.hh"
#include "dshgi_client.hh"
#include <variant>

namespace tr
{

class dshgi_renderer: public raster_renderer
{
public:
    struct options: raster_renderer::options
    {
        std::variant<
            sh_renderer::options,
            dshgi_client::options
        > sh_source;
    };

    dshgi_renderer(context& ctx, const options& opt);
    dshgi_renderer(const dshgi_renderer& other) = delete;
    dshgi_renderer(dshgi_renderer&& other) = delete;
    ~dshgi_renderer();

    void set_scene(scene* s) override;
    void render() override;

private:
    options opt;
    std::unique_ptr<sh_renderer> sh;
    std::unique_ptr<dshgi_client> client;
};

}

#endif

