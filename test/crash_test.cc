#include "tauray.hh"
#include <iostream>

int main() try
{
    tr::options opt;

    opt.renderer = TEST_RENDERER;
    opt.scene_paths = {"test/test.glb"};
    opt.headless = "test";
    opt.filetype = tr::headless::EMPTY;
    opt.replay = true;

    std::unique_ptr<tr::context> ctx(tr::create_context(opt));
    tr::scene_data sd = tr::load_scenes(*ctx, opt);
    tr::run(*ctx, sd, opt);
    return 0;
}
catch(tr::option_parse_error& e)
{
    if(strlen(e.what())) std::cerr << e.what() << "\n" << std::endl;
    return 1;
}

