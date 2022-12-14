#include "tauray.hh"
#include <iostream>

int main(int, char** argv) try
{
    std::ios_base::sync_with_stdio(false);

    tr::options opt;
    tr::parse_command_line_options(argv, opt);
    std::unique_ptr<tr::context> ctx(tr::create_context(opt));

    tr::scene_data sd = tr::load_scenes(*ctx, opt);

    tr::run(*ctx, sd, opt);

    return 0;
}
catch (std::runtime_error& e)
{
    if (strlen(e.what())) std::cerr << e.what() << "\n" << std::endl;

    tr::print_help(argv[0]);
    return 1;
}
