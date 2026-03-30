#include "tauray.hh"
#include <iostream>
#include <fstream>

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

    if (opt.auto_spp > 0.0f)
    {
        std::unique_ptr<tr::context> ctx(tr::create_context(opt));

        tr::scene_data sd = tr::load_scenes(*ctx, opt);
        tr::search_matching_spp(*ctx, sd, opt, opt.auto_spp);
        std::this_thread::sleep_for(std::chrono::seconds(30));
    }

    std::unique_ptr<tr::context> ctx(tr::create_context(opt));

    tr::scene_data sd = tr::load_scenes(*ctx, opt);

    tr::run(*ctx, sd, opt);
    return opt.samples_per_pixel;
}
catch (std::runtime_error& e)
{
    // Can't use TR_ERR here, because the logger may not yet be initialized or
    // it's output file may already be closed.
    if (strlen(e.what())) std::cerr << e.what() << "\n" << std::endl;
    return 1;
}
