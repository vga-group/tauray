#include "rt_stage.hh"
#include "mesh.hh"
#include "shader_source.hh"
#include "placeholders.hh"
#include "scene_stage.hh"
#include "environment_map.hh"
#include "misc.hh"
#include "texture.hh"
#include "sampler.hh"
#include "log.hh"

namespace
{
using namespace tr;

struct sampling_data_buffer
{
    puvec3 size;
    uint sampling_start_counter;
    uint rng_seed;
};

}

namespace tr
{

rt_pipeline::options rt_stage::get_common_options(
    scene_stage& ss,
    const rt_shader_sources& src,
    const options&,
    vk::SpecializationInfo specialization
){
    rt_pipeline::options state = {
        src,
        {},
        1, {}, false,
        {&ss.get_descriptors()}
    };
    state.specialization = specialization;
    return state;
}

void rt_stage::get_common_defines(
    std::map<std::string, std::string>& defines,
    const options& opt
){
    switch(opt.local_sampler)
    {
    case sampler_type::SOBOL_Z_ORDER_2D:
        defines["USE_SOBOL_Z_ORDER_SAMPLING"];
        defines["SOBOL_Z_ORDER_CURVE_DIMS"] = "2";
        break;
    case sampler_type::SOBOL_Z_ORDER_3D:
        defines["USE_SOBOL_Z_ORDER_SAMPLING"];
        defines["SOBOL_Z_ORDER_CURVE_DIMS"] = "3";
        break;
    case sampler_type::SOBOL_OWEN:
        defines["USE_SOBOL_OWEN_SAMPLING"];
        break;
    default:
        break;
    }

    if(opt.pre_transformed_vertices)
        defines["PRE_TRANSFORMED_VERTICES"];
}

rt_stage::rt_stage(
    device& dev,
    scene_stage& ss,
    const options& opt,
    const std::string& timer_name,
    unsigned pass_count
):  single_device_stage(dev),
    ss(&ss),
    opt(opt),
    pass_count(pass_count),
    rt_timer(dev, timer_name),
    sampling_data(
        dev, sizeof(sampling_data_buffer),
        vk::BufferUsageFlagBits::eUniformBuffer
    ),
    sampling_frame_counter_increment(1),
    sample_counter(0),
    scene_state_counter(0),
    force_refresh(true)
{
}

void rt_stage::set_local_sampler_parameters(
    uvec3 target_size,
    uint32_t frame_counter_increment
){
    sampling_target_size = target_size;
    sampling_frame_counter_increment = frame_counter_increment;
}

void rt_stage::reset_sample_counter()
{
    sample_counter = 0;
}

void rt_stage::update(uint32_t frame_index)
{
    sampling_data.map<sampling_data_buffer>(
        frame_index,
        [&](sampling_data_buffer* suni){
            suni->size = sampling_target_size;
            suni->sampling_start_counter = sample_counter;
            suni->rng_seed = opt.rng_seed != 0 ? pcg(opt.rng_seed) : 0;
        }
    );
    sample_counter += sampling_frame_counter_increment;

    if(ss->check_update(scene_stage::GEOMETRY|scene_stage::LIGHT|scene_stage::ENVMAP, scene_state_counter))
    {
        init_scene_resources();
        record_command_buffers();
        force_refresh = false;
    }

    if(force_refresh)
    {
        record_command_buffers();
        force_refresh = false;
    }
}

void rt_stage::init_scene_resources() {}

void rt_stage::init_descriptors(basic_pipeline& pp)
{
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        pp.update_descriptor_set({
            {"sampling_data", {sampling_data[dev->id], 0, VK_WHOLE_SIZE}}
        }, i);
    }
}

unsigned rt_stage::get_pass_count() const
{
    return pass_count;
}

void rt_stage::record_command_buffers()
{
    clear_commands();
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        size_t pass_count_left = pass_count;
        while(pass_count_left != 0)
        {
            vk::CommandBuffer cb = begin_graphics();
            if(pass_count_left == pass_count)
                rt_timer.begin(cb, dev->id, i);
            sampling_data.upload(dev->id, i, cb);
            size_t local_pass_count = opt.max_passes_per_command_buffer == 0 ?
                pass_count :
                min(pass_count_left, opt.max_passes_per_command_buffer);
            for(size_t j = 0; j < local_pass_count; ++j)
                record_command_buffer(cb, i, (pass_count - pass_count_left) + j, j == 0);
            pass_count_left -= local_pass_count;
            if(pass_count_left == 0)
                rt_timer.end(cb, dev->id, i);
            end_graphics(cb, i);
        }
    }
}

void rt_stage::force_command_buffer_refresh()
{
    force_refresh = true;
}

}
