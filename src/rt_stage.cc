#include "rt_stage.hh"
#include "mesh.hh"
#include "shader_source.hh"
#include "placeholders.hh"
#include "scene.hh"
#include "environment_map.hh"
#include "misc.hh"
#include "texture.hh"
#include "sampler.hh"

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
    const rt_shader_sources& src,
    const options& opt,
    vk::SpecializationInfo specialization
){
    rt_pipeline::options state = {
        src,
        {
            {"vertices", (uint32_t)opt.max_instances},
            {"indices", (uint32_t)opt.max_instances},
            {"textures", (uint32_t)opt.max_samplers},
        },
        1, {}, false
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
    device_data& dev,
    const options& opt,
    const std::string& timer_name,
    unsigned pass_count
):  stage(dev),
    opt(opt),
    pass_count(pass_count),
    rt_timer(dev, timer_name),
    cur_scene(nullptr),
    sampling_data(
        dev, sizeof(sampling_data_buffer),
        vk::BufferUsageFlagBits::eUniformBuffer
    ),
    sampling_frame_counter_increment(1),
    sample_counter(0)
{
}

void rt_stage::set_scene(scene* s)
{
    cur_scene = s;
    if(s->get_instance_count() > opt.max_instances)
        throw std::runtime_error(
            "The scene has more meshes than this pipeline can support!"
        );
    init_scene_resources();
    record_command_buffers();
}

scene* rt_stage::get_scene()
{
    return cur_scene;
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
}

void rt_stage::init_scene_resources() {}

void rt_stage::init_descriptors(basic_pipeline& pp)
{
    // Init descriptor set references to some placeholder value to silence
    // the validation layer (these should never actually be accessed)
    pp.update_descriptor_set({
        {"vertices", opt.max_instances},
        {"indices", opt.max_instances},
        {"textures", opt.max_samplers}
    });
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        pp.update_descriptor_set({
            {"sampling_data", {*sampling_data, 0, VK_WHOLE_SIZE}}
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
        vk::CommandBuffer cb = begin_graphics();
        rt_timer.begin(cb, i);
        sampling_data.upload(i, cb);
        for(size_t j = 0; j < pass_count; ++j)
            record_command_buffer(cb, i, j);
        rt_timer.end(cb, i);
        end_graphics(cb, i);
    }
}

}
