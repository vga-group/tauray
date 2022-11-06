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
};

}

namespace tr
{

gfx_pipeline::pipeline_state rt_stage::get_common_state(
    uvec2 output_size,
    uvec4 viewport,
    const shader_sources& src,
    const options& opt,
    vk::SpecializationInfo specialization
){
    gfx_pipeline::pipeline_state state = {
        output_size,
        viewport,
        src,
        {
            {"vertices", (uint32_t)opt.max_meshes},
            {"indices", (uint32_t)opt.max_meshes},
            {"textures", (uint32_t)opt.max_samplers},
        },
        mesh::get_bindings(),
        mesh::get_attributes(),
        {}, {},
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
}

rt_stage::rt_stage(
    device_data& dev,
    const gfx_pipeline::pipeline_state& state,
    const options& opt,
    const std::string& timer_name,
    unsigned pass_count
):  stage(dev),
    gfx(dev, state),
    opt(opt),
    pass_count(pass_count),
    rt_timer(dev, timer_name),
    cur_scene(nullptr),
    sampling_data(
        dev, sizeof(sampling_data_buffer),
        vk::BufferUsageFlagBits::eUniformBuffer
    )
{
    init_resources();
}

void rt_stage::set_scene(scene* s)
{
    cur_scene = s;
    if(s->get_meshes().size() > opt.max_meshes)
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

void rt_stage::update(uint32_t frame_index)
{
    sampling_data.map<sampling_data_buffer>(
        frame_index,
        [&](sampling_data_buffer* suni){
            suni->size = sampling_target_size;
            suni->sampling_start_counter = dev->ctx->get_frame_counter() * sampling_frame_counter_increment;
            if(opt.rng_seed != 0)
                suni->sampling_start_counter += pcg(opt.rng_seed);
        }
    );
}

void rt_stage::init_resources()
{
    // Init descriptor set references to some placeholder value to silence
    // the validation layer (these should never actually be accessed)
    gfx.update_descriptor_set({
        {"vertices", opt.max_meshes},
        {"indices", opt.max_meshes},
        {"textures", opt.max_samplers}
    });
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        gfx.update_descriptor_set({
            {"sampling_data", {*sampling_data, 0, VK_WHOLE_SIZE}}
        }, i);
    }
}

void rt_stage::init_scene_resources()
{
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
