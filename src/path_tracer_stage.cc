#include "path_tracer_stage.hh"
#include "light_tree_stage.hh"
#include "radiance_cascades_stage.hh"
#include "scene_stage.hh"
#include "misc.hh"
#include "environment_map.hh"

namespace
{
using namespace tr;

struct push_constant_buffer
{
    uint32_t samples;
    uint32_t previous_samples;
    float min_ray_dist;
    float indirect_clamping;
    float film_radius;
    float russian_roulette_delta;
    int antialiasing;
    float regularization_gamma;
};

// The minimum maximum size for push constant buffers is 128 bytes in vulkan.
static_assert(sizeof(push_constant_buffer) <= 128);

}

namespace tr
{

path_tracer_stage::path_tracer_stage(
    device& dev,
    scene_stage& ss,
    const gbuffer_target& output_target,
    const options& opt
):  rt_camera_stage(
        dev, ss, output_target, opt, "path tracing " + std::to_string(opt.samples_per_pixel) + " spp",
        opt.samples_per_pixel / opt.samples_per_pass
    ),
    desc(dev),
    pt_pipeline(dev),
    opt(opt)
{
    std::map<std::string, std::string> defines;
    defines["MAX_BOUNCES"] = std::to_string(opt.max_ray_depth);
    defines["SAMPLES_PER_PASS"] = std::to_string(opt.samples_per_pass);

    if(opt.russian_roulette_delta > 0)
        defines["USE_RUSSIAN_ROULETTE"];

    if(opt.use_shadow_terminator_fix)
        defines["USE_SHADOW_TERMINATOR_FIX"];

    if(opt.use_white_albedo_on_first_bounce)
        defines["USE_WHITE_ALBEDO_ON_FIRST_BOUNCE"];

    if(opt.hide_lights)
        defines["HIDE_LIGHTS"];

    if(opt.transparent_background)
        defines["USE_TRANSPARENT_BACKGROUND"];

    if(opt.regularization_gamma != 0.0f)
        defines["PATH_SPACE_REGULARIZATION"];

    if(opt.depth_of_field)
        defines["USE_DEPTH_OF_FIELD"];

    int set_index = 2;

    if(opt.rc_source)
        defines["RADIANCE_CASCADES_SET"] = std::to_string(set_index++);
    if(opt.light_tree_source)
        defines["LIGHT_TREE_SET"] = std::to_string(set_index++);

#define TR_GBUFFER_ENTRY(name, ...)\
    if(output_target.name) defines["USE_"+to_uppercase(#name)+"_TARGET"];
    TR_GBUFFER_ENTRIES
#undef TR_GBUFFER_ENTRY

    add_defines(opt.sampling_weights, defines);
    add_defines(opt.film, defines);
    add_defines(opt.mis_mode, defines);
    add_defines(opt.bounce_mode, defines);
    add_defines(opt.tri_light_mode, defines);

    if(opt.rc_source)
        opt.rc_source->add_defines(defines);

    if(opt.light_tree_source)
        opt.light_tree_source->add_defines(defines);

    get_common_defines(defines);

    shader_source src = {"shader/path_tracer.comp", defines};
    desc.add(src);
    std::vector<tr::descriptor_set_layout*> layout = {&desc, &ss.get_descriptors()};
    if(opt.rc_source)
        layout.push_back(&opt.rc_source->get_descriptors());
    if(opt.light_tree_source)
        layout.push_back(&opt.light_tree_source->get_descriptors());
    pt_pipeline.init(src, layout);
}

void path_tracer_stage::record_command_buffer_pass(
    vk::CommandBuffer cb,
    uint32_t,
    uint32_t pass_index,
    uvec3 expected_dispatch_size,
    bool first_in_command_buffer
){
    if(first_in_command_buffer)
    {
        pt_pipeline.bind(cb);
        get_descriptors(desc);
        pt_pipeline.push_descriptors(cb, desc, 0);
        pt_pipeline.set_descriptors(cb, ss->get_descriptors(), 0, 1);
        int set_index = 2;
        if(opt.rc_source)
            pt_pipeline.set_descriptors(cb, opt.rc_source->get_descriptors(), 0, set_index++);
        if(opt.light_tree_source)
            pt_pipeline.set_descriptors(cb, opt.light_tree_source->get_descriptors(), 0, set_index++);
    }

    push_constant_buffer control;

    control.film_radius = opt.film_radius;
    control.russian_roulette_delta = opt.russian_roulette_delta;
    control.min_ray_dist = opt.min_ray_dist;
    control.indirect_clamping = opt.indirect_clamping;
    control.regularization_gamma = opt.regularization_gamma;

    control.previous_samples = pass_index * opt.samples_per_pass;
    control.samples = opt.samples_per_pass;
    control.antialiasing = opt.film != film_filter::POINT ? 1 : 0;

    pt_pipeline.push_constants(cb, control);

    uvec3 wg = uvec3(
        (expected_dispatch_size.x+7u)/8u,
        (expected_dispatch_size.y+7u)/8u,
        expected_dispatch_size.z
    );
    cb.dispatch(wg.x, wg.y, wg.z);
}

}
