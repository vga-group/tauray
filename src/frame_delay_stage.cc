#include "frame_delay_stage.hh"

namespace tr
{

frame_delay_stage::frame_delay_stage(
    device_data& dev,
    gbuffer_target& input_features
): stage(dev), delay_timer(dev, "frame_delay")
{
    textures.reset(new gbuffer_texture(
        dev, input_features.get_size(), input_features.get_layer_count(),
        input_features.get_msaa()
    ));
    gbuffer_spec spec = input_features.get_spec();
    spec.set_all_usage(
        vk::ImageUsageFlagBits::eStorage|
        vk::ImageUsageFlagBits::eSampled|
        vk::ImageUsageFlagBits::eTransferDst
    );
    spec.depth_usage = vk::ImageUsageFlagBits::eSampled|
        vk::ImageUsageFlagBits::eTransferDst;
    textures->add(spec);
    output_features = textures->get_array_target();

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();
        delay_timer.begin(cb, i);

        input_features.visit([&](render_target& target){
            target.transition_layout_temporary(cb, i, vk::ImageLayout::eTransferSrcOptimal);
        });
        output_features.visit([&](render_target& target){
            target.transition_layout_temporary(cb, i, vk::ImageLayout::eTransferDstOptimal);
        });

        for(size_t j = 0; j < MAX_GBUFFER_ENTRIES; ++j)
        {
            if(!output_features[j]) continue;
            uvec3 size = uvec3(input_features[j].get_size(), 1);
            cb.copyImage(
                input_features[j][i].image,
                vk::ImageLayout::eTransferSrcOptimal,
                output_features[j][i].image,
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageCopy(
                    input_features[j].get_layers(),
                    {0,0,0},
                    output_features[j].get_layers(),
                    {0,0,0},
                    {size.x, size.y, size.z}
                )
            );
        }

        output_features.visit([&](render_target& target){
            vk::ImageLayout old_layout = target.get_layout();
            target.set_layout(vk::ImageLayout::eTransferDstOptimal);
            target.transition_layout_temporary(cb, i, vk::ImageLayout::eGeneral);
            target.set_layout(old_layout);
        });

        delay_timer.end(cb, i);
        end_compute(cb, i);
    }
    input_features.visit([&](render_target& target){
        target.set_layout(vk::ImageLayout::eTransferSrcOptimal);
    });
    output_features.visit([&](render_target& target){
        target.set_layout(vk::ImageLayout::eGeneral);
    });
}

gbuffer_target frame_delay_stage::get_output()
{
    return output_features;
}

}
