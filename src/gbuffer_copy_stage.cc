#include "gbuffer_copy_stage.hh"
#include "misc.hh"

namespace tr
{

gbuffer_copy_stage::gbuffer_copy_stage(
    device& dev,
    gbuffer_target& in,
    gbuffer_target& out,
    int force_input_layer,
    int force_output_layer
): single_device_stage(dev), copy_timer(dev, "copy gbuffer")
{
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();
        copy_timer.begin(cb, dev.id, i);

        in.visit([&](render_target& target){
            transition_image_layout(
                cb, target.image, target.format, target.layout, vk::ImageLayout::eTransferSrcOptimal, 0, 1,
                force_input_layer >= 0 ? force_input_layer : target.base_layer,
                force_input_layer >= 0 ? 1 : target.layer_count,
                true, true
            );
        });
        out.visit([&](render_target& target){
            transition_image_layout(
                cb, target.image, target.format, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 0, 1,
                force_output_layer >= 0 ? force_output_layer : target.base_layer,
                force_output_layer >= 0 ? 1 : target.layer_count,
                true, true
            );
        });

        for(size_t j = 0; j < MAX_GBUFFER_ENTRIES; ++j)
        {
            if(!out[j] || !in[j]) continue;
            uvec3 size = uvec3(in[j].size, 1);
            cb.copyImage(
                in[j].image,
                vk::ImageLayout::eTransferSrcOptimal,
                out[j].image,
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageCopy(
                    force_input_layer >= 0 ? vk::ImageSubresourceLayers(
                        deduce_aspect_mask(in[j].format),
                        0, force_input_layer, 1
                    ) : in[j].get_layers(),
                    {0,0,0},
                    force_output_layer >= 0 ? vk::ImageSubresourceLayers(
                        deduce_aspect_mask(out[j].format),
                        0, force_output_layer, 1
                    ) : out[j].get_layers(),
                    {0,0,0},
                    {size.x, size.y, size.z}
                )
            );
        }

        in.visit([&](render_target& target){
            transition_image_layout(
                cb, target.image, target.format, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral, 0, 1,
                force_input_layer >= 0 ? force_input_layer : target.base_layer,
                force_input_layer >= 0 ? 1 : target.layer_count,
                true, true
            );
        });
        out.visit([&](render_target& target){
            transition_image_layout(
                cb, target.image, target.format, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral, 0, 1,
                force_output_layer >= 0 ? force_output_layer : target.base_layer,
                force_output_layer >= 0 ? 1 : target.layer_count,
                true, true
            );
        });

        copy_timer.end(cb, dev.id, i);
        end_compute(cb, i);
    }
    in.visit([&](render_target& target){
        target.layout = vk::ImageLayout::eGeneral;
    });
    out.visit([&](render_target& target){
        target.layout = vk::ImageLayout::eGeneral;
    });
}

}
