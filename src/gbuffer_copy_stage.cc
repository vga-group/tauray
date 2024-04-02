#include "gbuffer_copy_stage.hh"

namespace tr
{

gbuffer_copy_stage::gbuffer_copy_stage(
    device& dev,
    gbuffer_target& in,
    gbuffer_target& out
): single_device_stage(dev), copy_timer(dev, "copy gbuffer")
{
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();
        copy_timer.begin(cb, dev.id, i);

        in.visit([&](render_target& target){
            target.transition_layout_temporary(cb, vk::ImageLayout::eTransferSrcOptimal, true, true);
        });
        out.visit([&](render_target& target){
            target.layout = vk::ImageLayout::eUndefined;
            target.transition_layout_temporary(cb, vk::ImageLayout::eTransferDstOptimal, true, true);
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
                    in[j].get_layers(),
                    {0,0,0},
                    out[j].get_layers(),
                    {0,0,0},
                    {size.x, size.y, size.z}
                )
            );
        }

        in.visit([&](render_target& target){
            vk::ImageLayout old_layout = target.layout;
            target.layout = vk::ImageLayout::eTransferSrcOptimal;
            target.transition_layout_temporary(cb, vk::ImageLayout::eGeneral, true, true);
            target.layout = old_layout;
        });
        out.visit([&](render_target& target){
            vk::ImageLayout old_layout = target.layout;
            target.layout = vk::ImageLayout::eTransferDstOptimal;
            target.transition_layout_temporary(cb, vk::ImageLayout::eGeneral, true, true);
            target.layout = old_layout;
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
