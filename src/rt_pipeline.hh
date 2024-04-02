#ifndef TAURAY_RT_PIPELINE_HH
#define TAURAY_RT_PIPELINE_HH
#include "context.hh"
#include "texture.hh"
#include "basic_pipeline.hh"
#include "render_target.hh"
#include <optional>

namespace tr
{

class rt_pipeline: public basic_pipeline
{
public:
    rt_pipeline(device& dev);

    void init(
        rt_shader_sources src,
        std::vector<tr::descriptor_set_layout*> layout,
        int max_recursion_depth = 1,
        vk::SpecializationInfo specialization = {}
    );

    void trace_rays(vk::CommandBuffer buf, uvec3 size);

protected:
    vkm<vk::Buffer> sbt_buffer;
    vk::StridedDeviceAddressRegionKHR rgen_sbt;
    vk::StridedDeviceAddressRegionKHR rchit_sbt;
    vk::StridedDeviceAddressRegionKHR rmiss_sbt;
    vk::StridedDeviceAddressRegionKHR rcallable_sbt;
};

}

#endif
