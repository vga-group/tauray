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
    struct options
    {
        rt_shader_sources src;

        // The actual bindings are automatically determined from shader
        // source, so you can only define array lengths here (those can't
        // be deduced since they can change).
        binding_array_length_info binding_array_lengths;

        int max_recursion_depth = 1;

        vk::SpecializationInfo specialization = {};
        bool use_push_descriptors = false;
    };

    rt_pipeline(device_data& dev, const options& state);

    void trace_rays(vk::CommandBuffer buf, uvec3 size);

protected:
    vkm<vk::Buffer> sbt_buffer;
    vk::StridedDeviceAddressRegionKHR rgen_sbt;
    vk::StridedDeviceAddressRegionKHR rchit_sbt;
    vk::StridedDeviceAddressRegionKHR rmiss_sbt;
    vk::StridedDeviceAddressRegionKHR rcallable_sbt;

private:
    void init_pipeline();

    options opt;
};

}

#endif
