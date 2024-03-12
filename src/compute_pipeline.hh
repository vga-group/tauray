#ifndef TAURAY_COMPUTE_PIPELINE_HH
#define TAURAY_COMPUTE_PIPELINE_HH
#include "basic_pipeline.hh"

namespace tr
{

struct shader_sources;

// Compute pipelines are per-device. Because of the nature of this program as a
// renderer, even the compute pipeline is somewhat related to graphics
// (in-flight mechanics)
class compute_pipeline: public basic_pipeline
{
public:
    struct params
    {
        shader_source src = {};
        binding_array_length_info binding_array_lengths;
        uint32_t max_descriptor_sets = MAX_FRAMES_IN_FLIGHT;
        bool use_push_descriptors = false;
        std::vector<tr::descriptor_set_layout*> layout = {};
    };

    compute_pipeline(device& dev, const params& p);
};

}

#endif
