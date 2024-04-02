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
    compute_pipeline(device& dev);
    void init(
        shader_source src,
        std::vector<tr::descriptor_set_layout*> layout
    );
};

}

#endif
