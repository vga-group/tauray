#ifndef TAURAY_SKINNING_STAGE_HH
#define TAURAY_SKINNING_STAGE_HH
#include "context.hh"
#include "compute_pipeline.hh"
#include "timer.hh"
#include "mesh.hh"
#include "stage.hh"

namespace tr
{

class scene;
// Applies animations to all skinned meshes. This should run _before_ scene
// update!
class skinning_stage: public stage
{
public:
    skinning_stage(device_data& dev, uint32_t max_instances);

    void set_scene(scene* s);
    scene* get_scene();

private:
    void update(uint32_t frame_index) override;

    compute_pipeline comp;
    scene* cur_scene;
    timer stage_timer;
    uint32_t max_instances;
};

}

#endif
