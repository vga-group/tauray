#ifndef TAURAY_SAMPLER_TABLE_HH
#define TAURAY_SAMPLER_TABLE_HH
#include "context.hh"
#include "texture.hh"
#include "material.hh"
#include "sampler.hh"

namespace tr
{

class scene;

// This class exists to create a list of sampler-texture pairs out of all
// materials in a given scene, with no duplicate pairs. If only a texture is
// specified, this also fills out the default sampler.
// It exists separately from rt_pipeline and raster_pipeline only for code
// reuse.
class sampler_table
{
public:
    sampler_table(device_mask dev, bool mipmap_default);

    void update_scene(scene* s);
    std::vector<vk::DescriptorImageInfo> get_image_infos(device_id id) const;
    int find_tex_id(combined_tex_sampler cs);

private:
    void register_tex_id(combined_tex_sampler cs);

    sampler default_sampler;
    int index_counter = 0;
    std::unordered_map<
        combined_tex_sampler, int, combined_tex_sampler_hash
    > table;
};

}

#endif
