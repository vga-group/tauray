#ifndef TAURAY_ENVIRONMENT_MAP_HH
#define TAURAY_ENVIRONMENT_MAP_HH
#include "texture.hh"
#include "transformable.hh"

namespace tr
{

class environment_map: public texture, public transformable
{
public:
    enum projection
    {
        LAT_LONG = 0
    };

    environment_map(
        device_mask dev,
        const std::string& path,
        projection proj = LAT_LONG,
        vec3 factor = vec3(1.0f)
    );

    void set_factor(vec3 factor);
    vec3 get_factor() const;

    projection get_projection() const;
    vk::Buffer get_alias_table(size_t device_index) const;

private:
    void generate_alias_table();

    vec3 factor;
    projection proj;

    double average_luminance;
    struct alias_table_entry
    {
        uint32_t alias_id;
        uint32_t probability;
        float pdf;
        float alias_pdf;
    };
    std::vector<alias_table_entry> alias_table;

    per_device<vkm<vk::Buffer>> alias_table_buffers;
};

}

#endif

