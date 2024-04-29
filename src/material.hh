#ifndef TAURAY_MATERIAL_HH
#define TAURAY_MATERIAL_HH
#include "math.hh"

namespace tr
{

class texture;
class sampler;

using combined_tex_sampler = std::pair<const texture*, const sampler*>;
struct material
{
    vec4 albedo_factor = vec4(1);
    combined_tex_sampler albedo_tex = combined_tex_sampler(nullptr, nullptr);

    float metallic_factor = 0.0f;
    float roughness_factor = 1.0f;
    combined_tex_sampler metallic_roughness_tex =
        combined_tex_sampler(nullptr, nullptr);

    float normal_factor = 1.0f;
    combined_tex_sampler normal_tex = combined_tex_sampler(nullptr, nullptr);

    float ior = 1.45f;
    vec3 emission_factor = vec3(0);
    combined_tex_sampler emission_tex = combined_tex_sampler(nullptr, nullptr);

    float transmittance = 0.0f;

    bool double_sided = true;
    // Flag to imply that the material / object it's attached to can change
    // wildly between frames.
    bool transient = false;

    std::string name = "";

    bool potentially_transparent() const;
};

struct combined_tex_sampler_hash
{
    size_t operator()(const combined_tex_sampler & v) const
    {
        return hash_combine(
            std::hash<const texture*>()(v.first),
            std::hash<const sampler*>()(v.second)
        );
    }
};

}

#endif
