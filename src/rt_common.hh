#ifndef TAURAY_RT_COMMON_HH
#define TAURAY_RT_COMMON_HH
#include <map>
#include <string>

namespace tr
{
    enum class film_filter
    {
        POINT = 0,
        BOX,
        BLACKMAN_HARRIS
    };
    void add_defines(film_filter filter, std::map<std::string, std::string>& defines);

    enum class multiple_importance_sampling_mode
    {
        MIS_DISABLED,
        MIS_BALANCE_HEURISTIC,
        MIS_POWER_HEURISTIC
    };

    enum class bd
    {
        OFF = 0,
        BOUNCE_COUNT,
        CONTRIBUTION,
        MATERIAL_ID,
        BSDF_SUM,
        BSDF_VAR,
        PDF_CONTRIBUTION,
        FULL_PDF_CONTRIBUTION,
    };

    enum class denoiser_type
    {
        NONE = 0,
        SVGF,
        BMFR
    };

    void add_defines(multiple_importance_sampling_mode mode, std::map<std::string, std::string>& defines);

    enum class bounce_sampling_mode
    {
        HEMISPHERE, // Degrades to spherical for transmissive objects
        COSINE_HEMISPHERE, // Degrades to double-sided for transmissive objects
        MATERIAL
    };
    void add_defines(bounce_sampling_mode mode, std::map<std::string, std::string>& defines);

    enum class tri_light_sampling_mode
    {
        AREA, // Okay for small solid angles, pretty bad for large solid angles
        SOLID_ANGLE, // Good for large solid angles, bad for small due to precision issues
        HYBRID // Tries to switch between area and solid angle sampling depending on precision pitfalls
    };
    void add_defines(tri_light_sampling_mode mode, std::map<std::string, std::string>& defines);

    struct light_sampling_weights
    {
        float point_lights = 1.0f;
        float directional_lights = 1.0f;
        float envmap = 1.0f;
        float emissive_triangles = 1.0f;
    };
    void add_defines(light_sampling_weights weights, std::map<std::string, std::string>& defines);
}

#endif
