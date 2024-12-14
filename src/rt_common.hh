#ifndef TAURAY_RT_COMMON_HH
#define TAURAY_RT_COMMON_HH
#include <map>
#include <string>
#include <vector>

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
        NORMAL,
        POSITION,
        POSITION_2
    };

    enum class denoiser_type
    {
        NONE = 0,
        SVGF,
        BMFR
    };

    const std::map<std::string, bd> str_bd_map =
    {
        {"bounce-count", bd::BOUNCE_COUNT},
        {"bounce-contribution", bd::CONTRIBUTION},
        {"material-id", bd::MATERIAL_ID},
        {"bsdf-sum", bd::BSDF_SUM},
        {"bsdf-variance", bd::BSDF_VAR},
        {"bsdf-contribution", bd::PDF_CONTRIBUTION},
        {"bsdf-nee-contribution", bd::FULL_PDF_CONTRIBUTION},
        {"normal", bd::NORMAL},
        {"position", bd::POSITION},
        {"position-2", bd::POSITION_2}
    };

    const std::map<bd, std::string> bd_str_map =
    {
        { bd::BOUNCE_COUNT,"BD_BOUNCE_COUNT"},
        { bd::CONTRIBUTION, "BD_CONTRIBUTION"},
        { bd::MATERIAL_ID, "BD_MATERIAL_ID"},
        { bd::BSDF_SUM, "BD_BSDF_SUM"},
        { bd::BSDF_VAR, "BD_BSDF_VAR"},
        { bd::PDF_CONTRIBUTION, "BD_PDF_CONTRIBUTION"},
        { bd::FULL_PDF_CONTRIBUTION, "BD_FULL_PDF_CONTRIBUTION"}
    };

    std::vector<std::string> parse_bounce_data(const std::string str);


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
