#include "rt_common.hh"

namespace tr
{

void add_defines(film_filter filter, std::map<std::string, std::string>& defines)
{
    switch(filter)
    {
    case film_filter::POINT:
        defines["USE_POINT_FILTER"];
        break;
    case film_filter::BOX:
        defines["USE_BOX_FILTER"];
        break;
    case film_filter::BLACKMAN_HARRIS:
        defines["USE_BLACKMAN_HARRIS_FILTER"];
        break;
    }
}

void add_defines(multiple_importance_sampling_mode mode, std::map<std::string, std::string>& defines)
{
    switch(mode)
    {
    case multiple_importance_sampling_mode::MIS_DISABLED:
        break;
    case multiple_importance_sampling_mode::MIS_BALANCE_HEURISTIC:
        defines["MIS_BALANCE_HEURISTIC"];
        break;
    case multiple_importance_sampling_mode::MIS_POWER_HEURISTIC:
        defines["MIS_POWER_HEURISTIC"];
        break;
    }
}

void add_defines(bounce_sampling_mode mode, std::map<std::string, std::string>& defines)
{
    switch(mode)
    {
    case bounce_sampling_mode::HEMISPHERE:
        defines["BOUNCE_HEMISPHERE"];
        break;
    case bounce_sampling_mode::COSINE_HEMISPHERE:
        defines["BOUNCE_COSINE_HEMISPHERE"];
        break;
    case bounce_sampling_mode::MATERIAL:
        defines["BOUNCE_MATERIAL"];
        break;
    }
}

void add_defines(tri_light_sampling_mode mode, std::map<std::string, std::string>& defines)
{
    switch(mode)
    {
    case tri_light_sampling_mode::AREA:
        defines["TRI_LIGHT_SAMPLE_AREA"];
        break;
    case tri_light_sampling_mode::SOLID_ANGLE:
        defines["TRI_LIGHT_SAMPLE_SOLID_ANGLE"];
        break;
    case tri_light_sampling_mode::HYBRID:
        defines["TRI_LIGHT_SAMPLE_HYBRID"];
        break;
    }
}

void add_defines(light_sampling_weights weights, std::map<std::string, std::string>& defines)
{
    if(weights.point_lights > 0)
        defines["NEE_SAMPLE_POINT_LIGHTS"] = std::to_string(weights.point_lights);
    if(weights.directional_lights > 0)
        defines["NEE_SAMPLE_DIRECTIONAL_LIGHTS"] = std::to_string(weights.directional_lights);
    if(weights.envmap > 0)
        defines["NEE_SAMPLE_ENVMAP"] = std::to_string(weights.envmap);
    if(weights.emissive_triangles > 0)
        defines["NEE_SAMPLE_EMISSIVE_TRIANGLES"] = std::to_string(weights.emissive_triangles);
}

}
