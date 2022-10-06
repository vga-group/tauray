#include "options.hh"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <algorithm>

namespace
{
using namespace tr;

std::string dashify_string(std::string str)
{
    std::replace(str.begin(), str.end(), '_', '-');
    return str;
}
#define DASHIFY(name) dashify_string(#name)

std::string vec_to_string(vec3 v)
{
    return "("+std::to_string(v.x)+", "+std::to_string(v.y)+", "+
        std::to_string(v.z)+")";
}

bool prefix(const char*& arg, const std::string& prefix)
{
    size_t len = strlen(prefix.c_str());
    if(strncmp(arg, prefix.c_str(), len) == 0)
    {
        arg += len;
        return true;
    }
    return false;
}

bool cmp(const char*& arg, const std::string& prefix)
{
    return arg == prefix;
}

int parse_int(const std::string& name, const char*& arg, int min, int max, char end = '\0')
{
    char* endptr = nullptr;
    int result = strtol(arg, &endptr, 10);
    if(*endptr != end && *endptr != '\0')
        throw option_parse_error(name + " expects integer, got: " + std::string(arg));
    if(result < min || result > max)
        throw option_parse_error(
            name + " expects integer in range [" + std::to_string(min) + ", " +
            std::to_string(max) + "], got: " + std::to_string(result)
        );
    arg = endptr;
    return result;
}

double parse_float(const std::string& name, const char*& arg, float min, float max, char end = '\0')
{
    char* endptr = nullptr;
    double result = strtod(arg, &endptr);
    if(*endptr != end && *endptr != '\0')
        throw option_parse_error(name + " expects number, got: " + std::string(arg));
    if((result < min || result > max) && !isnan(min))
        throw option_parse_error(
            name + " expects number in range [" + std::to_string(min) + ", " +
            std::to_string(max) + "], got: " + std::to_string(result)
        );
    arg = endptr;
    return result;
}

bool parse_toggle(const std::string& name, const char*& arg)
{
    if(!strcmp(arg, "on") || !strcmp(arg, "true") || !strcmp(arg, "1"))
        return true;
    if(!strcmp(arg, "off") || !strcmp(arg, "false") || !strcmp(arg, "0"))
        return false;
    throw option_parse_error(name + " expects on or off, got: " + std::string(arg));
}

template<typename T>
void enum_str(
    const std::string& name,
    T& to,
    const char*& arg,
    const std::vector<std::pair<std::string, T>>& allowed
){
    for(const auto& pair: allowed)
    {
        if(arg == pair.first)
        {
            arg += pair.first.size();
            to = pair.second;
            return;
        }
    }
    std::string error_msg = name + " expects one of {";
    for(size_t i = 0; i < allowed.size(); ++i)
    {
        error_msg += allowed[i].first;
        if(i != allowed.size()-1)
            error_msg += ", ";
    }
    throw option_parse_error(error_msg + "}, got " + arg);
}

template<typename T>
std::string gather_enum_str(
    const std::vector<std::pair<std::string, T>>& allowed
){
    std::string type_tag = "";
    for(const auto& pair: allowed)
        type_tag += pair.first + "|";
    type_tag.resize(type_tag.size()-1);
    return type_tag;
}

template<typename T>
std::string find_default_enum_string(
    const T& def,
    const std::vector<std::pair<std::string, T>>& allowed
){
    std::string tag = "";
    for(const auto& pair: allowed)
        if(pair.second == def)
            return pair.first;
    return "";
}

std::string build_option_string(
    const std::string& name,
    const std::string& type_tag,
    char shorthand,
    const std::string& description,
    const std::string& default_str
){
    std::string tag;
    if(type_tag != "") tag = "=<"+type_tag+">";

    std::string option_name = "--" + dashify_string(name) + tag;
    if(shorthand)
    {
        std::string short_option = "-" + std::string(1, shorthand);
        if(default_str != "") short_option += tag;
        option_name = short_option + ", " + option_name;
    }
    std::string full_description = description;
    if(default_str != "")
        full_description += " The default is " + default_str + ".";
    return "  " + option_name + "\n    " + full_description + "\n";
}

}

namespace tr
{

options parse_options(char** argv)
{
    options opt;
    argv++; // Skip name
    bool skip_flags = false;
    opt.film_radius = -1.0f;

    while(*argv)
    {
        const char* arg = *argv++;
        if(!skip_flags && prefix(arg, "-"))
        {
            if(prefix(arg, "-"))
            {
                if(*arg == 0) skip_flags = true;
                else if(cmp(arg, "help")) throw option_parse_error("");
#define TR_BOOL_OPT(name, description, default) \
                else if(cmp(arg, DASHIFY(name))) opt.name = true; \
                else if(prefix(arg, DASHIFY(name)+"=")) \
                    opt.name = parse_toggle(DASHIFY(name), arg);
#define TR_BOOL_SOPT(name, shorthand, description) \
                TR_BOOL_OPT(name, description, false)
#define TR_INT_OPT(name, description, default, min, max) \
                else if(prefix(arg, DASHIFY(name)+"=")) \
                    opt.name = parse_int(DASHIFY(name), arg, min, max);
#define TR_INT_SOPT(name, shorthand, description, default, min, max) \
                TR_INT_OPT(name, description, default, min, max)
#define TR_FLOAT_OPT(name, description, default, min, max) \
                else if(prefix(arg, DASHIFY(name)+"=")) \
                    opt.name = parse_float(DASHIFY(name), arg, min, max);
#define TR_STRING_OPT(name, description, default) \
                else if(prefix(arg, DASHIFY(name)+"=")) \
                    opt.name = arg;
#define TR_FLAG_STRING_OPT(name, description, default) \
                else if(cmp(arg, DASHIFY(name))) \
                    opt.name##_flag = true; \
                else if(prefix(arg, DASHIFY(name)+"=")) \
                {\
                    opt.name##_flag = true; \
                    opt.name = arg; \
                }
#define TR_VEC3_OPT(name, description, default, min, max) \
                else if(prefix(arg, DASHIFY(name)+"=")) \
                {\
                    opt.name.x = parse_float(DASHIFY(name)+".x", arg, min.x, max.x, ','); \
                    if(*arg == ',') arg++; \
                    opt.name.y = parse_float(DASHIFY(name)+".y", arg, min.y, max.y, ','); \
                    if(*arg == ',') arg++; \
                    opt.name.z = parse_float(DASHIFY(name)+".z", arg, min.z, max.z); \
                }
#define TR_ENUM_OPT(name, type, description, default, ...) \
                else if(prefix(arg, DASHIFY(name)+"=")) \
                    enum_str(DASHIFY(name), opt.name, arg, { __VA_ARGS__ });
#define TR_SETINT_OPT(name, description) \
                else if(prefix(arg, DASHIFY(name)+"=")) \
                {\
                    while(true) \
                    {\
                        opt.name.insert(parse_int(DASHIFY(name), arg, INT_MIN, INT_MAX, ','));\
                        if(*arg != ',') break;\
                        arg++; \
                    }\
                }
#define TR_VECFLOAT_OPT(name, description) \
                else if(prefix(arg, DASHIFY(name)+"=")) \
                {\
                    while(true) \
                    {\
                        opt.name.push_back(parse_float(DASHIFY(name), arg, -FLT_MAX, FLT_MAX, ','));\
                        if(*arg != ',') break;\
                        arg++; \
                    }\
                }
#define TR_STRUCT_OPT_INT(name, default, min, max) \
                if(*arg != '\0') { \
                    s.name = parse_int(name_prefix+DASHIFY(name), arg, min, max, ','); \
                    if(*arg == ',') arg++; \
                }
#define TR_STRUCT_OPT_FLOAT(name, default, min, max) \
                if(*arg != '\0') { \
                    s.name = parse_float(name_prefix+DASHIFY(name), arg, min, max, ','); \
                    if(*arg == ',') arg++; \
                }
#define TR_STRUCT_OPT(name, description, ...)\
                else if(prefix(arg, DASHIFY(name)+"=")) \
                {\
                    std::string name_prefix = DASHIFY(name)+"."; \
                    auto& s = opt.name; \
                    __VA_ARGS__ \
                    if(*arg != '\0') \
                        throw option_parse_error( \
                            "Unexpected extra value in struct: " +  \
                            std::string(arg) \
                        ); \
                }
                TR_OPTIONS
#undef TR_BOOL_OPT
#undef TR_BOOL_SOPT
#undef TR_INT_OPT
#undef TR_INT_SOPT
#undef TR_FLOAT_OPT
#undef TR_STRING_OPT
#undef TR_FLAG_STRING_OPT
#undef TR_VEC3_OPT
#undef TR_ENUM_OPT
#undef TR_SETINT_OPT
#undef TR_VECFLOAT_OPT
#undef TR_STRUCT_OPT_INT
#undef TR_STRUCT_OPT_FLOAT
#undef TR_STRUCT_OPT
#define TR_BOOL_OPT(...)
#define TR_BOOL_SOPT(...)
#define TR_INT_OPT(...)
#define TR_INT_SOPT(...)
#define TR_FLOAT_OPT(...)
#define TR_STRING_OPT(...)
#define TR_FLAG_STRING_OPT(...)
#define TR_VEC3_OPT(...)
#define TR_ENUM_OPT(...)
#define TR_SETINT_OPT(...)
#define TR_VECFLOAT_OPT(...)
#define TR_STRUCT_OPT(...)
                else throw option_parse_error(
                    "Unknown long flag " + std::string(arg));
            }
#undef TR_INT_SOPT
#define TR_INT_SOPT(name, shorthand, description, default, min, max) \
            else if(prefix(arg, std::string(1, shorthand)+"=")) \
                opt.name = parse_int(DASHIFY(name), arg, min, max);
            TR_OPTIONS
#undef TR_INT_SOPT
#define TR_INT_SOPT(...)
            else for(;*arg != 0; ++arg) switch(*arg)
            {
#undef TR_BOOL_SOPT
#define TR_BOOL_SOPT(name, shorthand, description) \
            case shorthand: \
                opt. name = true; \
                break;
                TR_OPTIONS
#undef TR_BOOL_SOPT
#define TR_BOOL_SOPT(...)
            default:
                throw option_parse_error(std::string("Unknown flag ") + *arg);
            }
        }
        else
        {
            opt.scene_paths.push_back(arg);
        }
    }

    // The frame client has no required options and is mostly a separate program
    // anyway, so let's just skip the pointless option validation.
    if(opt.display == options::display_type::FRAME_CLIENT)
        return opt;

    if(opt.scene_paths.size() == 0)
        throw option_parse_error("No scene specified!");

    // Sanitize options
    // Headless implies replay, since there can be no interactivity
    if(opt.headless.size()) opt.replay = true;
    else opt.skip_render = false;

    // Headless is not compatible with XR, since we can't open OpenXR.
    if(opt.headless.size())
        opt.display = options::display_type::HEADLESS;

    // XR is not compatible with a lot of camera options, as it overrides those.
    if(opt.display == options::display_type::OPENXR)
    {
        opt.camera_grid.w = 1;
        opt.camera_grid.h = 1;
        opt.gamma = 1.0; // Gamma correction is done by the XR runtime as needed.
        opt.hdr = true;
        opt.force_projection.reset();
    }
    else if(opt.display == options::display_type::LOOKING_GLASS)
    {
        opt.camera_grid.h = 1;
        opt.camera_grid.w = 1;
        opt.force_projection.reset();
    }

    if(std::get_if<feature_stage::feature>(&opt.renderer))
    {
        // Tonemapping is unwanted when rendering feature buffers
        opt.tonemap = tonemap_stage::LINEAR;
    }

    if(opt.film_radius < 0)
    {
        // Different radius defaults for different film types!
        if(opt.film == film::BOX) opt.film_radius = 0.5f;
        else if(opt.film == film::BLACKMAN_HARRIS)
            opt.film_radius = 1.0f;
    }

    return opt;
#undef TR_BOOL_OPT
#undef TR_BOOL_SOPT
#undef TR_INT_OPT
#undef TR_INT_SOPT
#undef TR_FLOAT_OPT
#undef TR_STRING_OPT
#undef TR_FLAG_STRING_OPT
#undef TR_VEC3_OPT
#undef TR_ENUM_OPT
#undef TR_SETINT_OPT
#undef TR_VECFLOAT_OPT
#undef TR_STRUCT_OPT
}

void print_help(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [options] scene" << R"(
'scene' must be a glTF 2.0 file, with a .glb extension.
The initial position of the camera will be set to the first camera
object described in the file.

Options:
  --help
    Show this information.
)";
    std::map<std::string, std::string> short_option_strings;
    std::map<std::string, std::string> long_option_strings;
#define lopt(name, ...) \
    long_option_strings[name] = build_option_string(name, __VA_ARGS__);
#define sopt(name, ...) \
    short_option_strings[name] = build_option_string(name, __VA_ARGS__);
#define TR_BOOL_OPT(name, description, default) \
    lopt(#name, "on|off", '\0', description, default ? "on" : "off");
#define TR_BOOL_SOPT(name, shorthand, description) \
    sopt(#name, "on|off", shorthand, description, "");
#define TR_INT_OPT(name, description, default, min, max) \
    lopt(#name, "integer", '\0', description, std::to_string(default));
#define TR_INT_SOPT(name, shorthand, description, default, min, max) \
    sopt(#name, "integer", shorthand, description, std::to_string(default));
#define TR_FLOAT_OPT(name, description, default, min, max)\
    lopt(#name, "number", '\0', description, std::to_string(default));
#define TR_STRING_OPT(name, description, default) \
    lopt(#name, "string", '\0', description, default);
#define TR_FLAG_STRING_OPT(name, description, default) \
    lopt(#name, "string", '\0', description, default);
#define TR_VEC3_OPT(name, description, default, min, max) \
    lopt(#name, "x,y,z", '\0', description, vec_to_string(default));
#define TR_ENUM_OPT(name, type, description, default, ...) \
    lopt(#name, \
        gather_enum_str(std::vector<std::pair<std::string, type>>{__VA_ARGS__}), \
        '\0', description, find_default_enum_string<type>(default, {__VA_ARGS__}));
#define TR_SETINT_OPT(name, description) \
    lopt(#name, "int,int,...", '\0', description, "");
#define TR_VECFLOAT_OPT(name, description) \
    lopt(#name, "float,float,...", '\0', description, "");
#define TR_STRUCT_OPT_INT(name, default, min, max) \
    type_tag += DASHIFY(name) + ","; \
    default_str += DASHIFY(name) + " = " + std::to_string(default) + ", ";
#define TR_STRUCT_OPT_FLOAT(name, default, min, max) \
    type_tag += DASHIFY(name) + ","; \
    default_str += DASHIFY(name) + " = " + std::to_string(default) + ", ";
#define TR_STRUCT_OPT(name, description, ...) \
    {\
        std::string type_tag = ""; \
        std::string default_str = ""; \
        __VA_ARGS__ \
        type_tag.resize(type_tag.size()-1);\
        default_str.resize(default_str.size()-2);\
        lopt(#name, type_tag, '\0', description, default_str); \
    }
    TR_OPTIONS
#undef TR_BOOL_OPT
#undef TR_BOOL_SOPT
#undef TR_INT_OPT
#undef TR_INT_SOPT
#undef TR_FLOAT_OPT
#undef TR_STRING_OPT
#undef TR_FLAG_STRING_OPT
#undef TR_VEC3_OPT
#undef TR_ENUM_OPT
#undef TR_SETINT_OPT
#undef TR_VECFLOAT_OPT
#undef TR_STRUCT_OPT
#undef lopt
#undef sopt
    for(const auto& pair: short_option_strings)
        std::cout << pair.second;
    for(const auto& pair: long_option_strings)
        std::cout << pair.second;
}

}
