#include "options.hh"
#include "misc.hh"
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

bool parse_identifier(const char*& arg, std::string& name)
{
    // Skip whitespace
    arg += strspn(arg, " \t\r\n");
    size_t len = strcspn(arg, " \t\r\n=");
    if(len == 0)
        return false;
    name = std::string(arg, len);
    arg += len;
    arg += strspn(arg, " \t\r\n=");
    return true;
}

void parse_param(const std::string& name, const char*& arg, std::string& param)
{
    // Skip leading whitespace
    arg += strspn(arg, " \t\r\n");
    // If quote
    if(*arg == '"' || *arg == '\'')
    {
        char sep = *arg;
        arg++;
        const char* end = strchr(arg, sep);
        if(end == nullptr)
            throw option_parse_error(name + " has quoted parameter with missing unquote!");
        param = std::string(arg, end-arg);
        arg = end+1;
        arg += strspn(arg, " \t\r\n");
    }
    else
    { // Read until newline, strip back
        const char* end = strchr(arg, '\n');
        if(end == nullptr) end = arg+strlen(arg);

        param = std::string(arg, end-arg);
        arg = end;
        arg += strspn(arg, " \t\r\n");

        size_t strip_len = param.find_last_not_of(" \t\r\n");
        if(strip_len != std::string::npos)
            param.erase(strip_len+1);
    }
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

bool parse_toggle(const std::string& name, const char*& arg, char end = '\0')
{
     bool res = false;
     if(prefix(arg, "on") || prefix(arg, "true") || prefix(arg, "1"))
         res = true;
     else if(prefix(arg, "off") || prefix(arg, "false") || prefix(arg, "0"))
         res = false;

     if(*arg != end && *arg != '\0')
         throw option_parse_error(name + " expects on or off, got: " + std::string(arg));
     return res;
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
    const std::string& default_str,
    bool argument = true
){
    std::string tag;
    if(type_tag != "")
    {
        if(argument) tag = "=<"+type_tag+">";
        else tag = " <"+type_tag+">";
    }

    std::string option_name = (argument ? "--" : "") + dashify_string(name) + tag;
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

// You can thank MSVC for these macros. They can't compile an
// else-if chain longer than 127 entries, which breaks our auto-generated
// options parsing.
#define IF(condition) if((passed = (condition)))
#define ELSEIF(condition) if(!passed && (passed = (condition)))

void parse_command_line_options(char** argv, options& opt)
{
    const char* program_name = *argv++;
    bool skip_flags = false;

    while(*argv)
    {
        const char* arg = *argv++;
        if(!skip_flags && prefix(arg, "-"))
        {
            if(prefix(arg, "-"))
            {
                bool passed = false;
                IF(*arg == 0) skip_flags = true;
                ELSEIF(cmp(arg, "help"))
                {
                    print_help(program_name);
                    throw option_parse_error("");
                }
                ELSEIF(prefix(arg, "config="))
                    parse_config_options(
                        load_text_file(arg).c_str(),
                        fs::path(arg).parent_path(),
                        opt
                    );
                ELSEIF(prefix(arg, "preset="))
                {
                    parse_config_options(
                        load_text_file(get_resource_path(std::string("data/presets/")+arg+".cfg")).c_str(),
                        "data/presets",
                        opt
                    );
                }
#define TR_BOOL_OPT(name, description, default) \
                ELSEIF(cmp(arg, DASHIFY(name))) opt.name = true; \
                ELSEIF(prefix(arg, DASHIFY(name)+"=")) \
                    opt.name = parse_toggle(DASHIFY(name), arg);
#define TR_BOOL_SOPT(name, shorthand, description) \
                TR_BOOL_OPT(name, description, false)
#define TR_INT_OPT(name, description, default, min, max) \
                ELSEIF(prefix(arg, DASHIFY(name)+"=")) \
                    opt.name = parse_int(DASHIFY(name), arg, min, max);
#define TR_INT_SOPT(name, shorthand, description, default, min, max) \
                TR_INT_OPT(name, description, default, min, max)
#define TR_FLOAT_OPT(name, description, default, min, max) \
                ELSEIF(prefix(arg, DASHIFY(name)+"=")) \
                    opt.name = parse_float(DASHIFY(name), arg, min, max);
#define TR_STRING_OPT(name, description, default) \
                ELSEIF(prefix(arg, DASHIFY(name)+"=")) \
                    opt.name = arg;
#define TR_FLAG_STRING_OPT(name, description, default) \
                ELSEIF(cmp(arg, DASHIFY(name))) \
                    opt.name##_flag = true; \
                ELSEIF(prefix(arg, DASHIFY(name)+"=")) \
                {\
                    opt.name##_flag = true; \
                    opt.name = arg; \
                }
#define TR_VEC3_OPT(name, description, default, min, max) \
                ELSEIF(prefix(arg, DASHIFY(name)+"=")) \
                {\
                    opt.name.x = parse_float(DASHIFY(name)+".x", arg, min.x, max.x, ','); \
                    if(*arg == ',') arg++; \
                    opt.name.y = parse_float(DASHIFY(name)+".y", arg, min.y, max.y, ','); \
                    if(*arg == ',') arg++; \
                    opt.name.z = parse_float(DASHIFY(name)+".z", arg, min.z, max.z); \
                }
#define TR_ENUM_OPT(name, type, description, default, ...) \
                ELSEIF(prefix(arg, DASHIFY(name)+"=")) \
                    enum_str(DASHIFY(name), opt.name, arg, { __VA_ARGS__ });
#define TR_SETINT_OPT(name, description) \
                ELSEIF(prefix(arg, DASHIFY(name)+"=")) \
                {\
                    while(true) \
                    {\
                        opt.name.insert(parse_int(DASHIFY(name), arg, INT_MIN, INT_MAX, ','));\
                        if(*arg != ',') break;\
                        arg++; \
                    }\
                }
#define TR_VECFLOAT_OPT(name, description) \
                ELSEIF(prefix(arg, DASHIFY(name)+"=")) \
                {\
                    while(true) \
                    {\
                        opt.name.push_back(parse_float(DASHIFY(name), arg, -FLT_MAX, FLT_MAX, ','));\
                        if(*arg != ',') break;\
                        arg++; \
                    }\
                }
#define TR_STRUCT_OPT_INT(member, default, min, max) \
        if(check_name ? (prefix(arg, DASHIFY(member)+"=")) : (*arg != '\0')) { \
            s.member = parse_int(name_prefix+DASHIFY(member), arg, min, max, ','); \
            if(*arg == ',') arg++; \
        }
#define TR_STRUCT_OPT_FLOAT(member, default, min, max) \
        if(check_name ? (prefix(arg, DASHIFY(member)+"=")) : (*arg != '\0')) { \
            s.member = parse_float(name_prefix+DASHIFY(member), arg, min, max, ','); \
            if(*arg == ',') arg++; \
        }
#define TR_STRUCT_OPT_BOOL(member, default) \
        if(check_name ? (prefix(arg, DASHIFY(member)+"=")) : (*arg != '\0')) { \
            s.member = parse_toggle(name_prefix+DASHIFY(member), arg, ','); \
            if(*arg == ',') arg++; \
        }
#define TR_STRUCT_OPT(name, description, ...)\
        ELSEIF(prefix(arg, DASHIFY(name)+".") || prefix(arg, DASHIFY(name)+"=")) \
        {\
            std::string name_prefix = DASHIFY(name)+"."; \
            auto& s = opt.name; \
            bool check_name = arg[-1] == '.'; \
            __VA_ARGS__ \
            if(check_name && *arg != '\0') \
                throw option_parse_error( \
                    "Unknown struct command member: " + std::string(arg) \
                ); \
            else if(*arg != '\0') \
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
#undef TR_STRUCT_OPT_BOOL
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
                if(!passed) throw option_parse_error(
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
        return;

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

bool parse_config_options(const char* config_str, fs::path relative_path, options& opt)
{
    bool got_any = false;
    while(*config_str != 0)
    {
        std::string identifier;
        if(!parse_identifier(config_str, identifier))
            continue;

        if(identifier[0] == '#')
        {
            const char* end = strchr(config_str, '\n');
            if(end == nullptr) end = config_str+strlen(config_str);
            config_str = end;
            continue;
        }

        std::string param;

        parse_param(identifier, config_str, param);

        const char* id = identifier.c_str();
        const char* arg = param.c_str();

        if(identifier == "help")
        {
            print_command_help(param);
            continue;
        }
        else if(identifier == "quit")
        {
            opt.running = false;
            continue;
        }
        else if(identifier == "config")
        {
            fs::path param_path(param);
            if(param_path.is_relative())
                param_path = relative_path / param_path;
            parse_config_options(
                load_text_file(param_path.string()).c_str(),
                param_path.parent_path(),
                opt
            );
        }
        else if(identifier == "preset")
        {
            parse_config_options(
                load_text_file(get_resource_path("data/presets/"+param+".cfg")).c_str(),
                "data/presets/",
                opt
            );
        }
        else if(identifier == "dump")
        {
            print_options(opt, param == "full");
            continue;
        }
#define TR_BOOL_OPT(name, description, default) \
        else if(identifier == DASHIFY(name)) \
            opt.name = parse_toggle(DASHIFY(name), arg);
#define TR_BOOL_SOPT(name, shorthand, description) \
        TR_BOOL_OPT(name, description, false)
#define TR_INT_OPT(name, description, default, min, max) \
        else if(identifier == DASHIFY(name)) \
            opt.name = parse_int(DASHIFY(name), arg, min, max);
#define TR_INT_SOPT(name, shorthand, description, default, min, max) \
        TR_INT_OPT(name, description, default, min, max)
#define TR_FLOAT_OPT(name, description, default, min, max) \
        else if(identifier == DASHIFY(name)) \
            opt.name = parse_float(DASHIFY(name), arg, min, max);
#define TR_STRING_OPT(name, description, default) \
        else if(identifier == DASHIFY(name)) \
            opt.name = arg;
#define TR_FLAG_STRING_OPT(name, description, default) \
        else if(identifier == DASHIFY(name)) \
        {\
            opt.name##_flag = true; \
            opt.name = arg; \
        }
#define TR_VEC3_OPT(name, description, default, min, max) \
        else if(identifier == DASHIFY(name)) \
        {\
            opt.name.x = parse_float(DASHIFY(name)+".x", arg, min.x, max.x, ','); \
            if(*arg == ',') arg++; \
            opt.name.y = parse_float(DASHIFY(name)+".y", arg, min.y, max.y, ','); \
            if(*arg == ',') arg++; \
            opt.name.z = parse_float(DASHIFY(name)+".z", arg, min.z, max.z); \
        }
#define TR_ENUM_OPT(name, type, description, default, ...) \
        else if(identifier == DASHIFY(name)) \
            enum_str(DASHIFY(name), opt.name, arg, { __VA_ARGS__ });
#define TR_SETINT_OPT(name, description) \
        else if(identifier == DASHIFY(name)) \
        {\
            while(true) \
            {\
                opt.name.insert(parse_int(DASHIFY(name), arg, INT_MIN, INT_MAX, ','));\
                if(*arg != ',') break;\
                arg++; \
            }\
        }
#define TR_VECFLOAT_OPT(name, description) \
        else if(identifier == DASHIFY(name)) \
        {\
            while(true) \
            {\
                opt.name.push_back(parse_float(DASHIFY(name), arg, -FLT_MAX, FLT_MAX, ','));\
                if(*arg != ',') break;\
                arg++; \
            }\
        }
#define TR_STRUCT_OPT_INT(member, default, min, max) \
        if(check_name ? (id == DASHIFY(member)) : (*arg != '\0')) { \
            s.member = parse_int(name_prefix+DASHIFY(member), arg, min, max, ','); \
            if(*arg == ',') arg++; \
        }
#define TR_STRUCT_OPT_FLOAT(member, default, min, max) \
        if(check_name ? (id == DASHIFY(member)) : (*arg != '\0')) { \
            s.member = parse_float(name_prefix+DASHIFY(member), arg, min, max, ','); \
            if(*arg == ',') arg++; \
        }
#define TR_STRUCT_OPT_BOOL(member, default) \
        if(check_name ? (id == DASHIFY(member)) : (*arg != '\0')) { \
            s.member = parse_toggle(name_prefix+DASHIFY(member), arg, ','); \
            if(*arg == ',') arg++; \
        }
#define TR_STRUCT_OPT(name, description, ...)\
        else if(prefix(id, DASHIFY(name))) \
        {\
            std::string name_prefix = DASHIFY(name)+"."; \
            auto& s = opt.name; \
            bool check_name = false; \
            if(*id == '.') \
            { \
                check_name = true; \
                id++; \
            } \
            __VA_ARGS__ \
            if(check_name && *arg != '\0') \
                throw option_parse_error( \
                    "Unknown struct command member: " + std::string(id) \
                ); \
            else if(*arg != '\0') \
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
#undef TR_STRUCT_OPT_BOOL
#undef TR_STRUCT_OPT
        else throw option_parse_error("Unknown option " + identifier);
        got_any = true;
    }
    return got_any;
}

bool parse_command(const char* config_str, options& opt)
{
    try
    {
        return parse_config_options(config_str, "", opt);
    }
    catch(const option_parse_error& err)
    {
        std::cerr << err.what() << std::endl;
        return false;
    }
}

void print_command_help(const std::string& command)
{
#define opt(name, ...) \
    if(command == dashify_string(name)) \
    { \
        std::cout << build_option_string(name, __VA_ARGS__, false); \
        return; \
    }
#define TR_BOOL_OPT(name, description, default) \
    opt(#name, "on|off", '\0', description, default ? "on" : "off");
#define TR_BOOL_SOPT(name, shorthand, description) \
    opt(#name, "on|off", shorthand, description, "");
#define TR_INT_OPT(name, description, default, min, max) \
    opt(#name, "integer", '\0', description, std::to_string(default));
#define TR_INT_SOPT(name, shorthand, description, default, min, max) \
    opt(#name, "integer", shorthand, description, std::to_string(default));
#define TR_FLOAT_OPT(name, description, default, min, max)\
    opt(#name, "number", '\0', description, std::to_string(default));
#define TR_STRING_OPT(name, description, default) \
    opt(#name, "string", '\0', description, default);
#define TR_FLAG_STRING_OPT(name, description, default) \
    opt(#name, "string", '\0', description, default);
#define TR_VEC3_OPT(name, description, default, min, max) \
    opt(#name, "x,y,z", '\0', description, vec_to_string(default));
#define TR_ENUM_OPT(name, type, description, default, ...) \
    opt(#name, \
        gather_enum_str(std::vector<std::pair<std::string, type>>{__VA_ARGS__}), \
        '\0', description, find_default_enum_string<type>(default, {__VA_ARGS__}));
#define TR_SETINT_OPT(name, description) \
    opt(#name, "int,int,...", '\0', description, "");
#define TR_VECFLOAT_OPT(name, description) \
    opt(#name, "float,float,...", '\0', description, "");
#define TR_STRUCT_OPT_INT(name, default, min, max) \
    type_tag += DASHIFY(name) + ","; \
    default_str += DASHIFY(name) + " = " + std::to_string(default) + ", ";
#define TR_STRUCT_OPT_FLOAT(name, default, min, max) \
    type_tag += DASHIFY(name) + ","; \
    default_str += DASHIFY(name) + " = " + std::to_string(default) + ", ";
#define TR_STRUCT_OPT_BOOL(name, default) \
    type_tag += DASHIFY(name) + ","; \
    default_str += DASHIFY(name) + " = " + (default ? "true" : "false") + ", ";
#define TR_STRUCT_OPT(name, description, ...) \
    {\
        std::string type_tag = ""; \
        std::string default_str = ""; \
        __VA_ARGS__ \
        type_tag.resize(type_tag.size()-1);\
        default_str.resize(default_str.size()-2);\
        opt(#name, type_tag, '\0', description, default_str); \
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
#undef TR_STRUCT_OPT_BOOL
#undef TR_STRUCT_OPT
#undef opt
    std::cout << "Unknown command: " << command << std::endl;
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
  --config=<string>
    Load the given config file.
  --preset=<reference|quality|accumulation|denoised|ddish-gi>
    Load the given preset file (config file that is shipped with Tauray).
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
#define TR_STRUCT_OPT_BOOL(name, default) \
    type_tag += DASHIFY(name) + ","; \
    default_str += DASHIFY(name) + " = " + (default ? "true" : "false") + ", ";
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
#undef TR_STRUCT_OPT_INT
#undef TR_STRUCT_OPT_FLOAT
#undef TR_STRUCT_OPT_BOOL
#undef TR_STRUCT_OPT
#undef lopt
#undef sopt
    for(const auto& pair: short_option_strings)
        std::cout << pair.second;
    for(const auto& pair: long_option_strings)
        std::cout << pair.second;
}

void print_options(options& opt, bool full)
{
#define dump(name, value) \
    std::cout << (name) << " " << (value) << std::endl;
#define TR_BOOL_OPT(name, description, default) \
    if(full || opt.name != default) dump(DASHIFY(name), (opt.name ? "on" : "off"))
#define TR_BOOL_SOPT(name, shorthand, description) \
    if(full || opt.name != false) dump(DASHIFY(name), (opt.name ? "on" : "off"))
#define TR_INT_OPT(name, description, default, min, max) \
    if(full || opt.name != default) dump(DASHIFY(name), std::to_string(opt.name))
#define TR_INT_SOPT(name, shorthand, description, default, min, max) \
    if(full || opt.name != default) dump(DASHIFY(name), std::to_string(opt.name))
#define TR_FLOAT_OPT(name, description, default, min, max) \
    if(full || (opt.name != default && (!isnan(double(default)) || !isnan(opt.name)))) dump(DASHIFY(name), std::to_string(opt.name))
#define TR_STRING_OPT(name, description, default) \
    if(full || opt.name != default) dump(DASHIFY(name), "\""+opt.name+"\"")
#define TR_FLAG_STRING_OPT(name, ...) \
    if(opt.name##_flag) dump(DASHIFY(name), "\""+opt.name+"\"")
#define TR_VEC3_OPT(name, description, default, min, max) \
    if(full || opt.name != default) \
        std::cout << DASHIFY(name) << " " << opt.name.x << "," << opt.name.y << "," << opt.name.z << std::endl;
#define TR_ENUM_OPT(name, type, description, default, ...) \
    if(full || !(opt.name == default)) \
    { \
        std::vector<std::pair<std::string, type>> options{__VA_ARGS__}; \
        std::cout << DASHIFY(name) << " "; \
        for(const auto& pair: options) \
        { \
            if(pair.second == opt.name) \
            { \
                std::cout << pair.first; \
                break; \
            } \
        } \
        std::cout << std::endl; \
    }
#define TR_SETINT_OPT(name, ...) \
    if(full || opt.name.size() != 0) \
    { \
        std::cout << DASHIFY(name) << " "; \
        bool first = true; \
        for(int n: opt.name) \
        { \
            if(!first) std::cout << ","; \
            std::cout << n; \
            first = false;\
        } \
        std::cout << std::endl; \
    }
#define TR_VECFLOAT_OPT(name, ...) \
    if(full || opt.name.size() != 0) \
    { \
        std::cout << DASHIFY(name) << " "; \
        bool first = true; \
        for(int n: opt.name) \
        { \
            if(!first) std::cout << ","; \
            std::cout << n; \
            first = false;\
        } \
        std::cout << std::endl; \
    }
#define TR_STRUCT_OPT_INT(name, default, ...) \
    if(full || value.name != default) \
        std::cout << name_start+DASHIFY(name) << " " << value.name << std::endl;
#define TR_STRUCT_OPT_FLOAT(name, default, ...) \
    if(full || value.name != default) \
        std::cout << name_start+DASHIFY(name) << " " << value.name << std::endl;
#define TR_STRUCT_OPT_BOOL(name, default) \
    if(full || value.name != default) \
        std::cout << name_start+DASHIFY(name) << " " << value.name << std::endl;
#define TR_STRUCT_OPT(name, description, ...) \
    {\
        auto& value = opt.name; \
        std::string name_start = DASHIFY(name) + "."; \
        __VA_ARGS__ \
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
#undef TR_STRUCT_OPT_BOOL
#undef TR_STRUCT_OPT
#undef dump
}

}
