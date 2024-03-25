#include "shader_source.hh"
#include <glslang/Public/ShaderLang.h>
#include <SPIRV/GlslangToSpv.h>
#include <StandAlone/DirStackFileIncluder.h>

// Yeah, I know this is horribly ugly. It's just how we get
// DefaultTBuiltInResource.
#include <StandAlone/ResourceLimits.cpp>
#include "spirv_reflect.h"

#include <filesystem>
#include "misc.hh"
namespace fs = std::filesystem;

namespace
{
using namespace tr;

EShLanguage detect_shader_language(const std::string& ext)
{
    if(ext == ".vert") return EShLangVertex;
    else if(ext == ".tesc") return EShLangTessControl;
    else if(ext == ".tese") return EShLangTessEvaluation;
    else if(ext == ".geom") return EShLangGeometry;
    else if(ext == ".frag") return EShLangFragment;
    else if(ext == ".comp") return EShLangCompute;
    else if(ext == ".rgen") return EShLangRayGen;
    else if(ext == ".rint") return EShLangIntersect;
    else if(ext == ".rahit") return EShLangAnyHit;
    else if(ext == ".rchit") return EShLangClosestHit;
    else if(ext == ".rmiss") return EShLangMiss;
    else throw std::runtime_error("Unknown shader extension " + ext);
}

vk::ShaderStageFlagBits detect_shader_stage(const std::string& ext)
{
    if(ext == ".vert") return vk::ShaderStageFlagBits::eVertex;
    else if(ext == ".tesc") return vk::ShaderStageFlagBits::eTessellationControl;
    else if(ext == ".tese") return vk::ShaderStageFlagBits::eTessellationEvaluation;
    else if(ext == ".geom") return vk::ShaderStageFlagBits::eGeometry;
    else if(ext == ".frag") return vk::ShaderStageFlagBits::eFragment;
    else if(ext == ".comp") return vk::ShaderStageFlagBits::eCompute;
    else if(ext == ".rgen") return vk::ShaderStageFlagBits::eRaygenKHR;
    else if(ext == ".rint") return vk::ShaderStageFlagBits::eIntersectionKHR;
    else if(ext == ".rahit") return vk::ShaderStageFlagBits::eAnyHitKHR;
    else if(ext == ".rchit") return vk::ShaderStageFlagBits::eClosestHitKHR;
    else if(ext == ".rmiss") return vk::ShaderStageFlagBits::eMissKHR;
    else throw std::runtime_error("Unknown shader extension " + ext);
}

std::string remove_newlines(const std::string& orig)
{
    std::string res = orig;
    res.erase(std::remove(res.begin(), res.end(), '\n'), res.end());
    return res;
}

std::string generate_definition_src(
    const std::map<std::string, std::string>& defines
){
    std::stringstream ss;
#if 0
    ss << "#extension GL_EXT_debug_printf : enable\n";
#endif
    for(auto& pair: defines)
        ss  << "#define " << pair.first << " "
            << remove_newlines(pair.second) << std::endl;
    return ss.str();
}

void append_shader_pc_ranges(
    std::vector<vk::PushConstantRange>& ranges,
    const shader_source& src
){
    // TODO: This probably isn't correct, but as of writing, we only use one
    // push constant range per program anyway so this works in that case.
    size_t i = 0;
    for(; i < src.push_constant_ranges.size() && i < ranges.size(); ++i)
        ranges[i].stageFlags |= src.push_constant_ranges[i].stageFlags;

    for(; i < src.push_constant_ranges.size(); ++i)
        ranges.push_back(src.push_constant_ranges[i]);
}

}

namespace tr
{

// Ad-hoc binary caching :P SPIR-V is platform independent, so the same
// "binaries" are fine on all GPUs.
static std::map<std::string, shader_source> binaries;

shader_source::shader_source(
    const std::string& path,
    const std::map<std::string, std::string>& defines
){
    std::string res_path = get_resource_path(path);
    fs::path fs_path(res_path);
    std::string ext = fs_path.extension().string();
    std::string dir_path = fs_path.parent_path().string();

    std::string src = load_text_file(res_path);

    // Splice defines into the source
    std::string definition_src = generate_definition_src(defines);

    size_t offset = src.find("#version");
    if(offset == std::string::npos) src = definition_src + src;
    else
    {
        offset = src.find_first_of('\n', offset) + 1;
        src = src.substr(0, offset) + definition_src + src.substr(offset);
    }

    auto it = binaries.find(src);
    if(it == binaries.end())
    {
        glslang::InitializeProcess();

        // Prepare the shader source for glslang
        EShLanguage type = detect_shader_language(ext);
        glslang::TShader shader(type);
        const char* c_str = src.c_str();
        shader.setStrings(&c_str, 1);
        shader.setEnvInput(glslang::EShSourceGlsl, type, glslang::EShClientVulkan, 100);
        shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
        shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);

        TBuiltInResource resources = glslang::DefaultTBuiltInResource;

        EShMessages messages = (EShMessages)(EShMsgSpvRules|EShMsgVulkanRules);

        // Preprocessing
        DirStackFileIncluder includer;
        includer.pushExternalLocalDirectory(dir_path);

        // Compiling
        if(!shader.parse(&resources, 100, ENoProfile, false, false, messages, includer))
            throw std::runtime_error(
                "Failed to compile " + path + ": " + shader.getInfoLog()
            );

        glslang::TProgram program;
        program.addShader(&shader);

        if(!program.link(messages))
            throw std::runtime_error(
                "Failed to link " + path + ": " + shader.getInfoLog()
            );

        spv::SpvBuildLogger logger;
        glslang::SpvOptions options;
        options.generateDebugInfo = true;
        glslang::GlslangToSpv(
            *program.getIntermediate(type), data, &logger, &options
        );

        glslang::FinalizeProcess();

        // glslang has built-in reflection, but it's crap! It doesn't find all
        // things, like blocks that contain unsized arrays. That's why we have
        // to use a separate library for this.
        vk::ShaderStageFlags stage = detect_shader_stage(ext);

        SpvReflectShaderModule mod;
        SpvReflectResult res = spvReflectCreateShaderModule(
            data.size() * sizeof(data[0]),
            data.data(),
            &mod
        );
        if(res != SPV_REFLECT_RESULT_SUCCESS)
            throw std::runtime_error(
                "Failed to reflect " + path + ": " + std::to_string(res)
            );

        // Determine descriptor bindings
        uint32_t count = 0;
        spvReflectEnumerateDescriptorBindings(&mod, &count, nullptr);
        std::vector<SpvReflectDescriptorBinding*> bindings(count);
        spvReflectEnumerateDescriptorBindings(&mod, &count, bindings.data());
        for(auto* binding: bindings)
        {
            this->bindings[binding->name] = binding_info{binding->set,
                vk::DescriptorSetLayoutBinding{
                    binding->binding,
                    vk::DescriptorType(binding->descriptor_type),
                    binding->count,
                    stage
                }
            };
        }

        // Determine push constant range
        spvReflectEnumeratePushConstantBlocks(&mod, &count, nullptr);
        std::vector<SpvReflectBlockVariable*> push_constant_blocks(count);
        spvReflectEnumeratePushConstantBlocks(
            &mod, &count, push_constant_blocks.data()
        );

        for(auto* pc: push_constant_blocks)
        {
            this->push_constant_ranges.push_back({
                stage, pc->offset, pc->size
            });
        }

        spvReflectDestroyShaderModule(&mod);

        binaries[src] = *this;
    }
    else operator=(it->second);
}

void shader_source::clear_binary_cache()
{
    binaries.clear();
}

std::vector<vk::PushConstantRange>
get_push_constant_ranges(const rt_shader_sources& src)
{
    std::vector<vk::PushConstantRange> ranges;
    append_shader_pc_ranges(ranges, src.rgen);

    for(const rt_shader_sources::hit_group& hg: src.rhit)
    {
        append_shader_pc_ranges(ranges, hg.rchit);
        append_shader_pc_ranges(ranges, hg.rahit);
        append_shader_pc_ranges(ranges, hg.rint);
    }

    for(const shader_source& src: src.rmiss)
        append_shader_pc_ranges(ranges, src);

    return ranges;
}

std::vector<vk::PushConstantRange>
get_push_constant_ranges(const raster_shader_sources& src)
{
    std::vector<vk::PushConstantRange> ranges;
    append_shader_pc_ranges(ranges, src.vert);
    append_shader_pc_ranges(ranges, src.frag);
    return ranges;
}

std::vector<vk::PushConstantRange>
get_push_constant_ranges(const shader_source& compute_src)
{
    std::vector<vk::PushConstantRange> ranges;
    append_shader_pc_ranges(ranges, compute_src);
    return ranges;
}

}
