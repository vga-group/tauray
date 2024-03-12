#ifndef TAURAY_SHADER_SOURCE_HH
#define TAURAY_SHADER_SOURCE_HH
#include <cstdint>
#include <cstddef>
#include <utility>
#include <vector>
#include <map>
#include "context.hh"

namespace tr
{

struct shader_source
{
    shader_source() = default;
    // Reads and compiles the given shader, inserting the 'preamble' after
    // the #version line.
    shader_source(
        const std::string& path,
        const std::map<std::string, std::string>& defines = {}
    );

    shader_source(const shader_source& other) = default;
    shader_source(shader_source&& other) = default;

    shader_source& operator=(const shader_source& other) = default;
    shader_source& operator=(shader_source&& other) = default;

    struct binding_info
    {
        uint32_t set;
        vk::DescriptorSetLayoutBinding binding;
    };
    std::map<std::string /* name */, binding_info> bindings;
    std::vector<vk::PushConstantRange> push_constant_ranges;
    std::vector<uint32_t> data;

    static void clear_binary_cache();
};

struct raster_shader_sources
{
    shader_source vert = {};
    shader_source frag = {};
};

struct rt_shader_sources
{
    shader_source rgen = {};

    struct hit_group
    {
        vk::RayTracingShaderGroupTypeKHR type =
            vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup;
        shader_source rchit = {};
        shader_source rahit = {};
        shader_source rint = {};
    };

    std::vector<hit_group> rhit = {};
    std::vector<shader_source> rmiss = {};
};


std::map<std::string /* name */, std::pair<uint32_t /*set*/, uint32_t /* binding */>>
    get_binding_names(const rt_shader_sources& src);

std::map<std::string /* name */, std::pair<uint32_t /*set*/, uint32_t /* binding */>>
    get_binding_names(const raster_shader_sources& src);

std::map<std::string /* name */, std::pair<uint32_t /*set*/, uint32_t /* binding */>>
    get_binding_names(const shader_source& compute_src);

std::vector<std::vector<vk::DescriptorSetLayoutBinding>> get_bindings(
    const rt_shader_sources& src,
    const std::map<
        std::string /* binding name */,
        uint32_t /* count */
    >& count_overrides = {}
);
std::vector<std::vector<vk::DescriptorSetLayoutBinding>> get_bindings(
    const raster_shader_sources& src,
    const std::map<std::string, uint32_t>& count_overrides = {}
);
std::vector<std::vector<vk::DescriptorSetLayoutBinding>> get_bindings(
    const shader_source& compute_src,
    const std::map<std::string, uint32_t>& count_overrides = {}
);

std::vector<vk::PushConstantRange> get_push_constant_ranges(const rt_shader_sources& src);
std::vector<vk::PushConstantRange> get_push_constant_ranges(const raster_shader_sources& src);
std::vector<vk::PushConstantRange> get_push_constant_ranges(const shader_source& compute_src);

}

#endif
