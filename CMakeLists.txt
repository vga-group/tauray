cmake_minimum_required(VERSION 3.16)

project(Tauray LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(VULKAN_VALIDATION "Load Vulkan validation layers by default when present" OFF)

find_package(Threads)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

function(add_glslang)
    # This just gives us a scope so that we don't overwrite CMAKE_BUILD_TYPE.
    set(SKIP_GLSLANG_INSTALL ON CACHE BOOL "" FORCE)
    set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
    set(ENABLE_OPT ON CACHE BOOL "" FORCE)
    set(ENABLE_PCH OFF CACHE BOOL "" FORCE)
    set(ENABLE_CTEST OFF CACHE BOOL "" FORCE)
    set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
    set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_CXX_FLAGS "-Iexternal")
    set(CMAKE_BUILD_TYPE Release)
    add_subdirectory(external/glslang)
endfunction()
add_glslang()

function(add_openxr)
    # This just gives us a scope so that we don't overwrite CMAKE_BUILD_TYPE.
    set(BUILD_WITH_STD_FILESYSTEM OFF CACHE BOOL "" FORCE)
    set(OPENGLES_INCOMPATIBLE ON CACHE BOOL "" FORCE)
    set(OPENGL_INCOMPATIBLE ON CACHE BOOL "" FORCE)
    set(VULKAN_INCOMPATIBLE OFF CACHE BOOL "" FORCE)
    set(BUILD_ALL_EXTENSIONS ON CACHE BOOL "" FORCE)
    set(DYNAMIC_LOADER OFF CACHE BOOL "" FORCE)
    set(CMAKE_BUILD_TYPE Release)
    add_subdirectory(external/OpenXR-SDK EXCLUDE_FROM_ALL)
endfunction()
add_openxr()


set(TAURAY_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
string(LENGTH "${TAURAY_PROJECT_ROOT}/" TAURAY_PROJECT_ROOT_PATH_SIZE)
add_compile_definitions("TAURAY_PROJECT_ROOT_PATH_SIZE=${TAURAY_PROJECT_ROOT_PATH_SIZE}")

add_subdirectory(external/fuchsia_radix_sort)

find_package(SDL2 REQUIRED)
find_package(glm REQUIRED)
find_package(czmq QUIET)
find_package(nng QUIET)
find_package(libcbor QUIET)
find_package(Vulkan REQUIRED)

add_library(tauray-core STATIC
  external/stb_image.c
  external/spirv_reflect.c
  external/tiny_gltf.cc
  external/tinyexr.cc
  external/tinyply.cc
  external/vk_mem_alloc.cc
  src/aabb_scene.cc
  src/acceleration_structure.cc
  src/animation.cc
  src/atlas.cc
  src/basic_pipeline.cc
  src/bmfr_stage.cc
  src/camera.cc
  src/compute_pipeline.cc
  src/context.cc
  src/dependency.cc
  src/descriptor_state.cc
  src/direct_stage.cc
  src/distribution_strategy.cc
  src/dshgi_renderer.cc
  src/dshgi_server.cc
  src/dshgi_client.cc
  src/environment_map.cc
  src/envmap_stage.cc
  src/example_denoiser_stage.cc
  src/feature_stage.cc
  src/frame_client.cc
  src/frame_delay_stage.cc
  src/frame_server.cc
  src/gbuffer.cc
  src/gfx_pipeline.cc
  src/gltf.cc
  src/gpu_buffer.cc
  src/headless.cc
  src/heightfield.cc
  src/light.cc
  src/light_scene.cc
  src/load_balancer.cc
  src/log.cc
  src/looking_glass.cc
  src/looking_glass_composition_stage.cc
  src/material.cc
  src/math.cc
  src/mesh.cc
  src/mesh_object.cc
  src/mesh_scene.cc
  src/misc.cc
  src/model.cc
  src/openxr.cc
  src/options.cc
  src/patched_sphere.cc
  src/path_tracer_stage.cc
  src/placeholders.cc
  src/plane.cc
  src/ply.cc
  src/post_processing_renderer.cc
  src/progress_tracker.cc
  src/radix_sort.cc
  src/raster_stage.cc
  src/raster_renderer.cc
  src/rectangle_packer.cc
  src/render_target.cc
  src/rt_camera_stage.cc
  src/rt_common.cc
  src/rt_stage.cc
  src/rt_renderer.cc
  src/sampler.cc
  src/sampler_table.cc
  src/scene.cc
  src/scene_graph.cc
  src/scene_update_stage.cc
  src/server_context.cc
  src/sh_grid.cc
  src/sh_compact_stage.cc
  src/sh_path_tracer_stage.cc
  src/sh_renderer.cc
  src/shader_source.cc
  src/shadow_map.cc
  src/shadow_map_stage.cc
  src/shadow_map_renderer.cc
  src/skinning_stage.cc
  src/spatial_reprojection_stage.cc
  src/stage.cc
  src/stitch_stage.cc
  src/svgf_stage.cc
  src/taa_stage.cc
  src/tauray.cc
  src/temporal_reprojection_stage.cc
  src/texture.cc
  src/timer.cc
  src/tonemap_stage.cc
  src/tracing.cc
  src/transformable.cc
  src/vkm.cc
  src/whitted_stage.cc
  src/window.cc
  src/z_pass_stage.cc
)

add_executable(tauray
  src/main.cc
)

target_include_directories(tauray-core PUBLIC SYSTEM
    "external"
    "external/SPIRV-Headers/include"
    "external/Vulkan-Headers/include"
)
target_link_libraries(tauray-core PUBLIC SDL2::SDL2 glm::glm glslang SPIRV openxr_loader Vulkan::Vulkan vk-radix-sort)
target_link_libraries(tauray PUBLIC tauray-core SDL2::SDL2main)

if(MSVC)
  target_compile_options(tauray-core PRIVATE /W4 /WX- /MP)
else()
  target_compile_options(tauray-core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NOT czmq_FOUND)
    if(UNIX)
        message(WARNING "CZMQ not found directly, falling back to PkgConfig")
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(libczmq REQUIRED libczmq IMPORTED_TARGET)
        target_link_libraries(tauray-core PUBLIC PkgConfig::libczmq)
    else()
        message(FATAL_ERROR "Please install CZMQ with vcpkg.")
    endif()
else()
    target_link_libraries(tauray-core PUBLIC czmq)
endif()

if(NOT nng_FOUND)
    if(UNIX)
        message(WARNING "NNG not found directly, falling back to PkgConfig")
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(libnng REQUIRED libnng IMPORTED_TARGET)
        target_link_libraries(tauray-core PUBLIC PkgConfig::libnng)
    else()
        message(FATAL_ERROR "Please install NNG with vcpkg.")
    endif()
else()
    target_link_libraries(tauray-core PUBLIC nng::nng)
endif()

if(NOT libcbor_FOUND)
    if(UNIX)
        message(WARNING "libcbor not found directly, falling back to PkgConfig")
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(libcbor REQUIRED libcbor IMPORTED_TARGET)
        target_link_libraries(tauray-core PUBLIC PkgConfig::libcbor)
    else()
        message(FATAL_ERROR "Please install libcbor with vcpkg.")
    endif()
else()
    target_link_libraries(tauray-core PUBLIC ${LIBCBOR_LIBRARY})
endif()

set(DATA_PATH "${CMAKE_INSTALL_PREFIX}/share/tauray")
target_compile_definitions(tauray-core  PUBLIC "TR_RESOURCE_PATH=\"${DATA_PATH}\"")
if(VULKAN_VALIDATION)
    target_compile_definitions(tauray-core PUBLIC ENABLE_VULKAN_VALIDATION)
endif()

target_compile_features(tauray-core PUBLIC cxx_std_17)
set_property(TARGET tauray-core PROPERTY CXX_STANDARD 17)
set_property(TARGET tauray-core PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET tauray-core PROPERTY CXX_EXTENSIONS OFF)

target_compile_features(tauray PUBLIC cxx_std_17)
set_property(TARGET tauray PROPERTY CXX_STANDARD 17)
set_property(TARGET tauray PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET tauray PROPERTY CXX_EXTENSIONS OFF)

install(TARGETS tauray
    RUNTIME DESTINATION bin
)
install(DIRECTORY shader data DESTINATION share/tauray)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING AND NOT MSVC)
        add_subdirectory(test)
    endif()
endif()

find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create documentation (requires Doxygen)" ${DOXYGEN_FOUND})
if(BUILD_DOCUMENTATION)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
