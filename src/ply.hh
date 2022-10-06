#ifndef TAURAY_PLY_HH
#define TAURAY_PLY_HH
#include "scene_graph.hh"
#include <fstream>

namespace tr
{

// Replaces the existing mesh data with the PLY-data from the istream.
void load_ply_refresh(scene_graph& sg, std::istream& stream);

// Initializes a scene for use with load_ply_refresh
void init_ply(
    context& ctx,
    scene_graph& s,
    const std::string& name,
    bool force_single_sided = false
);

scene_graph load_ply(
    context& ctx,
    const std::string& path,
    bool force_single_sided = false
);

struct ply_streamer
{
    scene_graph* sg;
    std::ifstream input;
    std::vector<char> pending;
    size_t pending_data_size;
    size_t pending_data_offset;
    size_t line_length; 
    std::stringstream clipped_input;

    ply_streamer(
        context& ctx,
        scene_graph& s,
        const std::string& path,
        bool force_single_sided
    );

    // If this returns true, a mesh in the scene has been updated and you must
    // set_scene the renderer again.
    bool refresh();

    bool read_pending();
};

}

#endif


