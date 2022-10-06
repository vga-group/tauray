#include "ply.hh"
#include "tinyply.h"
#include <fstream>
#include <numeric>

namespace
{
using namespace tr;

template<typename T>
void cast_ply_data(uint8_t* data, tinyply::Type t, int, T& out)
{
    switch(t)
    {
    case tinyply::Type::INT8:
        out = *reinterpret_cast<int8_t*>(data);
        break;
    case tinyply::Type::UINT8:
        out = *reinterpret_cast<uint8_t*>(data);
        break;
    case tinyply::Type::INT16:
        out = *reinterpret_cast<int16_t*>(data);
        break;
    case tinyply::Type::UINT16:
        out = *reinterpret_cast<uint16_t*>(data);
        break;
    case tinyply::Type::INT32:
        out = *reinterpret_cast<int32_t*>(data);
        break;
    case tinyply::Type::UINT32:
        out = *reinterpret_cast<uint32_t*>(data);
        break;
    case tinyply::Type::FLOAT32:
        out = *reinterpret_cast<float*>(data);
        break;
    case tinyply::Type::FLOAT64:
        out = *reinterpret_cast<double*>(data);
        break;
    default:
        out = 0;
        break;
    }
}

template<int size, typename T>
void cast_ply_data(
    uint8_t* data, tinyply::Type t, int channels, vec<size, T>& out
){
    int component_size = tinyply::PropertyTable[t].stride;
    out = glm::vec<size, T>(0);
    for(int i = 0; i < std::min(channels, (int)size); ++i)
        cast_ply_data(data+i*component_size, t, channels, out[i]);
}

template<typename T>
std::vector<T> read_ply_data(
    tinyply::PlyData& data,
    bool unravel_channels = false
){
    std::vector<T> res;
    if(data.count == 0) return res;

    int stride = data.buffer.size_bytes() / data.count;
    int component_size = tinyply::PropertyTable[data.t].stride;
    int channels = stride / component_size;
    if(unravel_channels)
    {
        stride = component_size;
        res.resize(data.count * channels);
        channels = 1;
    }
    else res.resize(data.count);

    uint8_t* data_ptr = data.buffer.get();
    for(T& entry: res)
    {
        cast_ply_data(data_ptr, data.t, channels, entry);
        data_ptr += stride;
    }
    return res;
}

}

namespace tr
{

void load_ply_refresh(scene_graph& sg, std::istream& stream)
{
    tinyply::PlyFile ply;
    if(!ply.parse_header(stream))
        throw std::runtime_error("Failed to read PLY file");

    // TODO: Colors? Tripstrips? Animation stuff?
    std::shared_ptr<tinyply::PlyData> ply_pos, ply_normals, ply_tangents, ply_uv, ply_indices;

    ply_pos = ply.request_properties_from_element("vertex", { "x", "y", "z" });

    try { ply_normals = ply.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
    catch(...) {}

    try { ply_tangents = ply.request_properties_from_element("vertex", { "tx", "ty", "tz" }); }
    catch(...) {}

    try { ply_uv = ply.request_properties_from_element("vertex", { "u", "v" }); }
    catch(...) {}

    try { ply_indices = ply.request_properties_from_element("face", { "vertex_indices" }, 3); }
    catch(...) {}

    ply.read(stream);

    std::vector<uint32_t> indices;

    if(ply_indices)
    {
        indices = read_ply_data<uint32_t>(*ply_indices, true);
    }
    else
    {
        indices.resize(ply_pos->count);
        std::iota(indices.begin(), indices.end(), 0);
    }

    std::vector<vec3> vert_pos;
    std::vector<vec3> vert_norm;
    std::vector<vec2> vert_uv;
    std::vector<vec4> vert_tangent;
    if(ply_pos) vert_pos = read_ply_data<vec3>(*ply_pos);
    if(ply_normals) vert_norm = read_ply_data<vec3>(*ply_normals);
    if(ply_uv) vert_uv = read_ply_data<vec2>(*ply_uv);
    if(ply_tangents) vert_tangent = read_ply_data<vec4>(*ply_tangents);

    std::vector<mesh::vertex> vertices;

    vertices.resize(vert_pos.size());
    vert_norm.resize(vert_pos.size(), vec3(0));
    vert_uv.resize(vert_pos.size(), vec2(0));
    vert_tangent.resize(vert_pos.size(), vec4(0));
    for(size_t i = 0; i < vert_pos.size(); ++i)
        vertices[i] = {vert_pos[i], vert_norm[i], vert_uv[i], vert_tangent[i]};

    mesh* primitive = sg.meshes.back().get();
    primitive->get_vertices() = std::move(vertices);
    primitive->get_indices() = std::move(indices);
    if(!ply_normals) primitive->calculate_normals();
    // We don't need tangents because PLY models don't have materials anyway...
    //if(!ply_tangents) primitive->calculate_tangents();
    if(vert_pos.size() > 0)
        primitive->refresh_buffers();
}

void init_ply(
    context& ctx,
    scene_graph& sg,
    const std::string& name,
    bool force_single_sided
){
    material mat;
    mat.double_sided = !force_single_sided;

    mesh* primitive = new mesh(ctx);
    primitive->set_opaque(true);
    primitive->get_vertices() = {{}, {}, {}};
    primitive->get_indices() = {0,1,2};
    sg.meshes.emplace_back(primitive);
    model mod;
    mod.add_vertex_group(mat, sg.meshes.back().get());
    sg.models[name] = std::move(mod);
    sg.mesh_objects[name] = mesh_object(&sg.models[name]);
}

scene_graph load_ply(
    context& ctx,
    const std::string& path,
    bool force_single_sided
){
    std::ifstream input_file(path, std::ifstream::in);
    if(!input_file)
        throw std::runtime_error("Failed to open "+path);

    scene_graph sg;
    init_ply(ctx, sg, path, force_single_sided);
    load_ply_refresh(sg, input_file);

    return sg;
}


ply_streamer::ply_streamer(
    context& ctx,
    scene_graph& s,
    const std::string& path,
    bool force_single_sided
):  sg(&s), input(path, std::ifstream::in),
    pending_data_size(0), pending_data_offset(0), line_length(0)
{
    if(!input)
        throw std::runtime_error("Failed to open "+path);

    init_ply(ctx, s, path, force_single_sided);
    s.meshes[0]->refresh_buffers();
    pending.resize(4096);
}

bool ply_streamer::refresh()
{
    while(!read_pending())
    {
        std::streamsize size = input.readsome(pending.data(), pending.size());
        if(size <= 0)
            return false;
        pending_data_offset = 0;
        pending_data_size = size;
    }

    clipped_input.seekg(0, std::ios_base::beg);
    load_ply_refresh(*sg, clipped_input);
    clipped_input.str("");
    return true;
}

bool ply_streamer::read_pending()
{
    // Read until end of current segment, returning true, or until all is read,
    // returning false.
    for(
        size_t i = pending_data_offset;
        i < pending_data_offset + pending_data_size;
        ++i
    ){
        if(pending[i] == '\n')
        {
            size_t length = i+1-pending_data_offset;
            clipped_input.write(pending.data() + pending_data_offset, length);
            pending_data_offset = i+1;
            pending_data_size -= length;
            line_length += length;
            size_t prev_line_length = line_length;
            line_length = 0;
            if(prev_line_length == 1)
                return true;
        }
    }

    if(pending_data_size > 0)
    {
        clipped_input.write(pending.data() + pending_data_offset, pending_data_size);
        line_length += pending_data_size;
        pending_data_offset = 0;
        pending_data_size = 0;
    }
    return false;
}

}
