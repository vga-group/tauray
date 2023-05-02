#include "obj.hh"
#include "log.hh"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <numeric>

namespace
{
using namespace tr;

// FIXME: I've been copied from gltf.cc! Make this a utility function
template<typename T>
auto add_unique_named(std::string& name, T& map) -> decltype(map[name])&
{
    std::string candidate_name = name;
    int count = 0;
    while(map.count(candidate_name) != 0)
    {
        candidate_name = name + std::to_string(count++);
    }
    name = candidate_name;
    return map[name];
}

template<typename T>
std::string gen_free_name(std::string& name, T& map)
{
    std::string candidate_name = name;
    int count = 0;
    while(map.count(candidate_name) != 0)
    {
        candidate_name = name + std::to_string(count++);
    }
    candidate_name;
}

std::vector<mesh::vertex> read_vertices(aiMesh* ai_mesh)
{
    std::vector<mesh::vertex> mesh_vert;
    mesh_vert.reserve(ai_mesh->mNumVertices);

    for (unsigned int j = 0; j < ai_mesh->mNumVertices; j++)
    {
        mesh::vertex v;
        v.pos = vec3(
            ai_mesh->mVertices[j].x,
            ai_mesh->mVertices[j].y,
            ai_mesh->mVertices[j].z
        );

        if (ai_mesh->HasNormals())
            v.normal = vec3(
                ai_mesh->mNormals[j].x,
                ai_mesh->mNormals[j].y,
                ai_mesh->mNormals[j].z
            );
        
        if (ai_mesh->HasTangentsAndBitangents())
            v.tangent = vec4(
                ai_mesh->mTangents[j].x,
                ai_mesh->mTangents[j].y,
                ai_mesh->mTangents[j].z,
                // Not sure if this is correct
                1.0f
            );

        if (ai_mesh->HasTextureCoords(0))
            v.uv = vec2(
                ai_mesh->mTextureCoords[0][j].x,
                ai_mesh->mTextureCoords[0][j].y
            );

        mesh_vert.push_back(v);
    }
    return mesh_vert;
}

}

namespace tr
{

scene_graph load_obj(context& ctx, const std::string& path)
{
    TR_LOG("Started loading OBJ scene from ", path);
    scene_graph md;

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
        path, 
        aiProcess_CalcTangentSpace |
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_SortByPType
    );

    if (!scene)
    {
        throw std::runtime_error(
            "Failed to load scene: " + std::string(importer.GetErrorString())
        );
    }

    // TEST MAT
    material mat;

    // Meshes
    for (unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* ai_mesh = scene->mMeshes[i];

        model m;
        mesh* out_mesh = new mesh(ctx);

        std::vector<mesh::vertex>& mesh_vert = out_mesh->get_vertices();
        mesh_vert = read_vertices(ai_mesh);

        std::vector<uint32_t>& mesh_ind = out_mesh->get_indices();

        for (unsigned int j = 0; j < ai_mesh->mNumFaces; j++)
        {
            aiFace face = ai_mesh->mFaces[j];
            for (unsigned int k = 0; k < face.mNumIndices; k++)
            {
                mesh_ind.push_back(face.mIndices[k]);
            }
        }

        if(!ai_mesh->HasNormals())
            out_mesh->calculate_normals();
        if(!ai_mesh->HasTangentsAndBitangents())
            out_mesh->calculate_tangents();

        md.meshes.emplace_back(out_mesh);

        material primitive_material;
        m.add_vertex_group(primitive_material, out_mesh);
        
        std::string tmp_name = ai_mesh->mName.C_Str();
        std::string name = gen_free_name(tmp_name, md.models);
        md.models[name] = std::move(m);

        TR_LOG("Finished loading mesh ", name);

        // Mesh is loaded, still need to an object for it
        std::string obj_name = name + "-obj";
        mesh_object& obj = add_unique_named(obj_name, md.mesh_objects);
        obj.set_model(&md.models[name]);
    }

    for(auto& m: md.meshes)
        m->refresh_buffers();


    TR_LOG("Finished loading OBJ scene ", path);
    return md;
}

}