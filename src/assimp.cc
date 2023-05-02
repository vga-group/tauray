#include "assimp.hh"
#include "log.hh"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <numeric>
#include <filesystem>

namespace fs = std::filesystem;

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
    return candidate_name;
}

std::vector<mesh::vertex> read_vertices(aiMesh* ai_mesh)
{
    std::vector<mesh::vertex> mesh_vert;
    mesh_vert.reserve(ai_mesh->mNumVertices);

    for(unsigned int j = 0; j < ai_mesh->mNumVertices; j++)
    {
        mesh::vertex v;
        v.pos = vec3(
            ai_mesh->mVertices[j].x,
            ai_mesh->mVertices[j].y,
            ai_mesh->mVertices[j].z
        );

        if(ai_mesh->HasNormals())
            v.normal = vec3(
                ai_mesh->mNormals[j].x,
                ai_mesh->mNormals[j].y,
                ai_mesh->mNormals[j].z
            );
        
        if(ai_mesh->HasTangentsAndBitangents())
            v.tangent = vec4(
                ai_mesh->mTangents[j].x,
                ai_mesh->mTangents[j].y,
                ai_mesh->mTangents[j].z,
                // Not sure if this is correct
                1.0f
            );

        if(ai_mesh->HasTextureCoords(0))
            v.uv = vec2(
                ai_mesh->mTextureCoords[0][j].x,
                ai_mesh->mTextureCoords[0][j].y
            );

        mesh_vert.push_back(v);
    }
    return mesh_vert;
}

std::vector<uint32_t> read_indices(aiMesh* ai_mesh)
{
    std::vector<uint32_t> mesh_ind;
    mesh_ind.reserve(ai_mesh->mNumFaces * 3);
    for(unsigned int j = 0; j < ai_mesh->mNumFaces; j++)
    {
        aiFace face = ai_mesh->mFaces[j];
        for(unsigned int k = 0; k < face.mNumIndices; k++)
        {
            mesh_ind.push_back(face.mIndices[k]);
        }
    }
    return mesh_ind;
}


vec3 to_vec3(aiColor3D& color)
{
    return vec3(color.r, color.g, color.b);
}

vec4 to_vec4(aiColor3D& color)
{
    return vec4(color.r, color.g, color.b, 1.0);
}

material create_material(
    context& ctx,
    scene_graph& md,
    fs::path& base_path,
    aiMaterial* mat
){
    // Assimp material docs: https://assimp.sourceforge.net/lib_html/materials.html

    material m;

    aiString name;
    if(mat->Get(AI_MATKEY_NAME, name) == AI_SUCCESS) {
        m.name = name.C_Str();
    }

    aiColor3D albedo;
    if(mat->Get(AI_MATKEY_COLOR_DIFFUSE, albedo) == AI_SUCCESS) {
        m.albedo_factor = to_vec4(albedo);
    }
    float opacity;
    if(mat->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS) {
        m.albedo_factor.a = opacity;
    }

    aiString diffuse_path;
    if(
        mat->Get(AI_MATKEY_TEXTURE(aiTextureType_DIFFUSE, 0), diffuse_path) ==
        AI_SUCCESS
    ){
        md.textures.emplace_back(
            new texture(ctx, base_path / diffuse_path.C_Str())
        );
        m.albedo_tex.first = md.textures.back().get();
    }

    // TODO: Make metals look ok
    // float shininess;
    // if(mat->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS) {
    //     m.metallic_factor = shininess;
    //     m.roughness_factor = shininess;
    // }

    aiString normal_path;
    if(
        mat->Get(AI_MATKEY_TEXTURE(aiTextureType_NORMALS, 0), normal_path) ==
        AI_SUCCESS
    ){
        md.textures.emplace_back(
            new texture(ctx, base_path / normal_path.C_Str())
        );
        m.normal_tex.first = md.textures.back().get();
    }

    float ior;
    if(mat->Get(AI_MATKEY_REFRACTI, ior) == AI_SUCCESS) {
        m.ior = ior;
    }

    aiColor3D emissive;
    if(mat->Get(AI_MATKEY_COLOR_EMISSIVE, emissive) == AI_SUCCESS) {
        m.emission_factor = to_vec3(emissive);
    }

    aiString emissive_path;
    if(
        mat->Get(AI_MATKEY_TEXTURE(aiTextureType_EMISSIVE, 0), emissive_path) ==
        AI_SUCCESS
    ){
        md.textures.emplace_back(
            new texture(ctx, base_path / emissive_path.C_Str())
        );
        m.emission_tex.first = md.textures.back().get();
    }

    bool twosided;
    if(mat->Get(AI_MATKEY_TWOSIDED, twosided) == AI_SUCCESS) {
        m.double_sided = twosided;
    }

    return m;
}

}

namespace tr
{

scene_graph load_assimp(context& ctx, const std::string& path)
{
    TR_LOG("Started loading scene from ", path);
    fs::path base_path = fs::path(path).parent_path();

    scene_graph md;

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
        path, 
        aiProcess_CalcTangentSpace |
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_SortByPType
    );

    if(!scene)
    {
        throw std::runtime_error(
            "Failed to load scene: " + std::string(importer.GetErrorString())
        );
    }


    for(unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* ai_mesh = scene->mMeshes[i];

        model m;
        mesh* out_mesh = new mesh(ctx);

        out_mesh->get_vertices() = read_vertices(ai_mesh);
        out_mesh->get_indices() = read_indices(ai_mesh);
       
        if(!ai_mesh->HasNormals())
            out_mesh->calculate_normals();
        if(!ai_mesh->HasTangentsAndBitangents())
            out_mesh->calculate_tangents();

        md.meshes.emplace_back(out_mesh);

        material mat = create_material(
            ctx,
            md,
            base_path,
            scene->mMaterials[ai_mesh->mMaterialIndex]
        );
        m.add_vertex_group(mat, out_mesh);
        
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

    TR_LOG("Finished loading scene ", path);
    return md;
}

}