#include "assimp.hh"
#include "log.hh"
#include "model.hh"
#include "stb_image.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <numeric>
#include <filesystem>

namespace fs = std::filesystem;

namespace
{
using namespace tr;

template<typename T>
T& add_unique_named(scene& s, std::string& name, T&& entry, entity* index = nullptr)
{
    entity id = s.add(std::move(entry), name_component{name});
    if(index) *index = id;
    return *s.get<T>(id);
}

vec2 to_vec2(aiVector3D& v)
{
    return vec2(v.x, v.y);
}

vec3 to_vec3(aiVector3D& v)
{
    return vec3(v.x, v.y, v.z);
}

vec3 to_vec3(aiColor3D& color)
{
    return vec3(color.r, color.g, color.b);
}

vec4 to_vec4(aiColor3D& color)
{
    return vec4(color.r, color.g, color.b, 1.0);
}

std::vector<mesh::vertex> read_vertices(aiMesh* ai_mesh)
{
    std::vector<mesh::vertex> mesh_vert(ai_mesh->mNumVertices, mesh::vertex());

    for(unsigned int i = 0; i < ai_mesh->mNumVertices; i++)
        mesh_vert[i].pos = to_vec3(ai_mesh->mVertices[i]);

    if(ai_mesh->HasNormals())
    {
        for(unsigned int i = 0; i < ai_mesh->mNumVertices; i++)
            mesh_vert[i].normal = to_vec3(ai_mesh->mNormals[i]);
    }

    if(ai_mesh->HasNormals() && ai_mesh->HasTangentsAndBitangents())
    {
        for(unsigned int i = 0; i < ai_mesh->mNumVertices; i++)
        {
            vec3 c = cross(
                to_vec3(ai_mesh->mNormals[i]),
                to_vec3(ai_mesh->mTangents[i])
            );
            float w = sign(dot(to_vec3(ai_mesh->mBitangents[i]), c));

            mesh_vert[i].tangent = vec4(to_vec3(ai_mesh->mTangents[i]), w);
        }
    }

    if(ai_mesh->HasTextureCoords(0))
    {
        for(unsigned int i = 0; i < ai_mesh->mNumVertices; i++)
            mesh_vert[i].uv = to_vec2(ai_mesh->mTextureCoords[0][i]);
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

bool is_opaque(std::vector<uint8_t>& data) {
    for (uint32_t i = 3; i < data.size(); i += 4) {
        if (data[i] < 255) {
            return false;
        }
    }
    return true;
}

std::unique_ptr<texture> read_texture(
    aiTextureType type,
    device_mask dev,
    const aiScene* ai_scene,
    const aiMaterial* ai_mat,
    fs::path& base_path
){
    aiString path;
    if(ai_mat->Get(AI_MATKEY_TEXTURE(type, 0), path) != AI_SUCCESS)
    {
        return nullptr;
    }

    // Check if texture is embedded (e.g in fbx models)
    if(auto ai_texture = ai_scene->GetEmbeddedTexture(path.C_Str()))
    {
        int width, height, components;
        std::vector <uint8_t> image_data;

        // If height is set to 0, the texture is compressed, stbi handles that
        if (ai_texture->mHeight == 0)
        {
            stbi_set_flip_vertically_on_load(false);
            unsigned char *data = stbi_load_from_memory(
                reinterpret_cast<unsigned char*>(ai_texture->pcData),
                ai_texture->mWidth, &width, &height, &components, 4
            );
            image_data = std::vector<uint8_t>(
                data,
                data + width * height * components
            );
            stbi_image_free(data);
        }
        else
        {
            image_data.reserve(ai_texture->mWidth * ai_texture->mHeight * 4);
            for(
                unsigned int i = 0;
                i < ai_texture->mWidth * ai_texture->mHeight;
                i++
            ){
                image_data.push_back(ai_texture->pcData[i].r);
                image_data.push_back(ai_texture->pcData[i].g);
                image_data.push_back(ai_texture->pcData[i].b);
                image_data.push_back(ai_texture->pcData[i].a);
            }
            width = ai_texture->mWidth;
            height = ai_texture->mHeight;
            components = 4;
        }

        bool opaque = is_opaque(image_data);

        auto t = std::unique_ptr<texture>(new texture(
            dev,
            uvec2(width, height),
            1,
            vk::Format::eR8G8B8A8Unorm,
            image_data.size(),
            image_data.data(),
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eSampled,
            vk::ImageLayout::eShaderReadOnlyOptimal
        ));

        t->set_opaque(opaque);

        return t;
    }

    // Texture is not embedded.
    // It should be in a file relative to the model file.
    return std::unique_ptr<texture>(
        new texture(dev, (base_path / path.C_Str()).string())
    );
}

material create_material(
    device_mask dev,
    scene_assets& md,
    fs::path& base_path,
    const aiScene* ai_scene,
    const aiMaterial* ai_mat
){
    // Almost up to date material docs (doesn't include PBR properties):
    // https://assimp-docs.readthedocs.io/en/latest/usage/use_the_lib.html?highlight=matkey#constants

    material mat;

    aiString name;
    if(ai_mat->Get(AI_MATKEY_NAME, name) == AI_SUCCESS)
    {
        mat.name = name.C_Str();
    }

    bool is_pbr = false;
    aiShadingMode shading_mode;
    if(ai_mat->Get(AI_MATKEY_SHADING_MODEL, shading_mode) == AI_SUCCESS)
    {
        if (shading_mode == aiShadingMode_PBR_BRDF) {
            is_pbr = true;
        }
    }

    // Check if pbr is using metallic/roughness or specular/glossiness workflow
    if(is_pbr)
    {
        // If Specular/glossiness is used, fall back to phong
        float glossiness;
        if (ai_mat->Get(AI_MATKEY_GLOSSINESS_FACTOR, glossiness) == AI_SUCCESS)
        {
            is_pbr = false;
        }
    }

    if(is_pbr)
    {
        // Read pbr properties.
        aiColor3D base;
        if(ai_mat->Get(AI_MATKEY_BASE_COLOR, base) == AI_SUCCESS)
        {
            mat.albedo_factor = to_vec4(base);
        }
        if(auto t = read_texture(aiTextureType_BASE_COLOR,
            dev, ai_scene, ai_mat, base_path))
        {
            md.textures.push_back(std::move(t));
            mat.albedo_tex.first = md.textures.back().get();
        }

        float metallic;
        if(ai_mat->Get(AI_MATKEY_METALLIC_FACTOR, metallic) == AI_SUCCESS)
        {
            mat.metallic_factor = metallic;
        }
        float roughness;
        if(ai_mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness) == AI_SUCCESS)
        {
            mat.roughness_factor = roughness;
        }
        if(auto t = read_texture(aiTextureType_DIFFUSE_ROUGHNESS,
            dev, ai_scene, ai_mat, base_path))
        {
            md.textures.push_back(std::move(t));
            mat.metallic_roughness_tex.first = md.textures.back().get();
        }

        float transmission;
        if(ai_mat->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmission) == AI_SUCCESS)
        {
            mat.transmittance = transmission;
        }
    }
    else
    {
        // Read phong properties. This mode is assumed, although shading_mode
        // could specify something else.
        aiColor3D albedo;
        if(ai_mat->Get(AI_MATKEY_COLOR_DIFFUSE, albedo) == AI_SUCCESS) {
            mat.albedo_factor = to_vec4(albedo);
        }
        if(auto t = read_texture(aiTextureType_DIFFUSE,
            dev, ai_scene, ai_mat, base_path))
        {
            md.textures.push_back(std::move(t));
            mat.albedo_tex.first = md.textures.back().get();
        }

        aiColor3D transparent;
        if(ai_mat->Get(AI_MATKEY_COLOR_TRANSPARENT, transparent) == AI_SUCCESS)
        {
            mat.transmittance =
                1.0f - (transparent.r + transparent.g + transparent.b) / 3.0f;
        }


        // TODO: Make metals look ok when pbr values don't exist
        // float shininess;
        // if(mat->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS) {
        //     m.metallic_factor = shininess;
        //     m.roughness_factor = shininess;
        // }
    }

    // Read parameter shared by both pbr and phong

    float opacity;
    if(ai_mat->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS)
    {
        mat.albedo_factor.a = opacity;
    }

    if(auto t = read_texture(aiTextureType_NORMALS,
        dev, ai_scene, ai_mat, base_path))
    {
        md.textures.push_back(std::move(t));
        mat.normal_tex.first = md.textures.back().get();
    }

    float ior;
    if(ai_mat->Get(AI_MATKEY_REFRACTI, ior) == AI_SUCCESS)
    {
        mat.ior = ior;
    }

    aiColor3D emissive;
    if(ai_mat->Get(AI_MATKEY_COLOR_EMISSIVE, emissive) == AI_SUCCESS)
    {
        mat.emission_factor = to_vec3(emissive);
    }
    if(auto t = read_texture(aiTextureType_EMISSIVE,
        dev, ai_scene, ai_mat, base_path))
    {
        md.textures.push_back(std::move(t));
        mat.emission_tex.first = md.textures.back().get();
    }

    bool twosided;
    if(ai_mat->Get(AI_MATKEY_TWOSIDED, twosided) == AI_SUCCESS)
    {
        mat.double_sided = twosided;
    }

    return mat;
}

}

namespace tr
{

scene_assets load_assimp(device_mask dev, scene& s, const std::string& path)
{
    TR_LOG("Started loading scene from ", path);
    fs::path base_path = fs::path(path).parent_path();

    scene_assets md;

    Assimp::Importer importer;
    const aiScene* ai_scene = importer.ReadFile(
        path,
        aiProcess_CalcTangentSpace |
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_SortByPType
    );

    if(!ai_scene)
    {
        throw std::runtime_error(
            "Failed to load scene: " + std::string(importer.GetErrorString())
        );
    }


    for(unsigned int i = 0; i < ai_scene->mNumMeshes; i++)
    {
        TR_LOG("Loading mesh ", i);
        aiMesh* ai_mesh = ai_scene->mMeshes[i];

        model m;
        mesh* out_mesh = new mesh(dev);

        out_mesh->get_vertices() = read_vertices(ai_mesh);
        out_mesh->get_indices() = read_indices(ai_mesh);

        if(!ai_mesh->HasNormals())
            out_mesh->calculate_normals();
        if(!ai_mesh->HasTangentsAndBitangents())
            out_mesh->calculate_tangents();

        md.meshes.emplace_back(out_mesh);

        material mat = create_material(
            dev,
            md,
            base_path,
            ai_scene,
            ai_scene->mMaterials[ai_mesh->mMaterialIndex]
        );
        m.add_vertex_group(mat, out_mesh);

        std::string name = ai_mesh->mName.C_Str();

        // Mesh is loaded, still need to an object for it
        entity id;
        add_unique_named(s, name, transformable{}, &id);
        s.attach(id, std::move(m));
    }

    for(auto& m: md.meshes)
        m->refresh_buffers();

    TR_LOG("Finished loading scene ", path);
    return md;
}

}
