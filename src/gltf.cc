#include "gltf.hh"
#include "scene.hh"
#include "log.hh"
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"
#include "stb_image.h"
#include <glm/gtc/type_ptr.hpp>
#include "misc.hh"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_set>

namespace
{
using namespace tr;

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

void flip_lines(unsigned pitch, unsigned char* line_a, unsigned char* line_b)
{
    for(unsigned i = 0; i < pitch; ++i)
    {
        unsigned char tmp = line_a[i];
        line_a[i] = line_b[i];
        line_b[i] = tmp;
    }
}

void flip_vector_image(std::vector<unsigned char>& image, unsigned height)
{
    unsigned pitch = image.size() / height;
    for(unsigned i = 0; i < height/2; ++i)
        flip_lines(
            pitch,
            image.data()+i*pitch,
            image.data()+(height-1-i)*pitch
        );
}

bool check_opaque(tinygltf::Image& img)
{
    if(img.component != 4) return true;
    if(img.bits == 8)
    {
        // Check that every fourth (alpha) value is 255.
        for(size_t i = 3; i < img.image.size(); i += 4)
            if(img.image[i] != 255)
                return false;
        return true;
    }
    return false;
}

template<typename T>
vec4 vector_to_vec4(const std::vector<T>& v, float fill_value = 0.0f)
{
    vec4 ret(fill_value);
    for(size_t i = 0; i < std::min<uint64_t>(4, v.size()); ++i)
        ret[i] = v[i];
    return ret;
}

template<typename T>
void cast_gltf_data(uint8_t* data, int componentType, int, T& out)
{
    switch(componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_BYTE:
        out = *reinterpret_cast<int8_t*>(data);
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        out = *reinterpret_cast<uint8_t*>(data);
        break;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
        out = *reinterpret_cast<int16_t*>(data);
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        out = *reinterpret_cast<uint16_t*>(data);
        break;
    case TINYGLTF_COMPONENT_TYPE_INT:
        out = *reinterpret_cast<int32_t*>(data);
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        out = *reinterpret_cast<uint32_t*>(data);
        break;
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
        out = *reinterpret_cast<float*>(data);
        break;
    case TINYGLTF_COMPONENT_TYPE_DOUBLE:
        out = *reinterpret_cast<double*>(data);
        break;
    }
}

template<int size, typename T>
void cast_gltf_data(
    uint8_t* data, int componentType, int type, vec<size, T>& out
){
    int component_size = tinygltf::GetComponentSizeInBytes(componentType);
    int components = tinygltf::GetNumComponentsInType(type);
    out = glm::vec<size, T>(0);
    for(int i = 0; i < min(components, (int)size); ++i)
        cast_gltf_data(data+i*component_size, componentType, type, out[i]);
}

template<int C, int R, typename T>
void cast_gltf_data(
    uint8_t* data, int componentType, int type, mat<C, R, T>& out
){
    int component_size = tinygltf::GetComponentSizeInBytes(componentType);
    int components = tinygltf::GetNumComponentsInType(type);
    out = glm::mat<C, R, T>(1);
    for(int x = 0; x < C; ++x)
    for(int y = 0; y < R; ++y)
    {
        int i = y+x*C;
        if(i < components)
            cast_gltf_data(
                data+i*component_size, componentType, type, out[x][y]
            );
    }
}

void cast_gltf_data(
    uint8_t* data, int componentType, int type, quat& out
){
    vec4 tmp;
    cast_gltf_data(data, componentType, type, tmp);
    out = quat(tmp.w, tmp.x, tmp.y, tmp.z);
}

template<typename T>
std::vector<T> read_accessor(tinygltf::Model& model, int index)
{
    tinygltf::Accessor& accessor = model.accessors[index];
    tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    tinygltf::Buffer& buf = model.buffers[view.buffer];

    std::vector<T> res(accessor.count);
    int stride = accessor.ByteStride(view);

    size_t offset = view.byteOffset + accessor.byteOffset;
    for(T& entry: res)
    {
        uint8_t* data = buf.data.data() + offset;
        cast_gltf_data(data, accessor.componentType, accessor.type, entry);
        offset += stride;
    }
    return res;
}

template<typename T>
std::vector<animation::sample<T>> read_animation_accessors(
    tinygltf::Model& model, int input, int output
){
    std::vector<animation::sample<T>> res;

    std::vector<float> timestamps = read_accessor<float>(model, input);
    std::vector<T> data = read_accessor<T>(model, output);

    bool has_tangents = data.size() >= 3*timestamps.size();
    res.resize(timestamps.size());
    for(size_t i = 0; i < res.size(); ++i)
    {
        // Convert timestamps into microseconds
        res[i].timestamp = round(timestamps[i]*1000000);
        if(has_tangents)
        {
            res[i].in_tangent = data[i*3];
            res[i].data = data[i*3+1];
            res[i].out_tangent = data[i*3+2];
        }
        else res[i].data = data[i];
    }

    return res;
}

texture* get_texture(tinygltf::Model& model, scene_graph& md, int index)
{
    if(index == -1) return nullptr;
    return md.textures[model.textures[index].source].get();
}

material create_material(
    tinygltf::Material& mat,
    tinygltf::Model& model,
    scene_graph& md
){
    material m;
    m.albedo_factor = vector_to_vec4(mat.pbrMetallicRoughness.baseColorFactor);

    m.albedo_tex.first =
        get_texture(model, md, mat.pbrMetallicRoughness.baseColorTexture.index);

    m.metallic_factor = mat.pbrMetallicRoughness.metallicFactor;
    m.roughness_factor = mat.pbrMetallicRoughness.roughnessFactor;

    m.metallic_roughness_tex.first = get_texture(
        model, md, mat.pbrMetallicRoughness.metallicRoughnessTexture.index
    );

    m.normal_factor = 1.0f;
    m.normal_tex.first = get_texture(model, md, mat.normalTexture.index);

    m.ior = 1.45f;

    m.emission_factor = vector_to_vec4(mat.emissiveFactor);
    m.emission_tex.first = get_texture(model, md, mat.emissiveTexture.index);

    m.double_sided = mat.doubleSided;
    m.name = mat.name;

    bool discard_tr_emission = false;

    if(mat.extensions.count("KHR_materials_emissive_strength"))
    {
        const tinygltf::Value& emissive_ext = mat.extensions["KHR_materials_emissive_strength"];
        if(emissive_ext.Has("emissiveStrength"))
        {
            m.emission_factor *= emissive_ext.Get("emissiveStrength").GetNumberAsDouble();
            discard_tr_emission = true;
        }
    }

    if(mat.pbrMetallicRoughness.extensions.count("TR_data"))
    {
        tinygltf::Value* tr_data = &mat.pbrMetallicRoughness.extensions["TR_data"];
        if(tr_data->Has("transmission"))
        {
            m.transmittance = tr_data->Get("transmission").GetNumberAsDouble();
        }
        if(tr_data->Has("ior"))
        {
            m.ior = tr_data->Get("ior").GetNumberAsDouble();
        }
        if(!discard_tr_emission && tr_data->Has("emission"))
        {
            m.emission_factor = vec3(
                tr_data->Get("emission").Get(0).GetNumberAsDouble(),
                tr_data->Get("emission").Get(1).GetNumberAsDouble(),
                tr_data->Get("emission").Get(2).GetNumberAsDouble()
            );
        }
    }

    if(mat.extensions.count("KHR_materials_transmission"))
    {
        tinygltf::Value* transmission_data =
            &mat.extensions["KHR_materials_transmission"];
        if(transmission_data->Has("transmissionFactor"))
        {
            m.transmittance = transmission_data->Get("transmissionFactor").GetNumberAsDouble();
        }
    }

    if(mat.extensions.count("KHR_materials_ior"))
    {
        tinygltf::Value* ior_data = &mat.extensions["KHR_materials_ior"];
        if(ior_data->Has("ior"))
        {
            m.ior = ior_data->Get("ior").GetNumberAsDouble();
        }
    }
    return m;
}

struct skin
{
    int root;
    std::vector<int> joints;
    std::vector<mat4> inverse_bind_matrices;
    std::unordered_map<int, int> node_index_to_skin_index;
    std::vector<animated_node*> joint_nodes;
    std::unordered_set<model*> related_models;
    animated_node* root_node = nullptr;

    std::vector<model::joint_data> build_joint_data() const
    {
        std::vector<model::joint_data> model_joints(joints.size());

        for(size_t i = 0; i < joints.size(); ++i)
        {
            model_joints[i].node = joint_nodes[i];
            model_joints[i].inverse_bind_matrix = inverse_bind_matrices[i];
        }
        return model_joints;
    }
};

struct node_meta_info
{
    std::unordered_map<int /*node*/, animation_pool*> animations;
    std::vector<skin> skins;
    std::unordered_map<int, std::vector<int>> node_to_skin;

    // These are passed in meta info due to the unfortunate way Blender's glTF
    // export works (Blender's light nodes aren't the actual light nodes but
    // just parents :/)
    float light_angle = 0.0f;
    float light_radius = 0.0f;
};

void load_gltf_node(
    tinygltf::Model& model,
    tinygltf::Scene& scene,
    int node_index,
    scene_graph& data,
    transformable_node* parent,
    node_meta_info& meta,
    bool static_lock
){
    tinygltf::Node& node = model.nodes[node_index];
    transformable_node* tnode = nullptr;
    animated_node* anode = nullptr;

    tinygltf::Value* tr_data = nullptr;
    if(node.extensions.count("TR_data"))
    {
        tr_data = &node.extensions["TR_data"];
        if(tr_data->Has("light"))
        {
            const tinygltf::Value* light_data = &tr_data->Get("light");
            if(light_data->Has("angle"))
                meta.light_angle = light_data->Get("angle").GetNumberAsDouble();
            if(light_data->Has("radius"))
                meta.light_radius = light_data->Get("radius").GetNumberAsDouble();
        }
    }

    // Add object if mesh is present
    if(node.mesh != -1)
    {
        mesh_object& obj = add_unique_named(node.name, data.mesh_objects);
        obj.set_model(&data.models[model.meshes[node.mesh].name]);
        if(tr_data && tr_data->Has("mesh"))
        {
            tinygltf::Value mesh = tr_data->Get("mesh");
            obj.set_shadow_terminator_offset(
                mesh.Get("shadow_terminator_offset").GetNumberAsDouble()
            );
        }

        if(node.skin != -1)
        {
            static_lock = false;
            meta.skins[node.skin].related_models.insert(
                const_cast<class model*>(obj.get_model())
            );
        }
        anode = &obj;
        tnode = anode;
    }
    // Add camera if that's present
    else if(node.camera != -1)
    {
        camera cam;
        tinygltf::Camera& c = model.cameras[node.camera];
        // Cameras can be moved dynamically basically always.
        static_lock = false;

        if(c.type == "perspective")
        {
            cam.perspective(
                glm::degrees(c.perspective.yfov), c.perspective.aspectRatio,
                c.perspective.znear, c.perspective.zfar
            );
        }
        else if(c.type == "orthographic")
            cam.ortho(
                -0.5*c.orthographic.xmag, 0.5*c.orthographic.xmag,
                -0.5*c.orthographic.ymag, 0.5*c.orthographic.ymag,
                c.orthographic.znear, c.orthographic.zfar
            );

        // Add object to graph
        add_unique_named(node.name, data.cameras) = std::move(cam);
        anode = &data.cameras[node.name];
        tnode = anode;
    }
    // Add light, if present.
    else if(node.extensions.count("KHR_lights_punctual"))
    {
        tinygltf::Light& l = model.lights[
            node.extensions["KHR_lights_punctual"].Get("light").Get<int>()
        ];

        // Apparently Blender's gltf exporter is broken in terms of light
        // intensity as of writing this, so the multipliers here are just magic
        // numbers. Fix this when this issue is solved:
        // https://github.com/KhronosGroup/glTF-Blender-IO/issues/564
        vec3 color(vector_to_vec4(l.color) * (float)l.intensity);
        if(l.type == "directional")
        {
            directional_light* dl = &add_unique_named(node.name, data.directional_lights);
            anode = dl;
            dl->set_color(color);
            dl->set_angle(degrees(meta.light_angle));
        }
        else if(l.type == "point")
        {
            point_light* pl = &add_unique_named(node.name, data.point_lights);
            anode = pl;
            pl->set_color(color/float(4*M_PI));
            pl->set_radius(meta.light_radius);
        }
        else if(l.type == "spot")
        {
            spotlight* sl = &add_unique_named(node.name, data.spotlights);
            anode = sl;
            sl->set_color(color/float(4*M_PI));
            sl->set_cutoff_angle(glm::degrees(l.spot.outerConeAngle));
            sl->set_inner_angle(
                glm::degrees(l.spot.innerConeAngle),
                4/255.0f
            );
            sl->set_radius(meta.light_radius);
        }
        tnode = anode;
    }
    // Add light probe
    else if(tr_data && tr_data->Has("light_probe"))
    {
        tinygltf::Value light_probe = tr_data->Get("light_probe");
        std::string type = light_probe.Get("type").Get<std::string>();
        if(type == "GRID") // Irradiance volume
        {
            uvec3 res;
            res.x = light_probe.Get("resolution_x").GetNumberAsDouble();
            res.y = light_probe.Get("resolution_y").GetNumberAsDouble();
            res.z = light_probe.Get("resolution_z").GetNumberAsDouble();
            sh_grid g(res);

            g.set_radius(light_probe.Get("radius").GetNumberAsDouble());
            // Irradiance volume scale must not be negative, it will break
            // things.
            for(double& v: node.scale) v = fabs(v);

            data.sh_grids.emplace(node.name, std::move(g));
            tnode = &data.sh_grids.at(node.name);
        }
    }
    // If there is nothing, just add an empty node so that transformations work
    // correctly.
    else
    {
        anode = &add_unique_named(node.name, data.control_nodes);
        tnode = anode;
    }

    if(anode)
    {
        auto it = meta.animations.find(node_index);
        if(it != meta.animations.end())
        {
            static_lock = false;
            anode->set_animation_pool(it->second);
        }
    }

    tnode->set_parent(parent);

    // Set transformation for node
    if(node.matrix.size())
        tnode->set_transform(glm::make_mat4(node.matrix.data()));
    else
    {
        if(node.translation.size())
            tnode->set_position((vec3)glm::make_vec3(node.translation.data()));

        if(node.scale.size())
            tnode->set_scaling((vec3)glm::make_vec3(node.scale.data()));

        if(node.rotation.size())
            tnode->set_orientation(glm::make_quat(node.rotation.data()));
    }

    tnode->set_static(static_lock);

    // Save joints & root node
    for(int skin_index: meta.node_to_skin[node_index])
    {
        skin& s = meta.skins[skin_index];
        auto it = s.node_index_to_skin_index.find(node_index);
        if(it != s.node_index_to_skin_index.end())
        {
            s.joint_nodes[it->second] = anode;
        }
        if(s.root == node_index)
        {
            s.root_node = anode;
        }
    }

    // Load child nodes
    for(int child_index: node.children)
    {
        load_gltf_node(model, scene, child_index, data, tnode, meta, static_lock);
    }
}

}

namespace tr
{

scene_graph load_gltf(
    device_mask dev,
    const std::string& path,
    bool force_single_sided,
    bool force_double_sided
){
    TR_LOG("Started loading glTF scene from ", path);
    scene_graph md;

    std::string err, warn;
    tinygltf::Model gltf_model;
    tinygltf::TinyGLTF loader;

    // TinyGLTF uses stb_image too, and expects this value.
    stbi_set_flip_vertically_on_load(true);

    if(!loader.LoadBinaryFromFile(&gltf_model, &err, &warn, path))
        throw std::runtime_error(err);

    for(tinygltf::Image& image: gltf_model.images)
    {
        if(image.bufferView != -1)
        {// Embedded image
            vk::Format format;

            switch(image.component)
            {
            case 1:
                format = image.bits > 8 ? vk::Format::eR16Unorm : vk::Format::eR8Unorm;
                break;
            case 2:
                format = image.bits > 8 ?
                    vk::Format::eR16G16Unorm : vk::Format::eR8G8Unorm;
                break;
            default:
            case 3:
                format = image.bits > 8 ?
                    vk::Format::eR16G16B16Unorm : vk::Format::eR8G8B8Unorm;
                break;
            case 4:
                format = image.bits > 8 ?
                    vk::Format::eR16G16B16A16Unorm :
                    vk::Format::eR8G8B8A8Unorm;
                break;
            }

            flip_vector_image(image.image, image.height);

            md.textures.emplace_back(new texture(
                dev,
                uvec2(image.width, image.height),
                1,
                format,
                image.image.size(),
                image.image.data(),
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eSampled,
                vk::ImageLayout::eShaderReadOnlyOptimal
            ));

            if(check_opaque(image))
                md.textures.back()->set_opaque(true);
        }
        else
        {// URI
            md.textures.emplace_back(new texture(dev, image.uri));
        }
    }

    // Add animations
    node_meta_info meta;
    for(tinygltf::Animation& anim: gltf_model.animations)
    {
        for(tinygltf::AnimationChannel& chan: anim.channels)
        {
            auto it = meta.animations.find(chan.target_node);
            if(it == meta.animations.end())
                it = meta.animations.emplace(
                    chan.target_node,
                    md.animation_pools.emplace_back(new animation_pool()).get()
                ).first;

            animation& res = (*it->second)[anim.name];
            tinygltf::AnimationSampler& sampler = anim.samplers[chan.sampler];

            animation::interpolation interp = animation::LINEAR;
            if(sampler.interpolation == "LINEAR") interp = animation::LINEAR;
            else if(sampler.interpolation == "STEP") interp = animation::STEP;
            else if(sampler.interpolation == "CUBICSPLINE")
                interp = animation::CUBICSPLINE;

            if(chan.target_path == "translation")
                res.set_position(
                    interp,
                    read_animation_accessors<vec3>(
                        gltf_model, sampler.input, sampler.output
                    )
                );
            else if(chan.target_path == "rotation")
                res.set_orientation(
                    interp,
                    read_animation_accessors<quat>(
                        gltf_model, sampler.input, sampler.output
                    )
                );
            else if(chan.target_path == "scale")
                res.set_scaling(
                    interp,
                    read_animation_accessors<vec3>(
                        gltf_model, sampler.input, sampler.output
                    )
                );
            // Unknown target type (probably weights for morph targets)
            else continue;
        }
    }

    for(tinygltf::Skin& tg_skin: gltf_model.skins)
    {
        std::vector<int> joints = tg_skin.joints;
        std::vector<mat4> inverse_bind_matrices =
            read_accessor<mat4>(gltf_model, tg_skin.inverseBindMatrices);

        std::unordered_map<int, int> node_index_to_skin_index;
        for(size_t i = 0; i < tg_skin.joints.size(); i++)
        {
            node_index_to_skin_index[tg_skin.joints[i]] = i;
            meta.node_to_skin[tg_skin.joints[i]].push_back(meta.skins.size());
            meta.node_to_skin[tg_skin.skeleton].push_back(meta.skins.size());
        }

        skin s{
            tg_skin.skeleton,
            joints,
            inverse_bind_matrices,
            node_index_to_skin_index,
            std::vector<animated_node*>(joints.size()),
            {},
            {}
        };
        meta.skins.push_back(s);
    }

    for(tinygltf::Mesh& tg_mesh: gltf_model.meshes)
    {
        model m;

        for(tinygltf::Primitive& p: tg_mesh.primitives)
        {
            material primitive_material;
            if(p.material >= 0)
            {
                primitive_material = create_material(
                    gltf_model.materials[p.material], gltf_model, md
                );
                if(force_single_sided && primitive_material.transmittance == 0)
                    primitive_material.double_sided = false;
                if(force_double_sided)
                    primitive_material.double_sided = true;
            }

            std::vector<vec3> vert_pos;
            std::vector<vec3> vert_norm;
            std::vector<vec2> vert_uv;
            std::vector<vec4> vert_tangent;
            std::vector<uvec4> vert_joint;
            std::vector<vec4> vert_weight;

            for(const auto& pair: p.attributes)
            {
                if(pair.first == "POSITION")
                    vert_pos = read_accessor<vec3>(gltf_model, pair.second);
                else if(pair.first == "NORMAL")
                    vert_norm = read_accessor<vec3>(gltf_model, pair.second);
                else if(pair.first == "TEXCOORD_0")
                    vert_uv = read_accessor<vec2>(gltf_model, pair.second);
                else if(pair.first == "TANGENT")
                    vert_tangent = read_accessor<vec4>(gltf_model, pair.second);
                else if(pair.first == "JOINTS_0")
                    vert_joint = read_accessor<uvec4>(gltf_model, pair.second);
                else if(pair.first == "WEIGHTS_0")
                    vert_weight = read_accessor<vec4>(gltf_model, pair.second);
            }

            bool generate_tangents = false;
            if(vert_tangent.size() == 0 && primitive_material.normal_tex.first)
            {
                TR_WARN(
                    path, ": ", tg_mesh.name,
                    " uses a normal map but is missing tangent data. Please "
                    "export the asset with [Geometry > Tangents] ticked in "
                    "Blender."
                );
                generate_tangents = true;
            }

            bool generate_normals = vert_norm.size() == 0;

            mesh* prim_mesh = new mesh(dev);
            std::vector<mesh::vertex>& mesh_vert = prim_mesh->get_vertices();
            std::vector<mesh::skin_data>& mesh_skin = prim_mesh->get_skin();
            std::vector<uint32_t>& mesh_ind = prim_mesh->get_indices();

            mesh_vert.resize(vert_pos.size());
            vert_norm.resize(vert_pos.size(), vec3(0));
            vert_uv.resize(vert_pos.size(), vec2(0));
            vert_tangent.resize(vert_pos.size(), vec4(0));
            for(size_t i = 0; i < vert_pos.size(); ++i)
                mesh_vert[i] = {vert_pos[i], vert_norm[i], vert_uv[i], vert_tangent[i]};

            mesh_skin.resize(vert_joint.size());
            vert_weight.resize(vert_joint.size());
            for(size_t i = 0; i < vert_joint.size(); ++i)
            {
                // Some broken models have sums that go over 1 for some reason.
                // Anyway, that's incorrect so we fix it here.
                float weight_sum =
                    vert_weight[i].x + vert_weight[i].y + vert_weight[i].z +
                    vert_weight[i].w;
                mesh_skin[i] = {vert_joint[i], vert_weight[i]/weight_sum};
            }

            mesh_ind = read_accessor<uint32_t>(gltf_model, p.indices);
            if(mesh_ind.size() == 0)
            { // Missing indices, so we make them. Stupid model.
                mesh_ind.resize(vert_pos.size());
                std::iota(mesh_ind.begin(), mesh_ind.end(), 0);
            }

            if(generate_normals)
                prim_mesh->calculate_normals();
            if(generate_tangents)
                prim_mesh->calculate_tangents();

            md.meshes.emplace_back(prim_mesh);
            m.add_vertex_group(primitive_material, prim_mesh);
        }

        add_unique_named(tg_mesh.name, md.models) = std::move(m);
    }

    // Add objects & cameras
    for(tinygltf::Scene& scene: gltf_model.scenes)
    {
        for(int node_index: scene.nodes)
            load_gltf_node(gltf_model, scene, node_index, md, nullptr, meta, true);
    }

    // Apply skins to meshes
    for(skin& s: meta.skins)
    {
        for(model* m: s.related_models)
        {
            std::vector<model::joint_data> joints = s.build_joint_data();
            m->get_joints() = std::move(joints);
        }
    }

    // Upload buffer data here so that we have had time to fill in joint data
    for(auto& m: md.meshes)
        m->refresh_buffers();

    // Detach animated mesh clones
    for(auto& pair: md.mesh_objects)
    {
        if(pair.second.get_model()->get_joints().size() == 0)
            continue;

        // glTF explicitly specifies that skinned meshes must be placed at the
        // origin of the scene. This doesn't always seem to be the case in all
        // models, so fix it here.
        pair.second.set_transform(mat4(1));
        pair.second.set_parent(nullptr);
        pair.second.set_animation_pool(nullptr);

        model* animation_model = new model(*pair.second.get_model());
        animation_model->init_joints_buffer(dev);
        for(auto& vg: *animation_model)
        {
            mesh* animation_mesh = new mesh(vg.m);
            vg.m = animation_mesh;
            md.meshes.emplace_back(animation_mesh);
        }
        pair.second.set_model(animation_model);
        md.animation_models.emplace_back(animation_model);
    }

    TR_LOG("Finished loading glTF scene", path);
    return md;
}

}
