module;


#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <vector>
#include <filesystem>
#include <print>

module models;

import fastgltf;

void Meshes::add_meshes(const std::filesystem::path &filepath){
    fastgltf::Parser parser;
    fastgltf::GltfFileStream file_stream(filepath);
    if (!file_stream.isOpen()) {
        std::println("fastgltf couldn't open file {}", filepath.string());
        abort();
    }

    auto asset = parser.loadGltf(file_stream, filepath.parent_path());
    if (asset.error() != fastgltf::Error::None) {
        std::println("fastgltf error while parsing file {} with code {}", filepath.string(), fastgltf::to_underlying(asset.error()));
        assert(false);
        abort();
    }

    if (auto error = fastgltf::validate(asset.get()); error != fastgltf::Error::None){
        std::println("fastgltf error while validating asset {}, error code {}", filepath.string(), fastgltf::to_underlying(error));
        assert(false);
        abort();
    }

    for (const auto &gltf_mesh : asset->meshes){
        Mesh new_mesh;
        new_mesh.first_submesh_idx = submeshes.size();
        new_mesh.name = gltf_mesh.name;
        uint32_t primitive_count = 0;
        if (gltf_mesh.primitives.empty()){
            std::println("Mesh {} has no primitives. This is invalid.", gltf_mesh.name);
            abort();
        }
        for (const auto &primitive : gltf_mesh.primitives){
            Submesh new_submesh;
            new_submesh.name = gltf_mesh.name;
            new_submesh.name.append("_primitive").append(std::to_string(primitive_count++));
            new_submesh.first_index = indices.size();
            new_submesh.vertex_buffer_byte_offset = (sizeof(Vertex) * vertices.size());

            const fastgltf::Accessor &index_accessor = asset->accessors[primitive.indicesAccessor.value()];
            new_submesh.index_count = index_accessor.count;
            fastgltf::iterateAccessor<std::uint32_t>(asset.get(), index_accessor, [&](std::uint32_t index){
                indices.push_back(index);
            });

            const auto has_attribute = [&](const std::string &attribute_name){
                bool has_attribute = primitive.findAttribute(attribute_name) != primitive.attributes.end();
                if (!has_attribute){
                    std::println("Warning: file {} has mesh {} that has no attribute {}.", filepath.string(), gltf_mesh.name, attribute_name);
                }
                return has_attribute;
            };

            const auto get_accessor_for_attribute = [&](const std::string &attribute_name, const fastgltf::Asset& asset)->const fastgltf::Accessor&{
                assert(has_attribute(attribute_name) && "You should've checked that the attribute existed");
                size_t accessor_idx = primitive.findAttribute(attribute_name)->accessorIndex;
                return asset.accessors[accessor_idx];
            };

            uint32_t first_vertex_of_current_mesh_idx = vertices.size();

            if (!has_attribute("POSITION")) {abort();};
            const fastgltf::Accessor &pos_accessor = get_accessor_for_attribute("POSITION", asset.get());
            new_submesh.vertex_count = pos_accessor.count;
            fastgltf::iterateAccessor<glm::vec3>(asset.get(), pos_accessor, [&](glm::vec3 position){
                vertices.push_back(Vertex{.pos = position});
            });

            if (has_attribute("NORMAL")){
                const fastgltf::Accessor &norm_accessor = get_accessor_for_attribute("NORMAL", asset.get());
                fastgltf::iterateAccessorWithIndex<glm::vec3>(asset.get(), norm_accessor, [&](glm::vec3 norm, uint32_t idx){
                    vertices[first_vertex_of_current_mesh_idx + idx].normal = norm;
                });
            }

            if (has_attribute("TEXCOORD_0")){
                const fastgltf::Accessor &tex_accessor = get_accessor_for_attribute("TEXCOORD_0", asset.get());
                fastgltf::iterateAccessorWithIndex<glm::vec2>(asset.get(), tex_accessor, [&](glm::vec2 tex, uint32_t idx){
                    vertices[first_vertex_of_current_mesh_idx + idx].u = tex.x;
                    vertices[first_vertex_of_current_mesh_idx + idx].v = tex.y;
                });
            }

            if (has_attribute("COLOR_0")){
                const fastgltf::Accessor &color_accessor = get_accessor_for_attribute("COLOR_0", asset.get());
                if (color_accessor.type == fastgltf::AccessorType::Vec3){
                    fastgltf::iterateAccessorWithIndex<glm::vec3>(asset.get(), color_accessor, [&](glm::vec3 col, uint32_t idx){
                        vertices[first_vertex_of_current_mesh_idx + idx].color = glm::vec4(col, 1);
                    });
                }

                if (color_accessor.type == fastgltf::AccessorType::Vec4){
                    fastgltf::iterateAccessorWithIndex<glm::vec4>(asset.get(), color_accessor, [&](glm::vec4 col, uint32_t idx){
                        vertices[first_vertex_of_current_mesh_idx + idx].color = col;
                    });
                }

            }
            submeshes.push_back(new_submesh);
        }
        new_mesh.last_submesh_idx = submeshes.size() - 1;
        meshes.push_back(new_mesh);
    }
}


