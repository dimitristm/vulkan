module;

#include <glm/glm.hpp>
#include <vector>
#include <filesystem>
#include <stdexcept>
#include <span>
#include <print>

export module meshLoading;
import fastgltf;

// Sadly, fastgltf doesn't expose these in the module so we need to keep them here.
namespace fastgltf {
template<> struct ElementTraits<glm::vec2> : ElementTraitsBase<glm::vec2, AccessorType::Vec2, float> {};
template<> struct ElementTraits<glm::vec3> : ElementTraitsBase<glm::vec3, AccessorType::Vec3, float> {};
template<> struct ElementTraits<glm::vec4> : ElementTraitsBase<glm::vec4, AccessorType::Vec4, float> {};
template<> struct ElementTraits<glm::i8vec2> : ElementTraitsBase<glm::i8vec2, AccessorType::Vec2, std::int8_t> {};
template<> struct ElementTraits<glm::i8vec3> : ElementTraitsBase<glm::i8vec3, AccessorType::Vec3, std::int8_t> {};
template<> struct ElementTraits<glm::i8vec4> : ElementTraitsBase<glm::i8vec4, AccessorType::Vec4, std::int8_t> {};
template<> struct ElementTraits<glm::u8vec2> : ElementTraitsBase<glm::u8vec2, AccessorType::Vec2, std::uint8_t> {};
template<> struct ElementTraits<glm::u8vec3> : ElementTraitsBase<glm::u8vec3, AccessorType::Vec3, std::uint8_t> {};
template<> struct ElementTraits<glm::u8vec4> : ElementTraitsBase<glm::u8vec4, AccessorType::Vec4, std::uint8_t> {};
template<> struct ElementTraits<glm::i16vec2> : ElementTraitsBase<glm::i16vec2, AccessorType::Vec2, std::int16_t> {};
template<> struct ElementTraits<glm::i16vec3> : ElementTraitsBase<glm::i16vec3, AccessorType::Vec3, std::int16_t> {};
template<> struct ElementTraits<glm::i16vec4> : ElementTraitsBase<glm::i16vec4, AccessorType::Vec4, std::int16_t> {};
template<> struct ElementTraits<glm::u16vec2> : ElementTraitsBase<glm::u16vec2, AccessorType::Vec2, std::uint16_t> {};
template<> struct ElementTraits<glm::u16vec3> : ElementTraitsBase<glm::u16vec3, AccessorType::Vec3, std::uint16_t> {};
template<> struct ElementTraits<glm::u16vec4> : ElementTraitsBase<glm::u16vec4, AccessorType::Vec4, std::uint16_t> {};
template<> struct ElementTraits<glm::u32vec2> : ElementTraitsBase<glm::u32vec2, AccessorType::Vec2, std::uint32_t> {};
template<> struct ElementTraits<glm::u32vec3> : ElementTraitsBase<glm::u32vec3, AccessorType::Vec3, std::uint32_t> {};
template<> struct ElementTraits<glm::u32vec4> : ElementTraitsBase<glm::u32vec4, AccessorType::Vec4, std::uint32_t> {};
template<> struct ElementTraits<glm::mat2> : ElementTraitsBase<glm::mat2, AccessorType::Mat2, float> {};
template<> struct ElementTraits<glm::mat3> : ElementTraitsBase<glm::mat3, AccessorType::Mat3, float> {};
template<> struct ElementTraits<glm::mat4> : ElementTraitsBase<glm::mat4, AccessorType::Mat4, float> {};
} // namespace fastgltf



namespace{
struct Vertex {
    glm::vec3 pos;
    float u;
    glm::vec3 normal;
    float v;
    glm::vec4 color;
};
}

struct MeshData{
    std::string name;
    uint32_t vertex_buffer_byte_offset;
    uint32_t first_index;
};

struct Meshes{
    std::vector<MeshData> mesh_data;
    std::vector<Vertex>vertices;
    std::vector<uint32_t> indices;

    Meshes() = default;
    Meshes(const std::filesystem::path &filepath){
        add_meshes(filepath);
    }

    void add_meshes(const std::filesystem::path &filepath){
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
            uint32_t primitive_count = 0;
            for (const auto &primitive : gltf_mesh.primitives){
                MeshData new_mesh_data;
                new_mesh_data.name = gltf_mesh.name;
                new_mesh_data.name.append("_primitive").append(std::to_string(primitive_count));
                new_mesh_data.first_index = indices.size();
                new_mesh_data.vertex_buffer_byte_offset = (sizeof(Vertex) * vertices.size());
                mesh_data.push_back(new_mesh_data);

                const fastgltf::Accessor &index_accessor = asset->accessors[primitive.indicesAccessor.value()];
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
                    fastgltf::iterateAccessorWithIndex<glm::vec4>(asset.get(), color_accessor, [&](glm::vec4 col, uint32_t idx){
                        vertices[first_vertex_of_current_mesh_idx + idx].color = col;
                    });
                }
            }
        }
    }
};
