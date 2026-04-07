module;

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <vector>
#include <filesystem>

export module models;
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



export struct Vertex {
    glm::vec3 pos;
    float u;
    glm::vec3 normal;
    float v;
    glm::vec4 color;
};


export struct Submesh{
    std::string name;
    uint32_t vertex_buffer_byte_offset;
    uint32_t vertex_count;
    uint32_t first_index;
    uint32_t index_count;

    [[nodiscard]] uint32_t get_index_of_first_vertex() const {
        return vertex_buffer_byte_offset / sizeof(Vertex);
    }
    [[nodiscard]] uint32_t get_index_of_last_vertex() const {
        return get_index_of_first_vertex() + vertex_count - 1;
    }
};

export struct Mesh{
    std::string name;
    uint32_t first_submesh_idx;
    uint32_t last_submesh_idx;
};

export struct Meshes{
    std::vector<Mesh> meshes;
    std::vector<Submesh> submeshes;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    [[nodiscard]] size_t vertex_buffer_size_in_bytes() const {
        return vertices.size() * sizeof(decltype(vertices)::value_type);
    }
    [[nodiscard]] size_t index_buffer_size_in_bytes() const {
        return indices.size() * sizeof(decltype(indices)::value_type);
    }
    Meshes() = default;
    Meshes(const std::filesystem::path &filepath){
        add_meshes(filepath);
    }
    Meshes(const std::initializer_list<std::filesystem::path> &filepaths){
        for (const auto &path : filepaths) add_meshes(path);
    }
    Meshes(const std::span<std::filesystem::path> &filepaths){
        for (const auto &path : filepaths) add_meshes(path);
    }

    void add_meshes(const std::filesystem::path &filepath);
};
