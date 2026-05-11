module;

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <vulkan/vulkan_core.h>
#include <xxhash/xxhash.h>
#if !USE_IMPORT_STD
#include <vector>
#include <filesystem>
#include <unordered_map>
#endif

export module gltf;
#if USE_IMPORT_STD
import std;
#endif
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

export struct GltfSampler{
    VkFilter mag_filter;
    VkFilter min_filter;
    VkSamplerMipmapMode mipmap_mode;
    VkSamplerAddressMode wrap_s;
    VkSamplerAddressMode wrap_t;
};

export struct GltfImage {
    std::unique_ptr<std::byte[]> image;
    uint32_t size;
};

export struct GltfTexture {
    uint32_t image_idx;
    uint32_t sampler_idx;
};

export struct GltfMaterial {
    // texture indices into GltfScenes::textures (UINT32_MAX = not present)
    uint32_t base_color_tex_idx = UINT32_MAX;
    uint32_t normal_tex_idx = UINT32_MAX;
    uint32_t metallic_roughness_tex_idx = UINT32_MAX;

    glm::vec4 base_color_factor = {1.f, 1.f, 1.f, 1.f};
    float metallic_factor = 1.f;
    float roughness_factor = 1.f;
};

export struct GltfScene{
    std::vector<uint32_t> mesh_idx;
};

export struct GltfMesh{
    glm::mat4 transform;
    std::vector<uint32_t> mesh_prim_idx;
    std::string name; // Not in the hash
};

export struct GltfMeshPrimitive{
    uint32_t vertices_idx;
    uint32_t indices_idx;
    uint32_t material_idx;
};

export struct GltfScenes{
    using HashToGlobalIdx = std::unordered_map<XXH64_hash_t, uint32_t>;
    struct DeduplicationMaps{
        // The key of everything that starts with gltf_ is the hash of the actual struct, for instance
        // gltf_mesh contains a hash of a GltfMesh struct.
        // For things that don't start with gltf, it's a hash of the actual data, for instance images
        // has a hash of the full image data in GltfImage::image, vertices a hash of all the vertices
        // in a vector, etc.
        // The value is an index into the corresponding vector below.
        HashToGlobalIdx gltf_meshes;
        HashToGlobalIdx gltf_mesh_primitives;
        HashToGlobalIdx vertices;
        HashToGlobalIdx indices;
        HashToGlobalIdx gltf_materials;
        HashToGlobalIdx gltf_images;
        HashToGlobalIdx gltf_textures;
        HashToGlobalIdx gltf_samplers;
    };

    DeduplicationMaps deduplication_maps;

    std::vector<GltfScene> scene_infos;
    std::vector<GltfMeshPrimitive> mesh_primitives;
    std::vector<GltfTexture> textures;
    std::vector<GltfSampler> samplers;
    std::vector<GltfMaterial> materials;
    std::vector<std::vector<Vertex>> vertices;
    std::vector<std::vector<uint32_t>> indices;
    std::vector<GltfMesh> meshes;
    std::vector<GltfImage> gltf_images;

    GltfScenes() = default;
    GltfScenes(const std::filesystem::path &filepath){
        add_scenes(filepath);
    }
    GltfScenes(const std::initializer_list<std::filesystem::path> &filepaths){
        for (const auto &path : filepaths) add_scenes(path);
    }
    GltfScenes(const std::span<std::filesystem::path> &filepaths){
        for (const auto &path : filepaths) add_scenes(path);
    }

    void add_scenes(const std::filesystem::path &filepath);

    size_t get_vertex_count(){
        size_t count = 0;
        for (const auto &subvertices : vertices){
            count += subvertices.size();
        }
        return count;
    }

    size_t get_index_count(){
        size_t count = 0;
        for (const auto &subindices : indices){
            count += subindices.size();
        }
        return count;
    }
};
