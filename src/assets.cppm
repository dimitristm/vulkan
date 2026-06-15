module;

#include "ktx.h"
#include <stb/stb_image.h>
#include <vulkan/vulkan_core.h>
#include <glm/glm.hpp>
#include <xxhash/xxhash.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <boost/pfr.hpp>

#if !USE_IMPORT_STD
#include <unordered_map>
#include <print>
#include <filesystem>
#include <vector>
#include <cstring>
#include <variant>
#include <concepts>
#endif

export module assets;
#if USE_IMPORT_STD
import std;
#endif

import fastgltf;
import vulkanEngine;
import vulkanUtil;
import types;
import util;

using std::memcpy;
using util::ByteRange;

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

// Sums the sizeof every element in a tuple
template <typename Tuple, std::size_t... Indexes>
static constexpr std::size_t sum_tuple_sizes(std::index_sequence<Indexes...> /*unused*/) {
    return (sizeof(std::tuple_element_t<Indexes, Tuple>) + ... + 0);
}

// True if T has no added padding beyond the padding of its constituent fields.
// That means we can hash it without worrying about padding having random bytes in it,
// except of course if the constituent fields have padding themselves.
template <typename T>
static constexpr bool no_added_padding() {
    using TupleOfAllFieldsType = decltype(boost::pfr::structure_to_tuple(std::declval<T>()));
    constexpr std::size_t field_count = std::tuple_size_v<TupleOfAllFieldsType>;
    constexpr std::size_t total_size_of_fields = sum_tuple_sizes<TupleOfAllFieldsType>(std::make_index_sequence<field_count>{});

    return sizeof(T) == total_size_of_fields;
}

export struct Vertex {
    fvec3 pos;
    f32   u;
    fvec3 normal;
    f32   v;
    fvec4 color;
};
static_assert(no_added_padding<Vertex>());

template<typename T>
concept Hashable = requires(const T& item) {
    { item.hash() } -> std::same_as<XXH128_hash_t>;
};

static void KTX_CHECK(ktx_error_code_e result){
    if (result != KTX_SUCCESS){
        std::println("KTX function failed");
        abort();
    }
}

struct XXH128Hasher {
    std::size_t operator()(const XXH128_hash_t& hash) const noexcept {
        return XXH3_64bits(&hash, sizeof(hash));
    }
};

struct XXH128Equal {
    bool operator()(const XXH128_hash_t& a, const XXH128_hash_t& b) const noexcept {
        return static_cast<bool>(XXH128_isEqual(a, b));
    }
};

template<typename T>
using XXH128Map = std::unordered_map<XXH128_hash_t, T, XXH128Hasher, XXH128Equal>;

static constexpr u64 assetpack_segment_signifier = 0x6295672205783692;
struct AssetpackSampler {
    VkFilter mag_filter;
    VkFilter min_filter;
    VkSamplerMipmapMode mipmap_mode;
    VkSamplerAddressMode wrap_s;
    VkSamplerAddressMode wrap_t;

    AssetpackSampler()
    :mag_filter(VK_FILTER_LINEAR),
    min_filter(VK_FILTER_LINEAR),
    mipmap_mode(VK_SAMPLER_MIPMAP_MODE_LINEAR),
    wrap_s(VK_SAMPLER_ADDRESS_MODE_REPEAT),
    wrap_t(VK_SAMPLER_ADDRESS_MODE_REPEAT)
    {}

    explicit AssetpackSampler(const fastgltf::Sampler& s)
    :mag_filter(s.magFilter ? to_vk_filter(*s.magFilter) : VK_FILTER_LINEAR),
    min_filter(s.minFilter ? to_vk_filter(*s.minFilter) : VK_FILTER_LINEAR),
    mipmap_mode(s.minFilter ? to_vk_mipmap(*s.minFilter) : VK_SAMPLER_MIPMAP_MODE_LINEAR),
    wrap_s(to_vk_wrap(s.wrapS)),
    wrap_t(to_vk_wrap(s.wrapT))
    {}

    [[nodiscard]] XXH128_hash_t hash() const{
        return XXH3_128bits(this, sizeof(AssetpackSampler));
    }

private:
    static VkFilter to_vk_filter(fastgltf::Filter filter) {
        switch (filter) {
            case fastgltf::Filter::Nearest:
            case fastgltf::Filter::NearestMipMapNearest:
            case fastgltf::Filter::NearestMipMapLinear:
                return VK_FILTER_NEAREST;
            default:
                return VK_FILTER_LINEAR;
        }
    }
    static VkSamplerMipmapMode to_vk_mipmap(fastgltf::Filter filter) {
        switch (filter) {
            case fastgltf::Filter::NearestMipMapNearest:
            case fastgltf::Filter::LinearMipMapNearest:
                return VK_SAMPLER_MIPMAP_MODE_NEAREST;
            default:
                return VK_SAMPLER_MIPMAP_MODE_LINEAR;
        }
    }
    static VkSamplerAddressMode to_vk_wrap(fastgltf::Wrap wrap) {
        switch (wrap) {
            case fastgltf::Wrap::ClampToEdge:    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            case fastgltf::Wrap::MirroredRepeat: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            default:                             return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        }
    }
};
static_assert(no_added_padding<AssetpackSampler>());

struct AssetpackShardHeader{
    u64 offset_to_scenes;
    u64 offset_to_meshes;
    u64 offset_to_mesh_primitives;
    u64 offset_to_samplers;
    u64 offset_to_image_metadata;
    u64 offset_to_indices;
    u64 offset_to_vertices;
    u64 offset_to_images;
    u64 offset_to_after_end_of_shard;
    u64 offset_to_next_shard_or_0 = 0;
};
struct AssetpackHeader{
    static constexpr u32 max_name_length = 256;
    static constexpr u64 valid_file_signature = 0xBF269A65A2E91226;
    static constexpr u64 error_did_not_finish_building_file_signature = 0x85AC8E6951FC737D;
    const u64 file_signature{};
    const u64 version = 0;
    u32 name_length;
    char name[max_name_length];
    static constexpr u64 serialized_size = sizeof(file_signature) + sizeof(version)
        + sizeof(name_length) + max_name_length;
};
// just in case we later decide to hash entire assetpack files
static_assert(no_added_padding<AssetpackShardHeader>());


export class AssetpackBuilder{
    enum class ImageType : u32{
        Albedo = 0,
        Normal = 1,
        MetallicRoughness = 2,
    };

    struct AssetpackScene{
        std::vector<u32> mesh_idx;
        std::vector<fmat4> transforms;
    };

    struct AssetpackMesh{
        std::vector<u32> mesh_prim_idx;

        [[nodiscard]] XXH128_hash_t hash() const{
            return XXH3_128bits(mesh_prim_idx.data(), mesh_prim_idx.size() * sizeof(u32));
        }
    };

    // all indexes are global and shared between shards.
    struct AssetpackMeshPrimitive{
        // When the file is built, these byte are offsets into the actual full file, but while loading
        // from a gltf/preexisting assetpack, since we don't yet know where the full file offset will end up
        // as we haven't parsed the full thing, these are used as offsets into a conceptual
        // array of all vertices that will exist in the assetpack (all shards combined).
        // Also, the set of all *unique* ranges found in primitives must not contain
        // ranges that overlap (multiple primitives can have the same exact ranges, though).
        // This is for tracking what vertices/indices are loaded for dynamic
        // loading/unloading to gpu memory later on.
        u64 vertex_offset;
        u64 first_index_offset;

        u32 index_count;
        u32 sampler_idx;
        u32 albedo_idx;
        u32 normal_map_idx;
        u32 metallic_roughness_idx;
        fvec4 base_color_factor { 1.f, 1.f, 1.f, 1.f };
        f32 metallic_factor = 1.f;
        f32 roughness_factor = 1.f;
        u32 padding = 0;

        [[nodiscard]] XXH128_hash_t hash() const{
            return XXH3_128bits(this, sizeof(*this));
        }
    };
    static_assert(no_added_padding<AssetpackMeshPrimitive>());

    struct AssetpackImage{
        u32 width;
        u32 height;
        VkFormat format;
        // The offset in this range is similar to the offsets in AssetpackMeshPrimitive.
        std::vector<ByteRange> mips;
        std::vector<u64> local_data_mip_offset;
        static_assert(no_added_padding<ByteRange>());
        // in the actual assetpack data is stored seperately from the rest of the info,
        // put all image datas at the end
        std::vector<std::byte> data;

        AssetpackImage() = default;
        AssetpackImage(const uint8_t *pixels, u32 width, u32 height, ImageType type, u64 last_offset)
        :width(width), height(height),
         format((type == ImageType::Albedo) ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC5_UNORM_BLOCK)
        {
            std::vector<uint8_t> swizzled_storage;
            const uint8_t *src = pixels;
            if (type == ImageType::MetallicRoughness){
                swizzled_storage.resize(width * height * 4);
                for (u32 i = 0; i < width * height; ++i){
                    swizzled_storage[i*4 + 0] = pixels[i*4 + 1]; // R = G (roughness)
                    swizzled_storage[i*4 + 1] = pixels[i*4 + 2]; // G = B (metallic)
                    swizzled_storage[i*4 + 2] = 0;
                    swizzled_storage[i*4 + 3] = 255;
                }
                src = swizzled_storage.data();
            }

            util::Timer t2;
            t2.start();
            ktxTextureCreateInfo texture_create_info{
                .vkFormat = (type == ImageType::Albedo) ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM,
                .baseWidth = width,
                .baseHeight = height,
                .baseDepth = 1,
                .numDimensions = 2,
                .numLevels = 1,
                .numLayers = 1,
                .numFaces = 1,
                .isArray = KTX_FALSE,
                .generateMipmaps = KTX_FALSE, //todo: mipmaps
            };
            ktxTexture2 *tex = nullptr;
            KTX_CHECK(ktxTexture2_Create(&texture_create_info, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &tex));
            std::memcpy(tex->pData, src, width * height * 4);
            ktxBasisParams params{};
            params.structSize = sizeof(params);
            params.uastc      = KTX_TRUE;
            params.uastcFlags = KTX_PACK_UASTC_LEVEL_DEFAULT; //todo: use a higher level
            params.verbose    = KTX_FALSE;
            params.threadCount = 16;
            KTX_CHECK(ktxTexture2_CompressBasisEx(tex, &params));
            ktx_transcode_fmt_e tfmt = (type == ImageType::Albedo) ? KTX_TTF_BC7_RGBA : KTX_TTF_BC5_RG;
            KTX_CHECK(ktxTexture2_TranscodeBasis(tex, tfmt, 0));
            t2.end();
            //std::println("transcode took: {}", t2.elapsed());

            for (u32 mip_level = 0; mip_level < tex->numLevels; ++mip_level){
                ktx_size_t mip_offset = 0;
                ktxTexture_GetImageOffset(ktxTexture(tex), mip_level, 0, 0, &mip_offset);
                local_data_mip_offset.push_back(mip_offset);
                ktx_size_t mip_size = ktxTexture_GetImageSize(ktxTexture(tex), mip_level);
                this->mips.push_back({.offset = last_offset, .size = mip_size});
                last_offset += mip_size; //update offset into conceptual data buffer
                u64 data_size_before_this_append = this->data.size();
                this->data.resize(data.size() + mip_size);
                std::memcpy(this->data.data() + data_size_before_this_append, tex->pData + mip_offset, mip_size);
            }
            ktxTexture_Destroy(ktxTexture(tex));
        }

        [[nodiscard]] XXH128_hash_t hash() const{
            return XXH3_128bits(data.data(), data.size());
        }
    };

    // When we translate from conceptual offsets to file offsets, we need to know which ranges of
    // conceptual offsets correspond to which shard, and also the offset into that shard's vertice etc.
    // By storing one of these for each created shard, we can figure out which shard a conceptual
    // offset lies in, and then add that shard's first file offset to it. This gives the full file offset.
    struct ShardFirstOffsets{
        u64 after_end_index_conceptual_offset;
        u64 after_end_vertex_conceptual_offset;
        u64 after_end_image_conceptual_offset;
        u64 index_file_offset;
        u64 vertex_file_offset;
        u64 image_file_offset;
    };

    AssetpackHeader header;
    std::vector<ShardFirstOffsets> conceptual_offsets;
    std::ostream &out;
    std::vector<AssetpackScene> scenes;
    std::vector<AssetpackMesh> meshes;
    std::vector<AssetpackMeshPrimitive> mesh_primitives;
    std::vector<u32> indices;
    std::vector<Vertex> vertices;
    u64 curr_indices_conceptual_offset = 0;
    u64 curr_vertices_conceptual_offset = 0;
    std::vector<AssetpackSampler> samplers;
    std::vector<AssetpackImage> images;
    u64 curr_images_conceptual_offset = 0;

    using HashToIdx = XXH128Map<u64>;
    struct DeduplicationMaps{
        // AssetpackX.hash() -> index in the assetpack
        HashToIdx meshes;
        HashToIdx mesh_primitives;
        HashToIdx indices;
        HashToIdx vertices;
        HashToIdx samplers;
        HashToIdx images;
    };
    DeduplicationMaps dedup;

    static const u32 default_sampler_idx = 0;
    static const u32 default_albedo_idx = 0;
    static const u32 default_normal_map_idx = 1;
    static const u32 default_metallic_roughness_idx = 2;

    template<typename T>
    void write(const T& v){ out.write(reinterpret_cast<const char*>(&v), sizeof(v)); }

public:
    AssetpackBuilder(std::string_view name, std::ostream &out)
    :header({
        .file_signature = AssetpackHeader::error_did_not_finish_building_file_signature,
        .version = 0,
        .name_length = static_cast<u32>(name.size()),
    }),
    out(out)
    {
        if (out.tellp() != 0) std::println("Warning: passed ostream to AssetpackBuilder that wasn't pointing to the start of file. The builder will start writing at the start of the file regardless.");
        out.seekp(0, std::ios::beg);
        for (int i = 0; const auto &c : name){
            header.name[i++] = c;
        }
        write(header.file_signature);
        write(header.version);
        write(header.name_length);
        out.write(header.name, AssetpackHeader::max_name_length);
        {
            AssetpackSampler default_sampler{};
            dedup.samplers[default_sampler.hash()] = samplers.size();
            samplers.push_back(default_sampler);
        }

        const uint8_t white[4]     = { 255, 255, 255, 255 };
        const uint8_t flat_norm[4] = { 128, 128, 255, 255 };

        const auto insert_default = [&](const uint8_t* px, ImageType type) {
            AssetpackImage image(px, 1, 1, type, curr_images_conceptual_offset);
            curr_images_conceptual_offset += image.data.size();
            dedup.images[image.hash()] = images.size();
            images.push_back(std::move(image));
        };

        insert_default(white, ImageType::Albedo);
        insert_default(flat_norm, ImageType::Normal);
        insert_default(white, ImageType::MetallicRoughness);
    }

    AssetpackBuilder &add_from_gltf(const std::filesystem::path &path){
        struct GltfPreviouslyLoadedImage{
            u32 image_idx;
            ImageType type;
            XXH128_hash_t hash(){
                return XXH3_128bits(this, sizeof(GltfPreviouslyLoadedImage));
            }
        };
        static_assert(no_added_padding<GltfPreviouslyLoadedImage>());
        HashToIdx loaded_gltf_images;

        fastgltf::Parser parser;
        fastgltf::GltfFileStream file_stream(path);
        if (!file_stream.isOpen()) {
            std::println("Cannot open {}", path.string()); abort();
        }

        const auto options = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::DecomposeNodeMatrices;

        auto expected_asset = parser.loadGltf(file_stream, path.parent_path(), options);
        if (expected_asset.error() != fastgltf::Error::None) {
            std::println("fastgltf error: {}", fastgltf::getErrorMessage(expected_asset.error()));
            abort();
        }
        fastgltf::Asset& asset = expected_asset.get();
        if (auto err = fastgltf::validate(asset); err != fastgltf::Error::None) {
            std::println("fastgltf validation error: {}", fastgltf::getErrorMessage(err));
            abort();
        }

        const auto asset_dir = path.parent_path();

        //returns the index of the element and true if it was a duplicate
        static constexpr auto deduplicate_and_add =
        []<Hashable T>(HashToIdx &dedup_map, T &item, std::vector<T> &storage)->std::tuple<u32,bool>
        {
            XXH128_hash_t hash = item.hash();
            if (auto it = dedup_map.find(hash); it != dedup_map.end()){
                return {it->second, true};
            }
            u32 idx = static_cast<u32>(dedup_map.size());
            storage.push_back(item);
            dedup_map[hash] = idx;
            return {idx, false};
        };

        const auto get_or_add_sampler = [&](u32 gltf_sampler_idx) -> u32{
            AssetpackSampler assetpack_sampler(asset.samplers[gltf_sampler_idx]);
            return std::get<0>(deduplicate_and_add(dedup.samplers, assetpack_sampler, samplers));
        };

        const auto get_or_add_image = [&](u32 gltf_img_idx, ImageType type) -> u32{
            GltfPreviouslyLoadedImage prev_loaded_img{
                .image_idx = gltf_img_idx,
                .type = type,
            };
            XXH128_hash_t gltf_hash = prev_loaded_img.hash();
            if (auto it = loaded_gltf_images.find(gltf_hash); it != loaded_gltf_images.end()){
                return it->second;
            }

            const auto get_raw_image_bytes = [&]() -> std::vector<std::byte>{
                const auto& fastgltf_img = asset.images[gltf_img_idx];
                std::vector<std::byte> raw_bytes;
                std::visit(fastgltf::visitor{
                    [&](const fastgltf::sources::Array& arr) {
                        raw_bytes.resize(arr.bytes.size());
                        std::memcpy(raw_bytes.data(), arr.bytes.data(), arr.bytes.size());
                    },
                    [&](const fastgltf::sources::BufferView& bv_src) {
                        const auto& bv  = asset.bufferViews[bv_src.bufferViewIndex];
                        const auto& buf = asset.buffers[bv.bufferIndex];
                        std::visit(fastgltf::visitor{
                            [&](const fastgltf::sources::Array& arr) {
                                raw_bytes.resize(bv.byteLength);
                                std::memcpy(raw_bytes.data(),
                                    arr.bytes.data() + bv.byteOffset, bv.byteLength);
                            },
                            [](auto&&) { std::println("Unsupported buffer source"); abort(); }
                        }, buf.data);
                    },
                    [&](const fastgltf::sources::URI& uri_src) {
                        auto img_path = asset_dir / uri_src.uri.fspath();
                        std::ifstream file(img_path, std::ios::binary | std::ios::ate);
                        if (!file) { std::println("Cannot open image {}", img_path.string()); abort(); }
                        auto sz = static_cast<std::size_t>(file.tellg());
                        file.seekg(0);
                        raw_bytes.resize(sz);
                        file.read(reinterpret_cast<char*>(raw_bytes.data()), sz);
                    },
                    [](auto&&) { std::println("Unsupported image source"); abort(); }
                }, fastgltf_img.data);
                return raw_bytes;
            };

            std::vector<std::byte> raw_bytes = get_raw_image_bytes();

            int width, height, channels;
            stbi_uc* pixels = stbi_load_from_memory(
                reinterpret_cast<const stbi_uc*>(raw_bytes.data()),
                static_cast<int>(raw_bytes.size()),
                &width, &height, &channels, 4);
            if (!pixels) { std::println("stbi failed to decode image"); abort(); }

            AssetpackImage assetpack_img(pixels, static_cast<u32>(width), static_cast<u32>(height), type, curr_images_conceptual_offset);
            stbi_image_free(pixels);

            auto [idx, was_duplicate] = deduplicate_and_add(dedup.images, assetpack_img, images);
            if (!was_duplicate) curr_images_conceptual_offset += assetpack_img.data.size();
            loaded_gltf_images[gltf_hash] = idx;
            return idx;
        };

        struct MaterialData {
            u32 albedo_idx = default_albedo_idx;
            u32 normal_map_idx = default_normal_map_idx;
            u32 metallic_roughness_idx = default_metallic_roughness_idx;
            u32 sampler_idx = default_sampler_idx;
            fvec4 base_color_factor = { 1.f, 1.f, 1.f, 1.f };
            f32 metallic_factor = 1.f;
            f32 roughness_factor = 1.f;
        };

        auto get_material_data = [&](u32 gltf_material_idx) -> MaterialData{
            const auto &fastgltf_material = asset.materials[gltf_material_idx];

            u32 sampler_idx = [&](){
                if (fastgltf_material.pbrData.baseColorTexture.has_value()){
                    const auto &tex = asset.textures[fastgltf_material.pbrData.baseColorTexture->textureIndex];
                    if (tex.samplerIndex.has_value()){
                        return get_or_add_sampler(*tex.samplerIndex);
                    }
                }
                return default_sampler_idx;
            }();

            const auto image_idx = [&]<typename TexInfo>(
                const std::optional<TexInfo> &tex_info,
                ImageType type,
                u32 default_idx)
            {
                if (!tex_info.has_value()) return default_idx;
                const auto& tex = asset.textures[tex_info->textureIndex];
                if (!tex.imageIndex.has_value()) return default_idx;
                return get_or_add_image(*tex.imageIndex, type);
            };

            const auto &base_color = fastgltf_material.pbrData.baseColorFactor;
            return MaterialData{
                .albedo_idx = image_idx(fastgltf_material.pbrData.baseColorTexture, ImageType::Albedo, default_albedo_idx),
                .normal_map_idx = image_idx(fastgltf_material.normalTexture, ImageType::Normal, default_normal_map_idx),
                .metallic_roughness_idx = image_idx(fastgltf_material.pbrData.metallicRoughnessTexture, ImageType::MetallicRoughness, default_metallic_roughness_idx),
                .sampler_idx = sampler_idx,
                .base_color_factor = {base_color[0], base_color[1], base_color[2], base_color[3]},
                .metallic_factor = fastgltf_material.pbrData.metallicFactor,
                .roughness_factor = fastgltf_material.pbrData.roughnessFactor,
            };
        };

        const auto get_or_add_vertices = [&](const fastgltf::Primitive &prim)->u64{
            const auto *pos_it  = prim.findAttribute("POSITION");
            const auto *norm_it = prim.findAttribute("NORMAL");
            const auto *uv_it   = prim.findAttribute("TEXCOORD_0");
            const auto *col_it  = prim.findAttribute("COLOR_0");

            if (pos_it == prim.attributes.end()){
                std::println("error: mesh primitive without position attribute parsed");
                abort();
            }
            std::vector<Vertex> prim_vertices(asset.accessors[pos_it->accessorIndex].count);

            if (pos_it != prim.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                    asset, asset.accessors[pos_it->accessorIndex],
                    [&](fastgltf::math::fvec3 gltf_pos, std::size_t i) {
                        prim_vertices[i].pos = { gltf_pos.x(), gltf_pos.y(), gltf_pos.z() };
                    }
                );
            }
            if (norm_it != prim.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                    asset, asset.accessors[norm_it->accessorIndex],
                    [&](fastgltf::math::fvec3 gltf_norm, std::size_t i) {
                        prim_vertices[i].normal = { gltf_norm.x(), gltf_norm.y(), gltf_norm.z() };
                    }
                );
            }else{
                std::println("warning: mesh primitive without normal attribute parsed");
            }
            if (uv_it != prim.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(
                    asset, asset.accessors[uv_it->accessorIndex],
                    [&](fastgltf::math::fvec2 gltf_uv, std::size_t i) {
                        prim_vertices[i].u = gltf_uv.x();
                        prim_vertices[i].v = gltf_uv.y();
                    }
                );
            }else{
                std::println("warning: mesh primitive without uv coordinates parsed");
            }
            if (col_it != prim.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(
                    asset, asset.accessors[col_it->accessorIndex],
                    [&](fastgltf::math::fvec4 gltf_color, std::size_t i) {
                        prim_vertices[i].color = { gltf_color.x(), gltf_color.y(), gltf_color.z(), gltf_color.w() };
                    });
            }else{
                for (auto& vert : prim_vertices) vert.color = { 1.f, 1.f, 1.f, 1.f };
            }

            XXH128_hash_t hash = XXH3_128bits(prim_vertices.data(), prim_vertices.size() * sizeof(Vertex));
            if (auto it = dedup.vertices.find(hash); it != dedup.vertices.end()){
                return it->second;
            }
            u64 offset = curr_vertices_conceptual_offset;
            curr_vertices_conceptual_offset += util::get_data_size(prim_vertices);
            vertices.insert(vertices.end(), prim_vertices.begin(), prim_vertices.end());
            dedup.vertices[hash] = offset;
            return offset;
        };

        // return offset, amount of indices
        const auto get_or_add_indices = [&](const fastgltf::Primitive &prim) -> std::pair<u64, u32>{
            std::vector<u32> idx_buf;
            fastgltf::iterateAccessorWithIndex<u32>(
                asset, asset.accessors[prim.indicesAccessor.value()],
                [&](u32 v, std::size_t) { idx_buf.push_back(v); }
            );

            XXH128_hash_t hash = XXH3_128bits(idx_buf.data(), idx_buf.size() * sizeof(u32));
            if (auto it = dedup.indices.find(hash); it != dedup.indices.end())
                return { it->second, static_cast<u32>(idx_buf.size()) };

            u64 first = curr_indices_conceptual_offset;
            curr_indices_conceptual_offset += util::get_data_size(idx_buf);
            indices.insert(indices.end(), idx_buf.begin(), idx_buf.end());
            dedup.indices[hash] = first;
            return { first, static_cast<u32>(idx_buf.size()) };
        };

        const auto get_or_add_mesh_primitive = [&](const fastgltf::Primitive &prim)->u32{
            u64 vertex_offset = get_or_add_vertices(prim);
            auto [first_index, index_count] = get_or_add_indices(prim);
            MaterialData mat = prim.materialIndex.has_value() ? get_material_data(*prim.materialIndex)
                                                              : MaterialData();
            AssetpackMeshPrimitive mp{
                .vertex_offset= vertex_offset,
                .first_index_offset = first_index,
                .index_count = index_count,
                .sampler_idx = mat.sampler_idx,
                .albedo_idx = mat.albedo_idx,
                .normal_map_idx = mat.normal_map_idx,
                .metallic_roughness_idx = mat.metallic_roughness_idx,
                .base_color_factor = mat.base_color_factor,
                .metallic_factor = mat.metallic_factor,
                .roughness_factor = mat.roughness_factor,
            };
            return std::get<0>(deduplicate_and_add(dedup.mesh_primitives, mp, mesh_primitives));
        };

        const auto get_or_add_mesh = [&](u32 gltf_mesh_idx)->u32{
            const auto &fastgltf_mesh = asset.meshes[gltf_mesh_idx];
            AssetpackMesh ap_mesh{};
            for (const auto &prim : fastgltf_mesh.primitives){
                ap_mesh.mesh_prim_idx.push_back(get_or_add_mesh_primitive(prim));
            }
            return std::get<0>(deduplicate_and_add(dedup.meshes, ap_mesh, meshes));
        };

        const auto walk_nodes = [&](const fastgltf::Scene &fastgltf_scene){
            AssetpackScene ap_scene{};
            struct StackEntry { std::size_t node_idx; fmat4 parent_transform; };
            std::vector<StackEntry> stack;
            stack.reserve(128);
            for (auto root_idx : fastgltf_scene.nodeIndices){
                stack.push_back({ .node_idx=root_idx, .parent_transform=fmat4(1.f) });
            }

            while (!stack.empty()){
                auto [node_idx, parent_transform] = stack.back();
                stack.pop_back();

                const auto &node = asset.nodes[node_idx];
                fmat4 node_transform(1.f);
                if (const auto* trs = std::get_if<fastgltf::TRS>(&node.transform)) {
                    fmat4 T = glm::translate(fmat4(1.f), fvec3(trs->translation[0], trs->translation[1], trs->translation[2]));
                    glm::quat q(trs->rotation[3], trs->rotation[0],
                                trs->rotation[1], trs->rotation[2]);
                    fmat4 R = glm::mat4_cast(q);
                    fmat4 S = glm::scale(fmat4(1.f), fvec3(trs->scale[0], trs->scale[1], trs->scale[2]));
                    node_transform = T * R * S;
                }
                fmat4 world_transform = parent_transform * node_transform;
                if (node.meshIndex.has_value()) {
                    ap_scene.mesh_idx.push_back(get_or_add_mesh(static_cast<u32>(*node.meshIndex)));
                    ap_scene.transforms.push_back(world_transform);
                }

                for (auto child_idx : node.children){
                    stack.push_back({ .node_idx=child_idx, .parent_transform=world_transform });
                }
            }
            scenes.push_back(std::move(ap_scene));
        };

        for (const auto &fastgltf_scene : asset.scenes) walk_nodes(fastgltf_scene);
        return *this;
    }
public:
    AssetpackBuilder &build_shard(){
        build_shard(false);
        return *this;
    }
private:
    void build_shard(bool final_shard){
        const auto write_vector =
        [&]<typename T>(const std::vector<T> &vec){
            write(vec.size());
            out.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));
        };

        AssetpackShardHeader shard_header{};
        std::ios::pos_type start_of_shard = out.tellp();
        out.seekp(sizeof(shard_header), std::ios::cur);

        shard_header.offset_to_scenes = out.tellp();
        write(assetpack_segment_signifier);
        write(scenes.size());
        for (const auto &scene : scenes){
            assert(scene.mesh_idx.size() == scene.transforms.size());
            write_vector(scene.mesh_idx);
            write_vector(scene.transforms);
        }

        shard_header.offset_to_meshes = out.tellp();
        write(assetpack_segment_signifier);
        write(meshes.size());
        for (const auto &mesh : meshes){
            write_vector(mesh.mesh_prim_idx);
        }

        shard_header.offset_to_mesh_primitives = out.tellp();
        // Get the offset into the file for the start of indices and vertices, as
        // the offsets in the primitives need to be changed to point into the file
        // instead of into the conceptual array of all vertices/indices.
        const auto get_serialized_vector_size =
        []<typename T>(const std::vector<T> &vec){
            return sizeof(size_t) + (std::size(vec) * sizeof(T));
        };
        shard_header.offset_to_samplers =
            shard_header.offset_to_mesh_primitives
            + sizeof(assetpack_segment_signifier)
            + get_serialized_vector_size(mesh_primitives);
        shard_header.offset_to_image_metadata =
            shard_header.offset_to_samplers
            + sizeof(assetpack_segment_signifier)
            + get_serialized_vector_size(samplers);
        shard_header.offset_to_indices =
            shard_header.offset_to_image_metadata
            + sizeof(assetpack_segment_signifier)
            + sizeof(size_t)
            +((
                + sizeof(AssetpackImage::width)
                + sizeof(AssetpackImage::height)
                + sizeof(AssetpackImage::format)
            )* images.size());
        for (const auto &image : images){
            shard_header.offset_to_indices +=
                get_serialized_vector_size(image.mips);
        }

        shard_header.offset_to_vertices =
            shard_header.offset_to_indices
            + sizeof(assetpack_segment_signifier)
            + get_serialized_vector_size(indices);
        shard_header.offset_to_images =
            shard_header.offset_to_vertices
            + sizeof(assetpack_segment_signifier)
            + get_serialized_vector_size(vertices);
        conceptual_offsets.push_back({
            .after_end_index_conceptual_offset = curr_indices_conceptual_offset,
            .after_end_vertex_conceptual_offset = curr_vertices_conceptual_offset,
            .after_end_image_conceptual_offset = curr_images_conceptual_offset,
            .index_file_offset =  shard_header.offset_to_indices
                                  + sizeof(assetpack_segment_signifier)
                                  + sizeof(size_t),
            .vertex_file_offset = shard_header.offset_to_vertices
                                  + sizeof(assetpack_segment_signifier)
                                  + sizeof(size_t),
            .image_file_offset  = shard_header.offset_to_images
                                  + sizeof(assetpack_segment_signifier),
        });

        for (auto &prim : mesh_primitives){
            u64 vertex_offset_shard_index = 0;
            for (const auto &offset : conceptual_offsets){
                if (prim.vertex_offset >= offset.after_end_vertex_conceptual_offset) ++vertex_offset_shard_index;
            }
            u64 index_offset_shard_index = 0;
            for (const auto &offset : conceptual_offsets){
                if (prim.first_index_offset >= offset.after_end_index_conceptual_offset) ++index_offset_shard_index;
            }
            prim.vertex_offset += conceptual_offsets.at(vertex_offset_shard_index).vertex_file_offset
                               - (vertex_offset_shard_index == 0 ? 0 : conceptual_offsets.at(vertex_offset_shard_index-1).after_end_vertex_conceptual_offset);
            prim.first_index_offset += conceptual_offsets.at(index_offset_shard_index).index_file_offset
                                    - (index_offset_shard_index == 0 ? 0 : conceptual_offsets.at(index_offset_shard_index-1).after_end_index_conceptual_offset);
        }
        write(assetpack_segment_signifier);
        write_vector(mesh_primitives);

        assert(shard_header.offset_to_samplers == static_cast<u64>(out.tellp()));
        write(assetpack_segment_signifier);
        write_vector(samplers);

        for (auto &image : images){
            for (auto &mip : image.mips){
                u64 mip_offset_shard_index = 0;
                for (const auto &offset : conceptual_offsets){
                    if (mip.offset >= offset.after_end_image_conceptual_offset) ++mip_offset_shard_index;
                }
                mip.offset += conceptual_offsets.at(mip_offset_shard_index).image_file_offset 
                           - (mip_offset_shard_index == 0 ? 0 : conceptual_offsets.at(mip_offset_shard_index-1).after_end_image_conceptual_offset);
            }
        }
        assert(shard_header.offset_to_image_metadata == static_cast<u64>(out.tellp()));
        write(assetpack_segment_signifier);
        write(images.size());
        for (const auto &image : images){
            write(image.width);
            write(image.height);
            write(image.format);
            write_vector(image.mips);
        }

        assert(shard_header.offset_to_indices == static_cast<u64>(out.tellp()));
        write(assetpack_segment_signifier);
        write_vector(indices);

        assert(shard_header.offset_to_vertices == static_cast<u64>(out.tellp()));
        write(assetpack_segment_signifier);
        write_vector(vertices);

        assert(shard_header.offset_to_images == static_cast<u64>(out.tellp()));
        write(assetpack_segment_signifier);
        for (const auto &image : images){
            assert(image.mips.size() == image.local_data_mip_offset.size());
            for (u64 i = 0; i < image.mips.size(); ++i){
                out.write(reinterpret_cast<const char*>(image.data.data() + image.local_data_mip_offset.at(i)),
                          image.mips.at(i).size);
            }
        }
        shard_header.offset_to_after_end_of_shard = out.tellp();
        shard_header.offset_to_next_shard_or_0 = final_shard ? 0 : shard_header.offset_to_after_end_of_shard;

        out.seekp(start_of_shard, std::ios::beg);
        out.write(reinterpret_cast<char*>(&shard_header), sizeof(shard_header));
        out.seekp(shard_header.offset_to_after_end_of_shard, std::ios::beg);

        scenes.clear();
        meshes.clear();
        mesh_primitives.clear();
        samplers.clear();
        images.clear();
        vertices.clear();
        indices.clear();
    }
public:
    void build(){
        build_shard(true);
        std::ios::pos_type curr = out.tellp();
        out.seekp(0, std::ios::beg);
        write(AssetpackHeader::valid_file_signature);
        out.seekp(curr, std::ios::beg);
    }
};

export class AssetLoader{
    struct Offsets{
        u64 vertex_offset;
        u64 first_index_offset;
        bool operator==(const Offsets &other) const noexcept = default;
    };
    struct OffsetsHasher{
        size_t operator()(const Offsets &offsets){
            static_assert(no_added_padding<Offsets>());
            return XXH3_64bits(&offsets, sizeof(offsets));
        }
    };

    struct Scene{
        std::vector<u32> mesh_idx;
        std::vector<u32> mesh_idx_in_file;
        std::vector<fmat4> transforms;
    };
    struct Mesh{
        std::vector<u32> mesh_prim_idx;
        std::vector<u32> mesh_prim_idx_in_file;
    };
    struct MeshPrimitive{
        u32 index_count;
        u32 sampler_idx;
        // Offsets into storage buffers, indexes into vector of Textures
        struct Loaded{
            Offsets offsets;
            u32 albedo_idx;
            u32 normal_map_idx;
            u32 metallic_roughness_idx;
        } loaded;
        // Offsets into the file, indexes into vector of ImageAssets
        // which in turn contains offets into files.
        struct InFile{
            Offsets offsets;
            u32 albedo_idx;
            u32 normal_map_idx;
            u32 metallic_roughness_idx;
        } in_file;

        fvec4 base_color_factor;
        f32 metallic_factor;
        f32 roughness_factor;
    };
    struct ImageAsset{
        u32 width;
        u32 height;
        VkFormat format;
        std::vector<ByteRange> mips_in_file;
    };
    struct AssetFile{
        std::vector<Scene> scenes;
        std::vector<Mesh> meshes;
        std::vector<MeshPrimitive> primitives;
        std::vector<ImageAsset> image_assets;
    };
    std::vector<AssetFile> files;
    u32 sampler_count = 0;//temporary. todo: remove.

    // Count how many loaded
    // struct Refcount{
    //     std::unordered_map<Offsets, u64, OffsetsHasher> vertex;
    //     std::unordered_map<Offsets, u64, OffsetsHasher> index;
    //     std::unordered_map<u64, u64> texture;
    // } refcount;
    //std::vector<Texture> textures;
    //std::vector<ImageView> texture_views;
    // StorageBuffer object_transforms;
    // VertexBuffer<Vertex> vertices;
    // StorageBuffer albedo_texture_indices;
    // StorageBuffer normal_map_indices;
    // StorageBuffer matallic_roughness_map;
    // StorageBuffer obj_transform_indices;

public:
// returns the file id
    u64 check_assetpack(std::istream& in) {
        if (in.tellg() != 0) {
            std::println("Warning: istream not at start; seeking to beginning.");
            in.seekg(0, std::ios::beg);
        }
        files.emplace_back();
        AssetFile& file = files.back();

        const auto read = [&]<typename T>(T& v) {
            in.read(reinterpret_cast<char*>(&v), sizeof(v));
        };
        const auto read_vec = [&]<typename T>(std::vector<T>& v) {
            size_t n{}; read(n); v.resize(n);
            in.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
        };
        const auto check_sig = [&](u64 off, std::string_view label) -> bool {
            in.seekg(static_cast<std::streamoff>(off));
            u64 sig{}; read(sig);
            const bool ok = (sig == assetpack_segment_signifier);
            std::println("    {} at offset {} has segment signifier: {}", label, off, ok ? "OK" : "FAIL");
            return ok;
        };
        const auto check_pos = [&](u64 expected, std::string_view label) {
            const u64 actual = static_cast<u64>(in.tellg());
            if (actual != expected)
                std::println("    Error: mismatch after {}: stream is at {} but we expected to be at {}", label, actual, expected);
        };

        // file header
        u64 file_sig{}, version{};
        u32 name_len{};
        char name[AssetpackHeader::max_name_length]{};
        read(file_sig); read(version); read(name_len);
        in.read(name, AssetpackHeader::max_name_length);

        if (file_sig == AssetpackHeader::error_did_not_finish_building_file_signature) {
            std::println("Error: assetpack was not fully written (incomplete build).");
            abort();
        }
        if (file_sig != AssetpackHeader::valid_file_signature) {
            std::println("Error: bad file signature 0x{:016X} (expected 0x{:016X})", file_sig, AssetpackHeader::valid_file_signature);
            abort();
        }
        if (name_len > AssetpackHeader::max_name_length) {
            std::println("Error: name_length {} exceeds max {}", name_len, AssetpackHeader::max_name_length);
            abort();
        }
        std::println("=== Assetpack '{:.{}s}' version {} ===", name, name_len, version);

        // Accumulated data ranges across all shards - used for cross-shard validation at the end
        struct ShardRanges {
            u64 idx_start, idx_end;   // half-open byte ranges into the file
            u64 vertex_start, vertex_end;
            u64 img_start, img_end;
        };
        std::vector<ShardRanges> shard_ranges;

        for (u32 shard_num = 0; ; ++shard_num) {
            std::println("\n---Shard {}  header @ {}", shard_num, static_cast<u64>(in.tellg()));

            AssetpackShardHeader sh{};
            read(sh);

            std::println("    offset_to_scenes          = {}", sh.offset_to_scenes);
            std::println("    offset_to_meshes          = {}", sh.offset_to_meshes);
            std::println("    offset_to_mesh_primitives = {}", sh.offset_to_mesh_primitives);
            std::println("    offset_to_samplers        = {}", sh.offset_to_samplers);
            std::println("    offset_to_image_metadata  = {}", sh.offset_to_image_metadata);
            std::println("    offset_to_indices         = {}", sh.offset_to_indices);
            std::println("    offset_to_vertices        = {}", sh.offset_to_vertices);
            std::println("    offset_to_images          = {}", sh.offset_to_images);
            std::println("    offset_to_after_end       = {}", sh.offset_to_after_end_of_shard);

            const u64 offs[] = {
                sh.offset_to_scenes, sh.offset_to_meshes, sh.offset_to_mesh_primitives,
                sh.offset_to_samplers, sh.offset_to_image_metadata, sh.offset_to_indices,
                sh.offset_to_vertices, sh.offset_to_images, sh.offset_to_after_end_of_shard
            };
            for (size_t i = 1; i < std::size(offs); ++i)
                if (offs[i] <= offs[i-1])
                    std::println("    Error: offsets not strictly increasing at index {}: {} <= {}", i, offs[i], offs[i-1]);

            // Record data ranges for this shard (skipping signifier + size_t count prefix)
            shard_ranges.push_back({
                .idx_start = sh.offset_to_indices  + sizeof(assetpack_segment_signifier) + sizeof(size_t),
                .idx_end   = sh.offset_to_vertices,
                .vertex_start = sh.offset_to_vertices + sizeof(assetpack_segment_signifier) + sizeof(size_t),
                .vertex_end   = sh.offset_to_images,
                .img_start = sh.offset_to_images   + sizeof(assetpack_segment_signifier),
                .img_end   = sh.offset_to_after_end_of_shard,
            });

            // Scenes
            if (check_sig(sh.offset_to_scenes, "scenes")) {
                size_t scene_count{}; read(scene_count);
                std::println("    scenes: {}", scene_count);
                for (size_t i = 0; i < scene_count; ++i) {
                    Scene sc{};
                    read_vec(sc.mesh_idx_in_file);
                    sc.mesh_idx = sc.mesh_idx_in_file;
                    read_vec(sc.transforms);
                    if (sc.mesh_idx.size() != sc.transforms.size())
                        std::println("      Error: scene[{}] mesh/transform count mismatch", i);
                    std::println("      scene[{}]: {} meshes", i, sc.mesh_idx.size());
                    file.scenes.push_back(std::move(sc));
                }
                check_pos(sh.offset_to_meshes, "scenes");
            }

            // Meshes
            if (check_sig(sh.offset_to_meshes, "meshes")) {
                size_t mesh_count{}; read(mesh_count);
                std::println("    meshes: {}", mesh_count);
                for (size_t i = 0; i < mesh_count; ++i) {
                    Mesh m{};
                    read_vec(m.mesh_prim_idx_in_file);
                    m.mesh_prim_idx = m.mesh_prim_idx_in_file;
                    std::println("      mesh[{}]: {} primitives", i, m.mesh_prim_idx.size());
                    file.meshes.push_back(std::move(m));
                }
                check_pos(sh.offset_to_mesh_primitives, "meshes");
            }

            // Mesh primitives
            if (check_sig(sh.offset_to_mesh_primitives, "mesh_primitives")) {
                struct SerializedPrim {
                    u64   vertex_offset, first_index_offset;
                    u32   index_count, sampler_idx;
                    u32   albedo_idx, normal_map_idx, metallic_roughness_idx;
                    fvec4 base_color_factor;
                    f32   metallic_factor, roughness_factor;
                    u32   padding;
                };
                static_assert(sizeof(SerializedPrim) == 64);

                std::vector<SerializedPrim> raw;
                read_vec(raw);
                std::println("    mesh_primitives: {}", raw.size());
                for (size_t i = 0; i < raw.size(); ++i) {
                    const auto& rp = raw[i];
                    std::println(
                        "      prim[{}]: vtx_off={}  idx_off={}  idx_cnt={}"
                        "  alb={}  nrm={}  mr={}  color=[{:.2f},{:.2f},{:.2f},{:.2f}]",
                        i, rp.vertex_offset, rp.first_index_offset, rp.index_count,
                        rp.albedo_idx, rp.normal_map_idx, rp.metallic_roughness_idx,
                        rp.base_color_factor.x, rp.base_color_factor.y,
                        rp.base_color_factor.z, rp.base_color_factor.w);

                    MeshPrimitive mp{};
                    mp.index_count = rp.index_count;
                    mp.sampler_idx = rp.sampler_idx;
                    mp.in_file.offsets = { rp.vertex_offset, rp.first_index_offset };
                    mp.in_file.albedo_idx = rp.albedo_idx;
                    mp.in_file.normal_map_idx = rp.normal_map_idx;
                    mp.in_file.metallic_roughness_idx = rp.metallic_roughness_idx;
                    mp.base_color_factor = rp.base_color_factor;
                    mp.metallic_factor = rp.metallic_factor;
                    mp.roughness_factor = rp.roughness_factor;
                    file.primitives.push_back(mp);
                }
                check_pos(sh.offset_to_samplers, "mesh_primitives");
            }

            // Samplers
            if (check_sig(sh.offset_to_samplers, "samplers")) {
                std::vector<AssetpackSampler> samps;
                read_vec(samps);
                std::println("    samplers: {}", samps.size());
                sampler_count += samps.size();
                for (size_t i = 0; i < file.primitives.size(); ++i)
                    if (file.primitives[i].sampler_idx >= sampler_count)
                        std::println("      Error: prim[{}] sampler_idx={} out of range ({})", i, file.primitives[i].sampler_idx, sampler_count);
                check_pos(sh.offset_to_image_metadata, "samplers");
            }

            // Image metadata
            if (check_sig(sh.offset_to_image_metadata, "image_metadata")) {
                size_t image_count{}; read(image_count);
                std::println("    image_metadata: {}", image_count);
                for (size_t i = 0; i < image_count; ++i) {
                    u32 w{}, h{}; VkFormat fmt{};
                    read(w); read(h); read(fmt);
                    std::vector<ByteRange> mips;
                    read_vec(mips);
                    const ByteRange& m0 = mips.empty() ? ByteRange{} : mips[0];
                    std::println("      image[{}]: {}x{}  fmt={}  mips={}  mip[0]={{off={} sz={}}}",
                                 i, w, h, static_cast<u32>(fmt), mips.size(), m0.offset, m0.size);
                    file.image_assets.push_back({ w, h, fmt, std::move(mips) });
                }
                check_pos(sh.offset_to_indices, "image_metadata");

                const auto n = file.image_assets.size();
                for (size_t i = 0; i < file.primitives.size(); ++i) {
                    const auto& p = file.primitives[i];
                    if (p.in_file.albedo_idx >= n)
                        std::println("      Error: prim[{}] albedo_idx={} out of range ({})", i, p.in_file.albedo_idx, n);
                    if (p.in_file.normal_map_idx >= n)
                        std::println("      Error: prim[{}] normal_map_idx={} out of range ({})", i, p.in_file.normal_map_idx, n);
                    if (p.in_file.metallic_roughness_idx >= n)
                        std::println("      Error: prim[{}] metallic_roughness_idx={} out of range ({})", i, p.in_file.metallic_roughness_idx, n);
                }
            }

            // Indices
            if (check_sig(sh.offset_to_indices, "indices")) {
                size_t idx_cnt{}; read(idx_cnt);
                const u64 computed_end = static_cast<u64>(in.tellg()) + idx_cnt * sizeof(u32);
                std::println("    indices: {} u32s ({} bytes)", idx_cnt, idx_cnt * sizeof(u32));
                if (computed_end != sh.offset_to_vertices)
                    std::println("    Error: indices end {} != offset_to_vertices {}", computed_end, sh.offset_to_vertices);
                else
                    std::println("    indices end == offset_to_vertices : OK");
                in.seekg(static_cast<std::streamoff>(sh.offset_to_vertices));
            }

            // Vertices
            if (check_sig(sh.offset_to_vertices, "vertices")) {
                size_t vert_cnt{}; read(vert_cnt);
                const u64 computed_end = static_cast<u64>(in.tellg()) + vert_cnt * sizeof(Vertex);
                std::println("    vertices: {} ({} bytes each, {} bytes total)",
                             vert_cnt, sizeof(Vertex), vert_cnt * sizeof(Vertex));
                if (computed_end != sh.offset_to_images)
                    std::println("    Error: vertices end {} != offset_to_images {}", computed_end, sh.offset_to_images);
                else
                    std::println("    vertices end == offset_to_images : OK");
                in.seekg(static_cast<std::streamoff>(sh.offset_to_images));
            }

            // Image data
            if (check_sig(sh.offset_to_images, "image_data")) {
                const ShardRanges& sr = shard_ranges.back();
                u64 cursor = sr.img_start;
                bool all_ok = true;
                for (size_t i = 0; i < file.image_assets.size(); ++i) {
                    for (size_t m = 0; m < file.image_assets[i].mips_in_file.size(); ++m) {
                        const ByteRange& mip = file.image_assets[i].mips_in_file[m];
                        if (mip.offset < sr.img_start || mip.offset >= sr.img_end)// if it's greater or equal that's a mistake actually
                            continue; // lives in a different shard, skip
                        if (mip.offset != cursor) {
                            std::println("      Error: image[{}] mip[{}] stored offset {} != cursor {}",
                                         i, m, mip.offset, cursor);
                            all_ok = false;
                            cursor = mip.offset;
                        }
                        cursor += mip.size;
                    }
                }
                if (all_ok) std::println("    all image mip offsets in this shard : OK");
                if (cursor != sr.img_end)
                    std::println("    Error: image data end {} != offset_to_after_end {}", cursor, sr.img_end);
                else
                    std::println("    image data end == offset_to_after_end : OK");
            }

            if (sh.offset_to_next_shard_or_0 == 0) break;
            in.seekg(static_cast<std::streamoff>(sh.offset_to_next_shard_or_0));
        }

        // Cross-shard validation
        // Primitive offsets and image mip offsets can legally point into any shard.
        const auto in_any = [](const std::vector<ShardRanges>& rs, u64 off,
                                u64 ShardRanges::* start, u64 ShardRanges::* end) {
            for (const auto& r : rs) if (off >= r.*start && off < r.*end) return true;
            return false;
        };
        std::println("\n  [Cross-shard validation]");
        bool prim_ok = true;
        for (size_t i = 0; i < file.primitives.size(); ++i) {
            const auto& p = file.primitives[i];
            if (!in_any(shard_ranges, p.in_file.offsets.first_index_offset, &ShardRanges::idx_start, &ShardRanges::idx_end)) {
                std::println("    Error: prim[{}] first_index_offset={} not in any shard's index range", i, p.in_file.offsets.first_index_offset);
                prim_ok = false;
            }
            if (!in_any(shard_ranges, p.in_file.offsets.vertex_offset, &ShardRanges::vertex_start, &ShardRanges::vertex_end)) {
                std::println("    Error: prim[{}] vertex_offset={} not in any shard's vertex range", i, p.in_file.offsets.vertex_offset);
                prim_ok = false;
            }
        }
        if (prim_ok) std::println("    all primitive offsets in range : OK");
        bool img_ok = true;
        for (size_t i = 0; i < file.image_assets.size(); ++i) {
            for (size_t m = 0; m < file.image_assets[i].mips_in_file.size(); ++m) {
                const u64 off = file.image_assets[i].mips_in_file[m].offset;
                if (!in_any(shard_ranges, off, &ShardRanges::img_start, &ShardRanges::img_end)) {
                    std::println("    Error: image[{}] mip[{}] offset={} not in any shard's image range", i, m, off);
                    img_ok = false;
                }
            }
        }
        if (img_ok) std::println("    all image mip offsets in range : OK");

        std::println("\n=== load complete: {} scenes  {} meshes  {} primitives  {} images ===",
                     file.scenes.size(), file.meshes.size(),
                     file.primitives.size(), file.image_assets.size());
        return static_cast<u64>(files.size() - 1);
    }
};



























