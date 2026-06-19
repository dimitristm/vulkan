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


namespace Assetpack{

struct Sampler {
    VkFilter mag_filter;
    VkFilter min_filter;
    VkSamplerMipmapMode mipmap_mode;
    VkSamplerAddressMode wrap_s;
    VkSamplerAddressMode wrap_t;

    Sampler()
    :mag_filter(VK_FILTER_LINEAR),
    min_filter(VK_FILTER_LINEAR),
    mipmap_mode(VK_SAMPLER_MIPMAP_MODE_LINEAR),
    wrap_s(VK_SAMPLER_ADDRESS_MODE_REPEAT),
    wrap_t(VK_SAMPLER_ADDRESS_MODE_REPEAT)
    {}

    explicit Sampler(const fastgltf::Sampler& s)
    :mag_filter(s.magFilter ? to_vk_filter(*s.magFilter) : VK_FILTER_LINEAR),
    min_filter(s.minFilter ? to_vk_filter(*s.minFilter) : VK_FILTER_LINEAR),
    mipmap_mode(s.minFilter ? to_vk_mipmap(*s.minFilter) : VK_SAMPLER_MIPMAP_MODE_LINEAR),
    wrap_s(to_vk_wrap(s.wrapS)),
    wrap_t(to_vk_wrap(s.wrapT))
    {}

    [[nodiscard]] XXH128_hash_t hash() const{
        return XXH3_128bits(this, sizeof(Sampler));
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
static_assert(no_added_padding<Sampler>());

struct Scene{
    std::vector<u32> mesh_idx;
    std::vector<fmat4> transforms;
};

struct Mesh{
    std::vector<u32> mesh_prim_idx;

    [[nodiscard]] XXH128_hash_t hash() const{
        return XXH3_128bits(mesh_prim_idx.data(), mesh_prim_idx.size() * sizeof(u32));
    }
};

// all indexes are global
// all offsets that will be stored int the file are indexes into the file itself
struct IndexedVerticesMetadata{
    // The set of all *unique* ranges found in primitives must not contain
    // ranges that overlap (multiple primitives can have the same exact ranges, though).
    // This is for tracking what vertices/indices are loaded for dynamic
    // loading/unloading to gpu memory later on.
    struct Lod{
        u64 vertex_offset;
        u64 first_index_offset;
        u32 index_count;
    };

    std::vector<Lod> lods;
};
struct IndexedVertices{
    IndexedVerticesMetadata meta{};
    std::vector<Vertex> verts;
    std::vector<u32> indices;
    IndexedVertices() = default;
    IndexedVertices(const fastgltf::Asset &asset, const fastgltf::Primitive &prim){
        verts = [&]()->std::vector<Vertex>{
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
            return prim_vertices;
        }();

        indices = [&]() -> std::vector<u32>{
            std::vector<u32> indices;
            fastgltf::iterateAccessorWithIndex<u32>(
                asset, asset.accessors[prim.indicesAccessor.value()],
                [&](u32 v, std::size_t) { indices.push_back(v); }
            );
            return indices;
        }();
    }

    // We want to hash before processing to avoid useless work
    bool is_processed = false;
    XXH128_hash_t hash() const{
        assert(!is_processed && "Don't process before hashing");
        XXH3_state_t* state = XXH3_createState();
        XXH3_128bits_update(state, verts.data(), util::get_data_size(verts));
        XXH3_128bits_update(state, indices.data(), util::get_data_size(indices));
        XXH128_hash_t hash = XXH3_128bits_digest(state);
        XXH3_freeState(state);
        return hash;
    }
    // "Processing" refers to generating LoDs and optimizing.
    // this function also sets metadata, but it's unfinished. Specifically,
    // the offsets are into the data instead of into the file. The write
    // function will fix that.
    // todo: add lods and mesh optimization.
    void process(){
        assert(!is_processed && "Attempted to process an IndexedVertices that's already processed");
        if (is_processed) return;
        meta.lods.push_back({.vertex_offset = 0, .first_index_offset = 0, .index_count = static_cast<u32>(indices.size())});
        is_processed = true;
    }

    // Writes the data to the file and a metadata entry to the vector of metadata.
    void write(std::ostream &out, std::vector<IndexedVerticesMetadata> &metadata){
        u64 vertex_file_offset = out.tellp();
        out.write(reinterpret_cast<char*>(verts.data()), util::get_data_size(verts));
        u64 index_file_offset = out.tellp();
        out.write(reinterpret_cast<char*>(indices.data()), util::get_data_size(indices));
        for (auto &entry : meta.lods){
            entry.vertex_offset += vertex_file_offset;
            entry.first_index_offset += index_file_offset;
        }
        metadata.push_back(meta);
    }
};
struct MeshPrimitive{
    u32 indexed_verts_meta_idx;
    u32 sampler_idx;
    u32 albedo_meta_idx;
    u32 normal_map_meta_idx;
    u32 metallic_roughness_meta_idx;
    fvec4 base_color_factor { 1.f, 1.f, 1.f, 1.f };
    f32 metallic_factor = 1.f;
    f32 roughness_factor = 1.f;

    [[nodiscard]] XXH128_hash_t hash() const{
        return XXH3_128bits(this, sizeof(*this));
    }
};
static_assert(no_added_padding<MeshPrimitive>());

struct ImageMetadata{
    u32 width;
    u32 height;
    VkFormat format;
    std::vector<ByteRange> mips;
    static_assert(no_added_padding<ByteRange>());
};

struct Image{
    enum class Type : u32{
        Albedo = 0,
        Normal = 1,
        MetallicRoughness = 2,
    };
    ImageMetadata meta;
    std::vector<std::byte> data;
    const uint8_t *pixels;
    bool is_processed = false;
    Type type;

    Image() = default;
    Image(const uint8_t *pixels, u32 width, u32 height, Type type)
    :meta({.width = width, .height = height, .format = (type == Type::Albedo) ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC5_UNORM_BLOCK}),
    pixels(pixels), type(type)
    { }

    void process(){
        assert(!is_processed && "Attempted to process an Image that's already processed");
        if(is_processed) return;
        std::vector<uint8_t> swizzled_storage;
        const uint8_t *src = pixels;
        if (type == Type::MetallicRoughness){
            swizzled_storage.resize(meta.width * meta.height * 4);
            for (u32 i = 0; i < meta.width * meta.height; ++i){
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
            .vkFormat = (type == Type::Albedo) ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM,
            .baseWidth = meta.width,
            .baseHeight = meta.height,
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
        std::memcpy(tex->pData, src, meta.width * meta.height * 4);
        ktxBasisParams params{};
        params.structSize = sizeof(params);
        params.uastc      = KTX_TRUE;
        params.uastcFlags = KTX_PACK_UASTC_LEVEL_DEFAULT; //todo: use a higher level
        params.verbose    = KTX_FALSE;
        params.threadCount = 16;
        KTX_CHECK(ktxTexture2_CompressBasisEx(tex, &params));
        ktx_transcode_fmt_e tfmt = (type == Type::Albedo) ? KTX_TTF_BC7_RGBA : KTX_TTF_BC5_RG;
        KTX_CHECK(ktxTexture2_TranscodeBasis(tex, tfmt, 0));
        t2.end();
        //std::println("compress and transcode took: {}", t2.elapsed());

        u64 data_offset = 0;
        for (u32 mip_level = 0; mip_level < tex->numLevels; ++mip_level){
            ktx_size_t mip_offset = 0;
            ktxTexture_GetImageOffset(ktxTexture(tex), mip_level, 0, 0, &mip_offset);
            ktx_size_t mip_size = ktxTexture_GetImageSize(ktxTexture(tex), mip_level);
            this->meta.mips.push_back({.offset = data_offset, .size = mip_size});
            this->data.resize(data.size() + mip_size);
            std::memcpy(this->data.data() + data_offset, tex->pData + mip_offset, mip_size);
            data_offset += mip_size;
        }
        ktxTexture_Destroy(ktxTexture(tex));
        is_processed = true;
    }

    void write(std::ostream &out, std::vector<ImageMetadata> &metadata){
        for(auto &entry : meta.mips) entry.offset += out.tellp();
        out.write(reinterpret_cast<char*>(data.data()), util::get_data_size(data));
        metadata.push_back(meta);
    }

    [[nodiscard]] XXH128_hash_t hash() const{
        assert(!is_processed && "Don't process an image before hashing it");
        XXH3_state_t* state = XXH3_createState();
        XXH3_128bits_update(state, pixels, meta.width * meta.height * 4);
        XXH3_128bits_update(state, &type, sizeof(type));
        XXH128_hash_t hash = XXH3_128bits_digest(state);
        XXH3_freeState(state);
        return hash;
    }
};


struct Header{
    enum class Signatures : u64{
        valid = 0xBF269A65A2E91226,
        unfinished = 0x85AC8E6951FC737D,
        invalid = 0x2FD66A0C4AF56280,
        valid_end_of_file = 0x2B7FE37FC301CB84,
        invalid_end_of_file = 0x88F19A99ED09E0C7,
    } signature;
};
// Between the Header and ToC, there will be data: images, vertices, indices.

struct TableOfContents{
    std::vector<Scene> scenes;
    std::vector<Sampler> samplers;
    std::vector<Mesh> meshes;
    std::vector<MeshPrimitive> mesh_primitives;
    std::vector<ImageMetadata> image_metadata;
    std::vector<IndexedVerticesMetadata> indexed_verts_metadata;
};
struct Footer{
    u64 offset_to_toc;
    enum class EndSignature : u64{
        valid_end_of_file = 0x2B7FE37FC301CB84,
        invalid_end_of_file = 0x88F19A99ED09E0C7,
    } end_sig;
};

export class Builder{
    using HashToIdx = XXH128Map<u64>;
    struct DeduplicationMaps{
        // AssetpackX.hash() -> index in the assetpack
        HashToIdx meshes;
        HashToIdx mesh_primitives;
        HashToIdx indexed_verts_meta;
        HashToIdx samplers;
        HashToIdx images_meta;
    };
    DeduplicationMaps dedup;

    TableOfContents toc;
    std::fstream out;

    template<typename T>
    void write(const T& data){ out.write(reinterpret_cast<const char*>(&data), sizeof(data)); }
    
    template<typename T>
    void write_vector(const std::vector<T>& v){
        write(v.size());
        out.write(reinterpret_cast<const char*>(v.data()), util::get_data_size(v));
    }

public:
    Builder(const std::filesystem::path &filepath)
    :out(filepath, std::ios::binary | std::ios::trunc | std::ios::out)
    {
        Header header{.signature = Header::Signatures::unfinished};
        write(header);
    }

    Builder &add_from_gltf(const std::filesystem::path &path){
        struct GltfPreviouslyLoadedImage{
            u32 image_idx;
            Image::Type type;
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

        //returns the index of the entry
        static constexpr auto deduplicate_and_add_ToC_entry =
        []<Hashable T>(HashToIdx &dedup, T &item, std::vector<T> &storage)->u32
        {
            XXH128_hash_t hash = item.hash();
            if (auto it = dedup.find(hash); it != dedup.end()){
                return it->second;
            }
            u32 idx = static_cast<u32>(dedup.size());
            storage.push_back(item);
            dedup[hash] = idx;
            return idx;
        };
        // returns the idx of the toc entry. the metadata toc entry in turn points to the data in the file.
        const auto deduplicate_data_and_add_Toc_metadata =
        [&] <Hashable T, typename Y> (HashToIdx &dedup, T &data, std::vector<Y> &toc_metadata_vector)->u32{
            using metadata_type = std::remove_cvref_t<decltype(T::meta)>;
            static_assert(std::is_same_v<metadata_type, std::remove_cvref_t<Y>>);
            XXH128_hash_t hash = data.hash();
            if (auto it = dedup.find(hash); it != dedup.end()){
                return it->second;
            };
            u32 idx = static_cast<u32>(dedup.size());
            dedup[hash] = idx;
            data.process();
            data.write(out, toc_metadata_vector);
            return idx;
        };

        const auto create_or_get_default_sampler_idx = [&]()->u32{
            Sampler sampler{};
            return deduplicate_and_add_ToC_entry(dedup.samplers, sampler, toc.samplers);
        };

        const auto create_or_get_default_image_meta_idx = [&](Image::Type type)->u32{
            constexpr uint8_t magenta[4] = { 255, 0, 255, 255 };
            constexpr uint8_t flat[4] = { 128, 128, 255, 255 };
            constexpr uint8_t white[4] = { 255, 255, 255, 255 };
            Image img;
            switch (type){
            case Image::Type::Albedo:
                img = Image(magenta, 1, 1, type); break;
            case Image::Type::Normal:
                img = Image(flat, 1, 1, type); break;
            case Image::Type::MetallicRoughness:
                img = Image(white, 1, 1, type); break;
            }
            return deduplicate_data_and_add_Toc_metadata(dedup.images_meta, img, toc.image_metadata);
        };

        const auto get_or_add_sampler = [&](u32 gltf_sampler_idx) -> u32{
            Sampler sampler(asset.samplers[gltf_sampler_idx]);
            return deduplicate_and_add_ToC_entry(dedup.samplers, sampler, toc.samplers);
        };

        const auto get_or_add_image = [&](u32 gltf_img_idx, Image::Type type) -> u32{
            GltfPreviouslyLoadedImage prev_loaded_img{
                .image_idx = gltf_img_idx,
                .type = type,
            };
            XXH128_hash_t gltf_hash = prev_loaded_img.hash();
            if (auto it = loaded_gltf_images.find(gltf_hash); it != loaded_gltf_images.end()){
                return it->second;
            }

            std::vector<std::byte> raw_image_bytes = [&]() -> std::vector<std::byte>{
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
            }();

            int width, height, channels;
            stbi_uc* pixels = stbi_load_from_memory(
                reinterpret_cast<const stbi_uc*>(raw_image_bytes.data()),
                static_cast<int>(raw_image_bytes.size()),
                &width, &height, &channels, 4);
            if (!pixels) { std::println("stbi failed to decode image"); abort(); }

            Image img(pixels, static_cast<u32>(width), static_cast<u32>(height), type);

            u32 idx = deduplicate_data_and_add_Toc_metadata(dedup.images_meta, img, toc.image_metadata);
            stbi_image_free(pixels);
            loaded_gltf_images[gltf_hash] = idx;
            return idx;
        };

        struct MaterialData {
            u32 albedo_meta_idx;
            u32 normal_map_meta_idx;
            u32 metallic_roughness_meta_idx;
            u32 sampler_idx;
            fvec4 base_color_factor = { 1.f, 1.f, 1.f, 1.f };
            f32 metallic_factor = 1.f;
            f32 roughness_factor = 1.f;
        };

        const auto get_material_data = [&](const fastgltf::Primitive &prim) -> MaterialData{
            if(prim.materialIndex.has_value()){
                const auto &fastgltf_material = asset.materials[*prim.materialIndex];

                u32 sampler_idx = [&](){
                    if (fastgltf_material.pbrData.baseColorTexture.has_value()){
                        const auto &tex = asset.textures[fastgltf_material.pbrData.baseColorTexture->textureIndex];
                        if (tex.samplerIndex.has_value()){
                            return get_or_add_sampler(*tex.samplerIndex);
                        }
                    }
                    return create_or_get_default_sampler_idx();
                }();

                const auto image_idx =
                [&]<typename TexInfo>(const std::optional<TexInfo> &tex_info, Image::Type type) {
                    if (!tex_info.has_value()) return create_or_get_default_image_meta_idx(type);
                    const auto& tex = asset.textures[tex_info->textureIndex];
                    if (!tex.imageIndex.has_value()) return create_or_get_default_image_meta_idx(type);
                    return get_or_add_image(*tex.imageIndex, type);
                };

                const auto &base_color = fastgltf_material.pbrData.baseColorFactor;
                return MaterialData{
                    .albedo_meta_idx = image_idx(fastgltf_material.pbrData.baseColorTexture, Image::Type::Albedo),
                    .normal_map_meta_idx = image_idx(fastgltf_material.normalTexture, Image::Type::Normal),
                    .metallic_roughness_meta_idx = image_idx(fastgltf_material.pbrData.metallicRoughnessTexture, Image::Type::MetallicRoughness),
                    .sampler_idx = sampler_idx,
                    .base_color_factor = {base_color[0], base_color[1], base_color[2], base_color[3]},
                    .metallic_factor = fastgltf_material.pbrData.metallicFactor,
                    .roughness_factor = fastgltf_material.pbrData.roughnessFactor,
                };
            }else{
                return MaterialData{
                    .albedo_meta_idx = create_or_get_default_image_meta_idx(Image::Type::Albedo),
                    .normal_map_meta_idx = create_or_get_default_image_meta_idx(Image::Type::Normal),
                    .metallic_roughness_meta_idx = create_or_get_default_image_meta_idx(Image::Type::MetallicRoughness),
                    .sampler_idx = create_or_get_default_sampler_idx(),
                    .base_color_factor = { 1.f, 1.f, 1.f, 1.f },
                    .metallic_factor = 1.f,
                    .roughness_factor = 1.f,
                };
            }
        };

        const auto get_or_add_mesh_primitive = [&](const fastgltf::Primitive &prim)->u32{
            IndexedVertices indexed_verts(asset, prim);
            u32 idx = deduplicate_data_and_add_Toc_metadata(dedup.indexed_verts_meta, indexed_verts, toc.indexed_verts_metadata);
            MaterialData mat = get_material_data(prim);

            MeshPrimitive mp{
                .indexed_verts_meta_idx = idx,
                .sampler_idx = mat.sampler_idx,
                .albedo_meta_idx = mat.albedo_meta_idx,
                .normal_map_meta_idx = mat.normal_map_meta_idx,
                .metallic_roughness_meta_idx = mat.metallic_roughness_meta_idx,
                .base_color_factor = mat.base_color_factor,
                .metallic_factor = mat.metallic_factor,
                .roughness_factor = mat.roughness_factor,
            };
            return deduplicate_and_add_ToC_entry(dedup.mesh_primitives, mp, toc.mesh_primitives);
        };

        const auto get_or_add_mesh = [&](u32 gltf_mesh_idx)->u32{
            Mesh mesh{};
            for (const auto &prim : asset.meshes[gltf_mesh_idx].primitives){
                mesh.mesh_prim_idx.push_back(get_or_add_mesh_primitive(prim));
            }
            return deduplicate_and_add_ToC_entry(dedup.meshes, mesh, toc.meshes);
        };

        const auto walk_nodes = [&](const fastgltf::Scene &fastgltf_scene){
            Scene ap_scene{};
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
            toc.scenes.push_back(std::move(ap_scene));
        };

        for (const auto &fastgltf_scene : asset.scenes) walk_nodes(fastgltf_scene);
        return *this;
    }

    Builder &build(){
        write(toc.scenes.size());
        for (const auto &scene : toc.scenes){
            write_vector(scene.mesh_idx);
            write_vector(scene.transforms);
        }

        write_vector(toc.samplers);

        write(toc.meshes.size());
        for (const auto &mesh: toc.meshes) write_vector(mesh.mesh_prim_idx);

        write_vector(toc.mesh_primitives);

        write(toc.image_metadata.size());
        for (const auto &img_meta : toc.image_metadata){
            write(img_meta.width);
            write(img_meta.height);
            write(img_meta.format);
            write_vector(img_meta.mips);
        }

        write(toc.indexed_verts_metadata.size());
        for (const auto &idx_ver_meta : toc.indexed_verts_metadata) write_vector(idx_ver_meta.lods);

        Footer footer{.offset_to_toc = static_cast<u64>(out.tellp()), .end_sig = Footer::EndSignature::valid_end_of_file};
        static_assert(no_added_padding<Footer>());
        write(footer);

        Header done{.signature = Header::Signatures::valid};
        out.seekp(0, std::ios::beg);
        write(done.signature);
        return *this;
    }
};
}

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
};
