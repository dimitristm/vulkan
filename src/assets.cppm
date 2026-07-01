module;

#include <SDL3/SDL.h>
#include <SDL3/SDL_error.h>
#include <stb/stb_image.h>
#include <vulkan/vulkan_core.h>
#include <glm/glm.hpp>
#include <xxhash/xxhash.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <boost/pfr.hpp>
#include <SDL3/SDL_asyncio.h>
#include <bc7enc/bc7enc.h>
#include <bc7enc/rgbcx.h>

#if !USE_IMPORT_STD
#include <unordered_map>
#include <print>
#include <filesystem>
#include <vector>
#include <cstring>
#include <variant>
#include <concepts>
#include <mutex>
#include <execution>
#include <ranges>
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

static void XXH_CHECK(XXH_errorcode result){
    if (result == XXH_ERROR) {std::println("XXH error"); abort();}
}

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

    bool operator==(const Sampler&) const = default;

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
    [[nodiscard]] XXH128_hash_t hash() const{
        assert(!is_processed && "Don't process before hashing");
        XXH3_state_t* state = XXH3_createState();
        if (state == nullptr) {std::println("XXH3_createState returned null");}
        XXH_CHECK(XXH3_128bits_reset(state));
        XXH_CHECK(XXH3_128bits_update(state, verts.data(), util::get_data_size(verts)));
        XXH_CHECK(XXH3_128bits_update(state, indices.data(), util::get_data_size(indices)));
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
    const uint8_t *pixels_RGBA;
    bool is_processed = false;
    Type type;

    Image() = default;
    Image(const uint8_t *pixels_RGBA, u32 width, u32 height, Type type)
    :meta({.width = width, .height = height, .format = (type == Type::Albedo) ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC5_UNORM_BLOCK, .mips{} }),
    pixels_RGBA(pixels_RGBA), type(type)
    { }

    // Use BCn compression on the image.
    void process(){
        assert(!is_processed && "Attempted to process an Image that's already processed");
        if(is_processed) return;

        const uint8_t *src = pixels_RGBA;

        const u32 block_amount_x = (meta.width  + 3) / 4;
        const u32 block_amount_y = (meta.height + 3) / 4;
        const u64 total_blocks = block_amount_x * block_amount_y;
        // Both BC7 and BC5 encode to 16 bytes per 4x4 block
        const u64 compressed_image_size = total_blocks * 16;
        this->data.resize(compressed_image_size);

        // Extract one 4x4 RGBA block from src, clamping at image edges
        const auto extract_block = [&](u32 block_x, u32 block_y, uint8_t out[64]){
            for (u32 block_pixel_y = 0; block_pixel_y < 4; ++block_pixel_y){
                for (u32 block_pixel_x = 0; block_pixel_x < 4; ++block_pixel_x){
                    u32 src_x = std::min(block_x * 4 + block_pixel_x, meta.width - 1);
                    u32 src_y = std::min(block_y * 4 + block_pixel_y, meta.height - 1);
                    const uint8_t *src_pixel = src + (src_y * meta.width + src_x) * 4;
                    uint8_t *o = out + (block_pixel_y * 4 + block_pixel_x) * 4;
                    o[0] = src_pixel[0]; o[1] = src_pixel[1]; o[2] = src_pixel[2]; o[3] = src_pixel[3];
                }
            }
        };

        auto block_grid = std::views::cartesian_product(
            std::views::iota(0u, block_amount_y),
            std::views::iota(0u, block_amount_x)
        );

        if (type == Type::Albedo){
            bc7enc_compress_block_params params;
            bc7enc_compress_block_params_init(&params);
            params.m_uber_level = 4;

            std::for_each(std::execution::par_unseq, block_grid.begin(), block_grid.end(),
                [&](const auto &block_coordinates) {
                    auto [block_y, block_x] = block_coordinates;
                    uint8_t block[64];
                    extract_block(block_x, block_y, block);
                    u32 linear_index = block_y * block_amount_x + block_x;
                    std::byte *dst = reinterpret_cast<std::byte*>(this->data.data()) + linear_index * 16;
                    bc7enc_compress_block(dst, block, &params);
                }
            );

        } else if (type == Type::Normal || type == Type::MetallicRoughness){
            i32 channel0{}, channel1{};
            if (type == Type::Normal) {channel0 = 0; channel1 = 1;}
            if (type == Type::MetallicRoughness) {channel0 = 1; channel1 = 2;}
            std::for_each(std::execution::par_unseq, block_grid.begin(), block_grid.end(),
                [&](const auto &block_coordinates) {
                    auto [block_y, block_x] = block_coordinates;
                    uint8_t block[64];
                    extract_block(block_x, block_y, block);
                    u32 linear_index = block_y * block_amount_x + block_x;
                    std::byte *dst = reinterpret_cast<std::byte*>(this->data.data()) + linear_index * 16;
                    // Metallic and Roughness are stored in the G and B channels for gltfs
                    rgbcx::encode_bc5(dst, block, channel0, channel1, 4);
                }
            );
        } else {std::println("Unknown Image type"); abort();}

        this->meta.mips.push_back({.offset = 0, .size = compressed_image_size});

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
        if (state == nullptr) {std::println("XXH3_createState returned null");}
        XXH_CHECK(XXH3_128bits_reset(state));
        XXH_CHECK(XXH3_128bits_update(state, pixels_RGBA, meta.width * meta.height * 4));
        XXH_CHECK(XXH3_128bits_update(state, &type, sizeof(type)));
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
    } signature;
};
// Between the Header and ToC, there will be:
// 1. Data blob: images, vertices, indices.
// 2. The samplers.

struct TableOfContents{
    std::vector<Scene> scenes;
    std::vector<Mesh> meshes;
    std::vector<MeshPrimitive> mesh_primitives;
    std::vector<ImageMetadata> image_metadata;
    std::vector<IndexedVerticesMetadata> indexed_verts_metadata;
};
struct Footer{
    u64 offset_to_samplers;
    u64 offset_to_toc;
    u64 assetpack_version;
    enum class EndSignature : u64{
        valid_end_of_file = 0x2B7FE37FC301CB84,
        invalid_end_of_file = 0x88F19A99ED09E0C7,
    } end_signature;
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

    std::vector<Sampler> samplers;
    TableOfContents toc;
    std::fstream out;

    template<typename T>
    void write(const T& data){ out.write(reinterpret_cast<const char*>(&data), sizeof(data)); }

    template<typename T>
    void write_vector(const std::vector<T>& v){
        write(v.size());
        out.write(reinterpret_cast<const char*>(v.data()), util::get_data_size(v));
    }

    inline static std::once_flag init_bc7e_once{};
public:
    Builder(const std::filesystem::path &filepath)
    :out(filepath, std::ios::binary | std::ios::trunc | std::ios::out)
    {
        std::call_once(init_bc7e_once, [](){
            bc7enc_compress_block_init();
            rgbcx::init();
        });
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

        // If "item" is not already in storage, adds "item" as an entry into "storage". Return the index of the entry.
        // "dedup" must track the entries of storage.
        static constexpr auto deduplicate_and_add_entry =
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
            return deduplicate_and_add_entry(dedup.samplers, sampler, samplers);
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
            return deduplicate_and_add_entry(dedup.samplers, sampler, samplers);
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
            return deduplicate_and_add_entry(dedup.mesh_primitives, mp, toc.mesh_primitives);
        };

        const auto get_or_add_mesh = [&](u32 gltf_mesh_idx)->u32{
            Mesh mesh{};
            for (const auto &prim : asset.meshes[gltf_mesh_idx].primitives){
                mesh.mesh_prim_idx.push_back(get_or_add_mesh_primitive(prim));
            }
            return deduplicate_and_add_entry(dedup.meshes, mesh, toc.meshes);
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

        u64 offset_to_samplers = out.tellp();
        write_vector(samplers);

        Footer footer{
            .offset_to_samplers = offset_to_samplers,
            .offset_to_toc = static_cast<u64>(out.tellp()),
            .assetpack_version = 0,
            .end_signature = Footer::EndSignature::valid_end_of_file
        };
        static_assert(no_added_padding<Footer>());

        write(toc.scenes.size());
        for (const auto &scene : toc.scenes){
            write_vector(scene.mesh_idx);
            write_vector(scene.transforms);
        }

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

        write(footer);

        Header done{.signature = Header::Signatures::valid};
        out.seekp(0, std::ios::beg);
        write(done.signature);
        out.flush();
        return *this;
    }
};
}


export class ResourceLoader{
public:
    ResourceLoader(const ResourceLoader&) = delete;
    ResourceLoader(ResourceLoader&&) = delete;
    ResourceLoader& operator=(const ResourceLoader&) = delete;
    ResourceLoader& operator=(ResourceLoader&&) = delete;
private:

    class File{
    public:
        const std::filesystem::path filepath;
        SDL_AsyncIO * in;
        File(const std::filesystem::path &filepath)
        :filepath(std::filesystem::canonical(filepath)),//todo: make sure we always give a canonical so we don't have to transform it here
        in(SDL_AsyncIOFromFile(this->filepath.string().c_str(), "r"))
        {
            if(in == nullptr){
                std::println("AssetLoader::File couldn't open file {}. SDL Error: {}", this->filepath.string(), SDL_GetError());
                abort();
            }
        }
    };
    using FileID = u32; //FileID will just be the index into the files vector

    struct Scene{
        std::vector<u32> mesh_idx;
        std::vector<fmat4> transforms;
    };

    struct Mesh{
        std::vector<u32> mesh_prim_idx;
    };

    struct MeshPrimitive{
        u32 indexed_verts_idx;
        u32 sampler_idx;
        // Offsets into image assets
        u32 albedo_idx;
        u32 normal_map_idx;
        u32 metallic_roughness_idx;

        fvec4 base_color_factor;
        f32 metallic_factor;
        f32 roughness_factor;
    };

    struct ImageAsset{
        FileID file_idx;
        u32 width;
        u32 height;
        VkFormat format;
        std::vector<ByteRange> mips_in_file;

        static constexpr u32 VULKAN_IMAGE_NOT_LOADED = std::numeric_limits<u32>::max();
        u32 vulkan_image_idx;
    };

    struct IndexedVerticesAsset{
        struct Lod{
            static constexpr u64 INDEXED_VERTICES_NOT_LOADED = std::numeric_limits<u64>::max();
            // Index into the vertex buffer, not a byte offset
            u64 gpu_first_vertex_idx;
            // Index into the index buffer, not a byte offset
            u64 gpu_first_index_idx;

            u64 file_vertex_offset;
            u64 file_index_offset;
            u32 index_count;
            [[nodiscard]] u64 vertex_data_size() const { return file_index_offset - file_vertex_offset; }
            [[nodiscard]] u64 vertex_count() const { return vertex_data_size() / sizeof(Vertex); }
            [[nodiscard]] u64 index_data_size() const { return index_count * sizeof(u32); }
        };
        FileID file_idx;
        std::vector<Lod> lods;
    };

    SDL_AsyncIOQueue *queue;
    std::vector<File> files;
    std::vector<Scene> scenes;
    std::vector<Mesh> meshes;
    std::vector<MeshPrimitive> primitives;
    std::vector<ImageAsset> image_assets;
    std::vector<IndexedVerticesAsset> indexed_verts;

    struct SamplerHash {
        XXH64_hash_t operator()(const Assetpack::Sampler &s) const noexcept{
            return XXH3_64bits(&s, sizeof(s));
        }
    };
    // Indexes into the vulkan Samplers
    std::unordered_map<Assetpack::Sampler, u32, SamplerHash> sampler_idx;

    struct VulkanResources{
        static constexpr u64 INSTANCE_BUFFER_SIZE = 10000;
        VulkanEngine &vk;
        CommandPool cmd_pool;
        ImmediateSubmitter submitter;
        HostToDeviceUploader uploader;

        std::vector<Texture> textures;
        std::vector<ImageView> texture_views;
        VertexBuffer<Vertex> vertices;
        IndexBuffer indices;
        StorageBuffer object_transforms;
        StorageBuffer albedo_texture_indices;
        StorageBuffer normal_map_indices;
        StorageBuffer matallic_roughness_map;
        StorageBuffer obj_transform_indices;
        DescriptorSet graphics_desc_set;

        Sampler sampler;

        u64 vertex_cursor{};
        u64 index_cursor{};

        VulkanResources(VulkanEngine &vk):
            vk(vk),
            cmd_pool(vk),
            submitter(vk, cmd_pool),
            uploader(&vk, cmd_pool, 64UL * 1024 * 1024),
            vertices(vk, 1500000),
            indices(vk, 1000000),
            object_transforms(vk, 10000, false, true),
            albedo_texture_indices(vk, INSTANCE_BUFFER_SIZE, false, true),
            normal_map_indices(vk, INSTANCE_BUFFER_SIZE, false, true),
            matallic_roughness_map(vk, INSTANCE_BUFFER_SIZE, false, true),
            obj_transform_indices(vk, INSTANCE_BUFFER_SIZE, false, true),
            graphics_desc_set([&](){
                return DescriptorSetBuilder()
                    .bind(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000, VK_SHADER_STAGE_FRAGMENT_BIT)
                    .bind(1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT)
                    .bind(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT)
                    .bind(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT)
                    .bind(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT)
                    .build(vk);
            }()),
            sampler(vk, VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_FALSE, 0)
        {
            graphics_desc_set.update_sampler(vk, 1, sampler);
            graphics_desc_set.update_storage_buffer(vk, 2, obj_transform_indices);
            graphics_desc_set.update_storage_buffer(vk, 3, albedo_texture_indices);
            graphics_desc_set.update_storage_buffer(vk, 4, object_transforms);
        }
    };
    std::optional<VulkanResources> vk_resources = std::nullopt;
public:
    const VertexBuffer<Vertex> &get_vertex_buffer() { assert(vk_resources.has_value()); return vk_resources->vertices; }
    const IndexBuffer &get_index_buffer() { assert(vk_resources.has_value()); return vk_resources->indices; }
    const DescriptorSet &get_descriptor_set() { assert(vk_resources.has_value()); return vk_resources->graphics_desc_set; }

    ResourceLoader():
    queue(SDL_CreateAsyncIOQueue())
    {
        if (queue == nullptr) {
            std::println("Could not create async I/O queue: {}", SDL_GetError());
            abort();
        }
    }
    ~ResourceLoader(){
        for (const auto &file : files){
            if (!SDL_CloseAsyncIO(file.in, false, queue, nullptr)) {
                std::println("close failed: {}", SDL_GetError());
            }
        }
        SDL_DestroyAsyncIOQueue(queue);
    }

    FileID load_assetpack_table_of_contents(const std::filesystem::path &filepath){
        files.emplace_back(filepath);
        FileID file_id = files.size() - 1;
        const File& file = files.back();

        u64 file_size = [&](){
            i64 raw_size = SDL_GetAsyncIOSize(file.in);
            if (raw_size < 0) {
                std::println("Could not get size of '{}': {}", file.filepath.string(), SDL_GetError());
                abort();
            }
            return raw_size;
        }();

        if (file_size < sizeof(Assetpack::Header) + sizeof(Assetpack::Footer)) {
            std::println("'{}': file too small", file.filepath.string());
            abort();
        }

        std::vector<std::byte> buf(file_size);
        {
            SDL_AsyncIOOutcome outcome{};
            SDL_ReadAsyncIO(file.in, buf.data(), 0, file_size, queue, nullptr);
            if (!SDL_WaitAsyncIOResult(queue, &outcome, -1) || outcome.result != SDL_ASYNCIO_COMPLETE) {
                std::println("Read failed for '{}': {}", file.filepath.string(), SDL_GetError());
                abort();
            }
        }

        const std::byte* p = buf.data();
        const std::byte* end = buf.data() + file_size;

        const auto read = [&]<typename T>(T& v) {
            assert(p + sizeof(T) <= end && "read past end of file");
            std::memcpy(&v, p, sizeof(T));
            p += sizeof(T);
        };
        const auto read_vec = [&]<typename T>(std::vector<T>& v) {
            size_t n{}; read(n);
            assert(p + n * sizeof(T) <= end && "read_vec past end of file");
            v.resize(n);
            std::memcpy(v.data(), p, n * sizeof(T));
            p += n * sizeof(T);
        };
        const auto seek = [&](u64 offset) {
            assert(offset <= file_size && "seek past end of file");
            p = buf.data() + offset;
        };

        // Header
        Assetpack::Header header{};
        read(header);
        if (header.signature == Assetpack::Header::Signatures::unfinished) {
            std::println("'{}': build() did not complete", file.filepath.string());
            abort();
        }
        if (header.signature != Assetpack::Header::Signatures::valid) {
            std::println("'{}': unrecognised header signature 0x{:016X}", file.filepath.string(),
                static_cast<u64>(header.signature));
            abort();
        }

        // Footer
        Assetpack::Footer footer{};
        seek(file_size - sizeof(Assetpack::Footer));
        read(footer);
        if (footer.end_signature != Assetpack::Footer::EndSignature::valid_end_of_file) {
            std::println("'{}': bad footer end signature", file.filepath.string());
            abort();
        }

        // Samplers
        std::vector<Assetpack::Sampler> file_samplers;
        seek(footer.offset_to_samplers);
        read_vec(file_samplers);
        for (const auto& s : file_samplers) {
            if (!sampler_idx.contains(s)) {
                u32 idx = sampler_idx.size();
                sampler_idx[s] = idx;
            }
        }

        seek(footer.offset_to_toc);

        u32 mesh_base = meshes.size();
        u32 prim_base = primitives.size();
        u32 img_base = image_assets.size();
        u32 iv_base = indexed_verts.size();

        // Scenes
        size_t scene_count{}; read(scene_count);
        for (size_t i = 0; i < scene_count; ++i) {
            Scene sc{};
            read_vec(sc.mesh_idx);
            read_vec(sc.transforms);
            for (auto& idx : sc.mesh_idx) idx += mesh_base;
            scenes.push_back(std::move(sc));
        }

        // Meshes
        size_t mesh_count{}; read(mesh_count);
        for (size_t i = 0; i < mesh_count; ++i) {
            Mesh m{};
            read_vec(m.mesh_prim_idx);
            for (auto& idx : m.mesh_prim_idx) idx += prim_base;
            meshes.push_back(std::move(m));
        }

        // Primitives
        {
            std::vector<Assetpack::MeshPrimitive> file_prims;
            read_vec(file_prims);
            for (const auto& fp : file_prims) {
                primitives.push_back({
                    .indexed_verts_idx = fp.indexed_verts_meta_idx + iv_base,
                    .sampler_idx = sampler_idx.at(file_samplers[fp.sampler_idx]),
                    .albedo_idx = fp.albedo_meta_idx + img_base,
                    .normal_map_idx = fp.normal_map_meta_idx + img_base,
                    .metallic_roughness_idx = fp.metallic_roughness_meta_idx + img_base,
                    .base_color_factor = fp.base_color_factor,
                    .metallic_factor = fp.metallic_factor,
                    .roughness_factor = fp.roughness_factor,
                });
            }
        }

        // Image metadata
        size_t img_count{}; read(img_count);
        for (size_t i = 0; i < img_count; ++i) {
            ImageAsset ia{};
            ia.file_idx = file_id;
            ia.vulkan_image_idx = ImageAsset::VULKAN_IMAGE_NOT_LOADED;
            read(ia.width);
            read(ia.height);
            read(ia.format);
            read_vec(ia.mips_in_file);
            image_assets.push_back(std::move(ia));
        }

        // Indexed vertices metadata
        size_t iv_count{}; read(iv_count);
        for (size_t i = 0; i < iv_count; ++i) {
            IndexedVerticesAsset iva{};
            iva.file_idx = file_id;
            size_t lod_count{}; read(lod_count);
            for (size_t l = 0; l < lod_count; ++l) {
                Assetpack::IndexedVerticesMetadata::Lod file_lod{};
                read(file_lod);
                iva.lods.push_back({
                    .gpu_first_vertex_idx = IndexedVerticesAsset::Lod::INDEXED_VERTICES_NOT_LOADED,
                    .gpu_first_index_idx = IndexedVerticesAsset::Lod::INDEXED_VERTICES_NOT_LOADED,
                    .file_vertex_offset = file_lod.vertex_offset,
                    .file_index_offset = file_lod.first_index_offset,
                    .index_count = file_lod.index_count,
                });
            }
            indexed_verts.push_back(std::move(iva));
        }

        return file_id;
    }

    void init_vulkan_resources(VulkanEngine &vk){
        vk_resources.emplace(vk);
    }

    void load_everything_to_gpu() {
        VulkanResources &vkr = *vk_resources;
        const File &file = files[0];
        u64 file_size = [&](){
            i64 raw_size = SDL_GetAsyncIOSize(file.in);
            if (raw_size < 0) {
                std::println("Could not get size of '{}': {}", file.filepath.string(), SDL_GetError());
                abort();
            }
            return raw_size;
        }();

        std::vector<std::byte> buf(file_size);
        {
            SDL_AsyncIOOutcome outcome{};
            SDL_ReadAsyncIO(file.in, buf.data(), 0, file_size, queue, nullptr);
            if (!SDL_WaitAsyncIOResult(queue, &outcome, -1) || outcome.result != SDL_ASYNCIO_COMPLETE) {
                std::println("Read failed for '{}': {}", file.filepath.string(), SDL_GetError());
                abort();
            }
        }

        // Vertices and indices
        for (auto &iv : indexed_verts) {
            // if (iv.file_idx != file_id) continue;
            for (auto &lod : iv.lods) {
                if (lod.gpu_first_vertex_idx != IndexedVerticesAsset::Lod::INDEXED_VERTICES_NOT_LOADED) continue;
                lod.gpu_first_vertex_idx = vkr.vertex_cursor;
                lod.gpu_first_index_idx = vkr.index_cursor;
                vkr.uploader.queue_upload(buf.data() + lod.file_vertex_offset, vkr.vertices,
                    lod.vertex_data_size(), vkr.vertex_cursor * sizeof(Vertex));
                vkr.uploader.queue_upload(buf.data() + lod.file_index_offset, vkr.indices,
                    lod.index_data_size(), vkr.index_cursor * sizeof(u32));
                vkr.vertex_cursor += lod.vertex_count();
                vkr.index_cursor += lod.index_count;
            }
        }
        vkr.uploader.begin_and_finish_uploads();

        // Images
        u32 first_new_texture = vkr.textures.size();
        for (auto &ia : image_assets) {
            // if (ia.file_idx != file_id) continue;
            if (ia.vulkan_image_idx != ImageAsset::VULKAN_IMAGE_NOT_LOADED) continue;
            vkr.textures.emplace_back(vkr.vk, ia.width, ia.height, ia.format, ia.mips_in_file.size());
            vkr.texture_views.emplace_back(vkr.textures.back().make_view(vkr.vk));
            ia.vulkan_image_idx = vkr.textures.size() - 1;
        }
        {
            std::vector<BarrierInfo> barriers;
            for (u32 i = first_new_texture; i < vkr.textures.size(); ++i) {
                barriers.push_back(BarrierInfo{
                    .img = vkr.textures[i],
                    .old_layout_or_undefined_to_discard_current_data = VK_IMAGE_LAYOUT_UNDEFINED,
                    .new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .src_stage_mask = VK_PIPELINE_STAGE_2_NONE,
                    .src_access_mask = 0,
                    .dst_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dst_access_mask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .aspects = ImageAspects::COLOR,
                });
            }
            vkr.submitter.submit(vkr.vk, [&](){ vkr.submitter.cmd_buffer().barrier_span(barriers); });
        }
        for (const auto &ia : image_assets) {
            // if (ia.file_idx != file_id) continue;
            for (u32 mip = 0; mip < ia.mips_in_file.size(); ++mip) {
                const ByteRange &r = ia.mips_in_file[mip];
                vkr.uploader.queue_upload(buf.data() + r.offset,
                    vkr.textures[ia.vulkan_image_idx], r.size, mip, 0, 1, ImageAspects::COLOR);
            }
        }
        vkr.uploader.begin_and_finish_uploads();
        {
            std::vector<BarrierInfo> barriers;
            for (u32 i = first_new_texture; i < vkr.textures.size(); ++i) {
                barriers.push_back(BarrierInfo{
                    .img = vkr.textures[i],
                    .old_layout_or_undefined_to_discard_current_data = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    .src_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .src_access_mask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .dst_stage_mask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    .dst_access_mask = VK_ACCESS_2_SHADER_READ_BIT,
                    .aspects = ImageAspects::COLOR,
                });
            }
            vkr.submitter.submit(vkr.vk, [&](){ vkr.submitter.cmd_buffer().barrier_span(barriers); });
            // todo probably doesn't need to be immediately submitted... but we do need to check that it's done before we let the renderer use them
            // todo also layouts are per-mip, so if we want to do per mip uploads with low mips going first etc we must change how our barrier code works for performance.
        }
        if (!vkr.texture_views.empty()) vkr.graphics_desc_set.update_sampled_images(vkr.vk, 0, vkr.texture_views);

    }

private:
    std::optional<u32> previously_prepared_scene = std::nullopt;
    std::vector<VkDrawIndexedIndirectCommand> draw_commands;
public:
    const std::vector<VkDrawIndexedIndirectCommand> &prepare_to_draw_scene(u32 scene_idx) {
        if (previously_prepared_scene.has_value() && scene_idx == *previously_prepared_scene) return draw_commands;

        previously_prepared_scene = scene_idx;
        draw_commands.clear();
        auto &vkr = *vk_resources;
        const Scene &scene = scenes.at(scene_idx);

        vkr.uploader.queue_upload(scene.transforms, vkr.object_transforms);

        for (u32 i = 0; i < scene.mesh_idx.size(); ++i) {
            const Mesh &mesh = meshes[scene.mesh_idx[i]];
            for (u32 prim_idx : mesh.mesh_prim_idx) {
                const MeshPrimitive &prim = primitives[prim_idx];
                const IndexedVerticesAsset::Lod &lod = indexed_verts[prim.indexed_verts_idx].lods[0];
                u32 draw_idx = draw_commands.size();
                vkr.uploader.queue_upload(i, vkr.obj_transform_indices, draw_idx * sizeof(u32));
                vkr.uploader.queue_upload(image_assets[prim.albedo_idx].vulkan_image_idx, vkr.albedo_texture_indices, draw_idx * sizeof(u32));
                vkr.uploader.queue_upload(image_assets[prim.normal_map_idx].vulkan_image_idx, vkr.normal_map_indices, draw_idx * sizeof(u32));
                vkr.uploader.queue_upload(image_assets[prim.metallic_roughness_idx].vulkan_image_idx, vkr.matallic_roughness_map, draw_idx * sizeof(u32));
                draw_commands.push_back(VkDrawIndexedIndirectCommand{
                    .indexCount = lod.index_count,
                    .instanceCount = 1,
                    .firstIndex = static_cast<u32>(lod.gpu_first_index_idx),
                    .vertexOffset = static_cast<i32>(lod.gpu_first_vertex_idx),
                    .firstInstance = draw_idx,
                });
            }
        }

        vkr.uploader.begin_and_finish_uploads();

        return draw_commands;
    }

};
