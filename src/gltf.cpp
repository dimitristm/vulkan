module;


#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vulkan/vulkan_core.h>
#include <xxhash/xxhash.h>

#if !USE_IMPORT_STD
#include <vector>
#include <filesystem>
#include <print>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <string>
#include <variant>
#endif

module gltf;
#if USE_IMPORT_STD
import std;
#endif

import fastgltf;

namespace{
VkFilter to_vk_filter(fastgltf::Filter f) {
    switch (f) {
        case fastgltf::Filter::Nearest:
        case fastgltf::Filter::NearestMipMapNearest:
        case fastgltf::Filter::NearestMipMapLinear:
            return VK_FILTER_NEAREST;
        default:
            return VK_FILTER_LINEAR;
    }
}

VkSamplerMipmapMode to_vk_mipmap(fastgltf::Filter f) {
    switch (f) {
        case fastgltf::Filter::NearestMipMapNearest:
        case fastgltf::Filter::LinearMipMapNearest:
            return VK_SAMPLER_MIPMAP_MODE_NEAREST;
        default:
            return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    }
}

VkSamplerAddressMode to_vk_wrap(fastgltf::Wrap w) {
    switch (w) {
        case fastgltf::Wrap::ClampToEdge:return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case fastgltf::Wrap::MirroredRepeat: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        default: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}

GltfSampler make_default_sampler() {
    return GltfSampler{
        .mag_filter = VK_FILTER_LINEAR,
        .min_filter = VK_FILTER_LINEAR,
        .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .wrap_s = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .wrap_t = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    };
}

struct LocalDeduplicationMaps{
    // The "local" idx is the index in the gltf file itself, which is why we discard this structure
    // on each new file. Go through this first instead of rehashing every time.
    // Global idx is the idx into the vectors of GltfScenes.
    // Not being in local does not mean it can't already be in global.
    using LocalToGlobalIdx = std::unordered_map<uint32_t, uint32_t>;
    LocalToGlobalIdx gltf_meshes;
    LocalToGlobalIdx gltf_materials;
    LocalToGlobalIdx gltf_textures;
    LocalToGlobalIdx gltf_samplers;
    LocalToGlobalIdx gltf_images;
};
}

void GltfScenes::add_scenes(const std::filesystem::path &filepath){
    LocalDeduplicationMaps local_dedup;
    auto canonical_filepath_str = std::filesystem::canonical(filepath).string();
    fastgltf::Parser parser;
    fastgltf::GltfFileStream file_stream(filepath);
    if (!file_stream.isOpen()) {
        std::println("fastgltf couldn't open file {}", filepath.string());
        abort();
    }
    constexpr auto options =
        fastgltf::Options::LoadExternalBuffers |
        fastgltf::Options::DecomposeNodeMatrices;
    auto expected_asset = parser.loadGltf(file_stream, filepath.parent_path(), options);
    if (expected_asset.error() != fastgltf::Error::None) {
        std::println("fastgltf error while parsing file {} with error message: {}",
            canonical_filepath_str, fastgltf::getErrorMessage(expected_asset.error()));
        abort();
    }
    fastgltf::Asset &asset = expected_asset.get();
    if (auto error = fastgltf::validate(asset); error != fastgltf::Error::None){
        std::println("fastgltf error while validating asset {}, error code {}",
            canonical_filepath_str, fastgltf::getErrorMessage(error));
        abort();
    }
    const auto asset_dir = filepath.parent_path();

    auto get_or_add_sampler = [&](uint32_t local_idx) -> uint32_t {
        if (auto it = local_dedup.gltf_samplers.find(local_idx);
            it != local_dedup.gltf_samplers.end())
            return it->second;

        const auto &s = asset.samplers[local_idx];
        GltfSampler gs{
            .mag_filter  = s.magFilter ? to_vk_filter(*s.magFilter)  : VK_FILTER_LINEAR,
            .min_filter  = s.minFilter ? to_vk_filter(*s.minFilter)  : VK_FILTER_LINEAR,
            .mipmap_mode = s.minFilter ? to_vk_mipmap(*s.minFilter)  : VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .wrap_s      = to_vk_wrap(s.wrapS),
            .wrap_t      = to_vk_wrap(s.wrapT),
        };
        XXH64_hash_t h = XXH64(&gs, sizeof(gs), 0);
        if (auto it = deduplication_maps.gltf_samplers.find(h);
            it != deduplication_maps.gltf_samplers.end()) {
            local_dedup.gltf_samplers[local_idx] = it->second;
            return it->second; }
        auto idx = static_cast<uint32_t>(samplers.size());
        samplers.push_back(gs);
        deduplication_maps.gltf_samplers[h] = idx;
        local_dedup.gltf_samplers[local_idx] = idx;
        return idx;
    };

    auto get_or_add_image = [&](uint32_t local_img_idx) -> uint32_t {
        if (auto it = local_dedup.gltf_images.find(local_img_idx);
            it != local_dedup.gltf_images.end())
            return it->second;

        const auto &fg_img = asset.images[local_img_idx];
        std::vector<std::byte> raw_bytes;
        std::visit(fastgltf::visitor{
            [&](const fastgltf::sources::Array &arr) {
                raw_bytes.resize(arr.bytes.size());
                std::memcpy(raw_bytes.data(), arr.bytes.data(), arr.bytes.size());
            },
            [&](const fastgltf::sources::BufferView &bv_src) {
                const auto &bv  = asset.bufferViews[bv_src.bufferViewIndex];
                const auto &buf = asset.buffers[bv.bufferIndex];
                std::visit(fastgltf::visitor{
                    [&](const fastgltf::sources::Array &arr) {
                        raw_bytes.resize(bv.byteLength);
                        std::memcpy(raw_bytes.data(),
                            arr.bytes.data() + bv.byteOffset, bv.byteLength);
                    },
                    [](auto &&) { std::println("Unsupported buffer source"); abort(); }
                }, buf.data);
            },
            [&](const fastgltf::sources::URI &uri_src) {
                auto img_path = asset_dir / uri_src.uri.fspath();
                std::ifstream f(img_path, std::ios::binary | std::ios::ate);
                if (!f) { std::println("Cannot open {}", img_path.string()); abort(); }
                auto sz = static_cast<std::size_t>(f.tellg());
                f.seekg(0);
                raw_bytes.resize(sz);
                f.read(reinterpret_cast<char *>(raw_bytes.data()), sz);
            },
            [](auto &&) { std::println("Unsupported image source"); abort(); }
        }, fg_img.data);

        XXH64_hash_t h = XXH64(raw_bytes.data(), raw_bytes.size(), 0);
        if (auto it = deduplication_maps.gltf_images.find(h);
            it != deduplication_maps.gltf_images.end()) {
            local_dedup.gltf_images[local_img_idx] = it->second;
            return it->second;
        }

        auto img_data = std::make_unique<std::byte[]>(raw_bytes.size());
        std::memcpy(img_data.get(), raw_bytes.data(), raw_bytes.size());
        auto idx = static_cast<uint32_t>(gltf_images.size());
        gltf_images.push_back(GltfImage{ .image=std::move(img_data), .size=static_cast<uint32_t>(raw_bytes.size()) });
        deduplication_maps.gltf_images[h] = idx;
        local_dedup.gltf_images[local_img_idx] = idx;
        return idx;
    };

    auto get_or_add_texture = [&](uint32_t local_tex_idx) -> uint32_t {
        if (auto it = local_dedup.gltf_textures.find(local_tex_idx);
            it != local_dedup.gltf_textures.end())
            return it->second;

        const auto &fg_tex = asset.textures[local_tex_idx];

        uint32_t sampler_idx = fg_tex.samplerIndex.has_value()
            ? get_or_add_sampler(static_cast<uint32_t>(*fg_tex.samplerIndex))
            : [&]() -> uint32_t {
                GltfSampler def = make_default_sampler();
                XXH64_hash_t h  = XXH64(&def, sizeof(def), 0);
                if (auto it = deduplication_maps.gltf_samplers.find(h);
                    it != deduplication_maps.gltf_samplers.end())
                    return it->second;
                auto idx = static_cast<uint32_t>(samplers.size());
                samplers.push_back(def);
                deduplication_maps.gltf_samplers[h] = idx;
                return idx;
            }();

        uint32_t image_idx = get_or_add_image(
            static_cast<uint32_t>(fg_tex.imageIndex.value()));

        GltfTexture gt{ .image_idx=image_idx, .sampler_idx=sampler_idx };
        XXH64_hash_t tex_h = XXH64(&gt, sizeof(gt), 0);

        if (auto it = deduplication_maps.gltf_textures.find(tex_h);
            it != deduplication_maps.gltf_textures.end()) {
            local_dedup.gltf_textures[local_tex_idx] = it->second;
            return it->second;
        }

        auto idx = static_cast<uint32_t>(textures.size());
        textures.push_back(gt);
        deduplication_maps.gltf_textures[tex_h] = idx;
        local_dedup.gltf_textures[local_tex_idx] = idx;
        return idx;
    };

    auto get_or_add_material = [&](uint32_t local_mat_idx) -> uint32_t {
        if (auto it = local_dedup.gltf_materials.find(local_mat_idx);
            it != local_dedup.gltf_materials.end())
            return it->second;

        const auto &fm = asset.materials[local_mat_idx];
        GltfMaterial mat{};

        if (fm.pbrData.baseColorTexture.has_value())
            mat.base_color_tex_idx =
                get_or_add_texture(static_cast<uint32_t>(fm.pbrData.baseColorTexture->textureIndex));

        if (fm.normalTexture.has_value())
            mat.normal_tex_idx =
                get_or_add_texture(static_cast<uint32_t>(fm.normalTexture->textureIndex));

        if (fm.pbrData.metallicRoughnessTexture.has_value())
            mat.metallic_roughness_tex_idx =
                get_or_add_texture(static_cast<uint32_t>(fm.pbrData.metallicRoughnessTexture->textureIndex));

        const auto &bc = fm.pbrData.baseColorFactor;
        mat.base_color_factor = { bc[0], bc[1], bc[2], bc[3] };
        mat.metallic_factor   = fm.pbrData.metallicFactor;
        mat.roughness_factor  = fm.pbrData.roughnessFactor;

        XXH64_hash_t h = XXH64(&mat, sizeof(mat), 0);
        if (auto it = deduplication_maps.gltf_materials.find(h);
            it != deduplication_maps.gltf_materials.end()) {
            local_dedup.gltf_materials[local_mat_idx] = it->second;
            return it->second;
        }

        auto idx = static_cast<uint32_t>(materials.size());
        materials.push_back(mat);
        deduplication_maps.gltf_materials[h] = idx;
        local_dedup.gltf_materials[local_mat_idx] = idx;
        return idx;
    };

    auto get_or_add_vertices = [&](const fastgltf::Primitive &prim) -> uint32_t {
        const auto *pos_it  = prim.findAttribute("POSITION");
        const auto *norm_it = prim.findAttribute("NORMAL");
        const auto *uv_it   = prim.findAttribute("TEXCOORD_0");
        const auto *col_it  = prim.findAttribute("COLOR_0");

        std::vector<Vertex> verts(asset.accessors[pos_it->accessorIndex].count);

        if (pos_it != prim.attributes.end()) {
            fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                asset, asset.accessors[pos_it->accessorIndex],
                [&](fastgltf::math::fvec3 v, std::size_t i) {
                    verts[i].pos = { v.x(), v.y(), v.z() };
                });
        }
        if (norm_it != prim.attributes.end()) {
            fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                asset, asset.accessors[norm_it->accessorIndex],
                [&](fastgltf::math::fvec3 n, std::size_t i) {
                    verts[i].normal = { n.x(), n.y(), n.z() };
                });
        }
        if (uv_it != prim.attributes.end()) {
            fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(
                asset, asset.accessors[uv_it->accessorIndex],
                [&](fastgltf::math::fvec2 uv, std::size_t i) {
                    verts[i].u = uv.x();
                    verts[i].v = uv.y();
                });
        }
        if (col_it != prim.attributes.end()) {
            fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(
                asset, asset.accessors[col_it->accessorIndex],
                [&](fastgltf::math::fvec4 c, std::size_t i) {
                    verts[i].color = { c.x(), c.y(), c.z(), c.w() };
                });
        } else {
            for (auto &vert : verts) vert.color = { 1.f, 1.f, 1.f, 1.f };
        }

        XXH64_hash_t h = XXH64(verts.data(), verts.size() * sizeof(Vertex), 0);
        if (auto it = deduplication_maps.vertices.find(h);
            it != deduplication_maps.vertices.end())
            return it->second;

        auto idx = static_cast<uint32_t>(vertices.size());
        vertices.push_back(std::move(verts));
        deduplication_maps.vertices[h] = idx;
        return idx;
    };

    auto get_or_add_indices = [&](const fastgltf::Primitive &prim) -> uint32_t {
        std::vector<uint32_t> idx_buf;
        fastgltf::iterateAccessorWithIndex<uint32_t>(
            asset, asset.accessors[prim.indicesAccessor.value()],
            [&](uint32_t v, std::size_t) { idx_buf.push_back(v); });

        XXH64_hash_t h = XXH64(idx_buf.data(), idx_buf.size() * sizeof(uint32_t), 0);
        if (auto it = deduplication_maps.indices.find(h);
            it != deduplication_maps.indices.end())
            return it->second;

        auto idx = static_cast<uint32_t>(indices.size());
        indices.push_back(std::move(idx_buf));
        deduplication_maps.indices[h] = idx;
        return idx;
    };

    auto get_or_add_mesh_primitive = [&](const fastgltf::Primitive &prim) -> uint32_t {
        GltfMeshPrimitive mp{};
        mp.vertices_idx = get_or_add_vertices(prim);
        mp.indices_idx  = get_or_add_indices(prim);
        mp.material_idx = prim.materialIndex.has_value()
            ? get_or_add_material(static_cast<uint32_t>(*prim.materialIndex))
            : UINT32_MAX;

        XXH64_hash_t h = XXH64(&mp, sizeof(mp), 0);
        if (auto it = deduplication_maps.gltf_mesh_primitives.find(h);
            it != deduplication_maps.gltf_mesh_primitives.end())
            return it->second;

        auto idx = static_cast<uint32_t>(mesh_primitives.size());
        mesh_primitives.push_back(mp);
        deduplication_maps.gltf_mesh_primitives[h] = idx;
        return idx;
    };

    auto make_mesh = [&](uint32_t local_mesh_idx,
                         const glm::mat4 &world_transform) -> uint32_t {
        const auto &fg_mesh = asset.meshes[local_mesh_idx];
        GltfMesh gm{};
        gm.transform = world_transform;
        gm.name      = std::string(fg_mesh.name);

        for (const auto &prim : fg_mesh.primitives)
            gm.mesh_prim_idx.push_back(get_or_add_mesh_primitive(prim));

        XXH64_state_t *state = XXH64_createState();
        XXH64_reset(state, 0);
        XXH64_update(state, &gm.transform, sizeof(gm.transform));
        XXH64_update(state, gm.mesh_prim_idx.data(),
                     gm.mesh_prim_idx.size() * sizeof(uint32_t));
        XXH64_hash_t h = XXH64_digest(state);
        XXH64_freeState(state);

        if (auto it = deduplication_maps.gltf_meshes.find(h);
            it != deduplication_maps.gltf_meshes.end())
            return it->second;

        auto idx = static_cast<uint32_t>(meshes.size());
        meshes.push_back(std::move(gm));
        deduplication_maps.gltf_meshes[h] = idx;
        return idx;
    };

    auto walk_nodes = [&](const fastgltf::Scene &scene) -> GltfScene {
        GltfScene scene_info{};

        struct StackEntry { std::size_t node_idx; glm::mat4 parent_transform; };
        std::vector<StackEntry> stack;
        stack.reserve(64);

        for (auto root_idx : scene.nodeIndices)
            stack.push_back({ .node_idx=root_idx, .parent_transform=glm::mat4(1.f) });

        while (!stack.empty()) {
            auto [node_idx, parent_xform] = stack.back();
            stack.pop_back();

            const auto &node = asset.nodes[node_idx];

            // DecomposeNodeMatrices option guarantees TRS variant
            glm::mat4 local_xform(1.f);
            if (const auto *trs = std::get_if<fastgltf::TRS>(&node.transform)) {
                glm::mat4 T = glm::translate(glm::mat4(1.f),
                    glm::vec3(trs->translation[0], trs->translation[1], trs->translation[2]));
                glm::quat q(trs->rotation[3], trs->rotation[0],
                            trs->rotation[1], trs->rotation[2]);
                glm::mat4 R = glm::mat4_cast(q);
                glm::mat4 S = glm::scale(glm::mat4(1.f),
                    glm::vec3(trs->scale[0], trs->scale[1], trs->scale[2]));
                local_xform = T * R * S;
            }
            glm::mat4 world_xform = parent_xform * local_xform;

            if (node.meshIndex.has_value()) {
                uint32_t mesh_idx = make_mesh(
                    static_cast<uint32_t>(*node.meshIndex), world_xform);
                scene_info.mesh_idx.push_back(mesh_idx);
            }

            for (auto child_idx : node.children)
                stack.push_back({ .node_idx=child_idx, .parent_transform=world_xform });
        }
        return scene_info;
    };

    for (const auto &scene : asset.scenes)
        scene_infos.push_back(walk_nodes(scene));
}
