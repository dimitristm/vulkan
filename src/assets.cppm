module;

#include <stb/stb_image.h>
#include <vulkan/vulkan_core.h>

#if !USE_IMPORT_STD
#include <print>
#include <filesystem>
#include <vector>
#endif

export module assets;
#if USE_IMPORT_STD
import std;
#endif

import gltf;
import vulkanEngine;
import vulkanUtil;
import types;

struct Scene{
    std::vector<u32> mesh_idx;
};

struct Material {
    fvec4 base_color_factor = {1.f, 1.f, 1.f, 1.f};
    f32 metallic_factor = 1.f;
    f32 roughness_factor = 1.f;
};

struct Mesh{
    u32 unique_transform_idx;
    std::vector<u32> mesh_prim_idx;
};

struct MeshPrimitive{
    u32 vertex_offset;
    u32 first_index;
    u32 index_count;
    u32 albedo_texture_idx;
    u32 material_idx;
};

export struct Scenes{
    std::vector<Scene> scenes;
    std::vector<Mesh> meshes;
    std::vector<MeshPrimitive> primitives;
    std::vector<Material> materials;

    std::vector<Texture> textures;
    std::vector<ImageView> texture_views;
    StorageBuffer obj_transforms;
    VertexBuffer<Vertex> vertices;
    IndexBuffer indices;
    StorageBuffer albedo_texture_indices;
    StorageBuffer obj_transform_indices;

    static Scenes make_from_glb(
        VulkanEngine &vk, 
        HostToDeviceUploader &uploader,
        ImmediateSubmitter &submitter,
        const std::initializer_list<std::filesystem::path> glb_files)
    {
        GltfScenes gltf_scenes(glb_files);
        using VertexType = typename std::remove_cvref_t<decltype(gltf_scenes.vertices[0])>::value_type;
        static_assert(std::is_same_v<VertexType, Vertex>);
        VertexBuffer<Vertex> vertices(vk, gltf_scenes.get_vertex_count());
        IndexBuffer indices(vk, gltf_scenes.get_index_count());
        std::vector<Texture> textures;
        textures.reserve(gltf_scenes.textures.size());
        std::vector<ImageView> texture_views;
        texture_views.reserve(gltf_scenes.textures.size());
        u32 max_instance_count = [&](){
            u32 sum = 0;
            for (const auto &mesh : gltf_scenes.meshes) sum += mesh.mesh_prim_idx.size();
            return sum;
        }();
        StorageBuffer albedo_texture_indices(vk, max_instance_count * sizeof(u32), false, true);
        StorageBuffer obj_transform_indices(vk, max_instance_count * sizeof(u32), false, true);
        StorageBuffer obj_transforms(vk, gltf_scenes.meshes.size() * sizeof(fmat4), false, true);

        std::vector<Scene> scenes;
        scenes.resize(gltf_scenes.scene_infos.size());
        std::vector<Mesh> meshes;
        meshes.resize(gltf_scenes.meshes.size());
        std::vector<MeshPrimitive> primitives;
        primitives.resize(gltf_scenes.mesh_primitives.size());
        std::vector<Material> materials;
        materials.resize(gltf_scenes.materials.size());

        for (i32 i = 0; auto &gltf_scene: gltf_scenes.scene_infos){
            scenes[i++].mesh_idx = std::move(gltf_scene.mesh_idx);
        }
        uploader.start_queue_upload(obj_transforms);
        for (i32 i = 0; auto &gltf_mesh : gltf_scenes.meshes){
            uploader.add_to_last_upload(gltf_mesh.transform);
            meshes[i].mesh_prim_idx = std::move(gltf_mesh.mesh_prim_idx);
            meshes[i].unique_transform_idx = i;
            ++i;
        }
        for (i32 i = 0; auto &gltf_primitive : gltf_scenes.mesh_primitives){
            primitives[i].material_idx = gltf_primitive.material_idx;
            const auto &material = gltf_scenes.materials[gltf_primitive.material_idx];
            primitives[i].albedo_texture_idx = material.base_color_tex_idx;
            primitives[i].index_count = gltf_scenes.indices[gltf_primitive.indices_idx].size();
            primitives[i].first_index = [&](){
                u32 sum = 0;
                for (u32 j = 0; j < gltf_primitive.indices_idx; ++j) sum += gltf_scenes.indices[j].size();
                return sum;
            }();
            primitives[i].vertex_offset = [&](){
                u32 sum = 0;
                for (u32 j = 0; j < gltf_primitive.vertices_idx; ++j) sum += gltf_scenes.vertices[j].size();
                return sum;
            }();
            ++i;
        }
        for (i32 i = 0; auto &gltf_material : gltf_scenes.materials){
            materials[i].base_color_factor = gltf_material.base_color_factor;
            materials[i].metallic_factor = gltf_material.metallic_factor;
            materials[i].roughness_factor = gltf_material.roughness_factor;
        }
        uploader.start_queue_upload(vertices);
        for (auto &gltf_vertices : gltf_scenes.vertices){
            uploader.add_to_last_upload(gltf_vertices);
        }
        uploader.start_queue_upload(indices);
        for (auto &gltf_indices : gltf_scenes.indices){
            uploader.add_to_last_upload(gltf_indices);
        }
        for(auto &gltf_texture : gltf_scenes.textures){
            const auto &img = gltf_scenes.gltf_images[gltf_texture.image_idx];
            i32 x, y, channels;
            if(!stbi_info_from_memory((stbi_uc*)img.image.get(), img.size, &x, &y, &channels)){
                std::println("stbi: Error getting info from an image");
                abort();
            }
            textures.emplace_back(vk, x, y, 1);
            texture_views.emplace_back(textures.back().make_view(vk));
        }
        {
            std::vector<BarrierInfo> barriers;
            barriers.reserve(textures.size());
            for (auto &texture : textures){
                barriers.push_back( BarrierInfo{
                    .img = texture, .old_layout_or_undefined_to_discard_current_data=VK_IMAGE_LAYOUT_UNDEFINED, .new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .src_stage_mask = VK_PIPELINE_STAGE_2_NONE,
                    .src_access_mask = 0,
                    .dst_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dst_access_mask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .aspects = ImageAspects::COLOR
                });
            }
            submitter.submit(vk,[&](){
                submitter.cmd_buffer().barrier_span(barriers);
            });
        }
        for (i32 i = 0; auto &gltf_texture : gltf_scenes.textures){
            const auto &img = gltf_scenes.gltf_images[gltf_texture.image_idx];
            i32 x, y, channels;
            stbi_uc *img_loaded = stbi_load_from_memory((stbi_uc*) img.image.get(), img.size, &x, &y, &channels, 4);
            uploader.queue_upload(img_loaded, textures[i], x*y*4, 0, 0, 1, ImageAspects::COLOR);
            stbi_image_free(img_loaded);
            ++i;
        }
        uploader.begin_and_finish_uploads();
        {
            std::vector<BarrierInfo> barriers;
            barriers.reserve(textures.size());
            for (auto &texture : textures){
                barriers.push_back(BarrierInfo{
                    .img = texture, .old_layout_or_undefined_to_discard_current_data=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    .src_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .src_access_mask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .dst_stage_mask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    .dst_access_mask = VK_ACCESS_2_SHADER_READ_BIT,
                    .aspects = ImageAspects::COLOR
                });
            }
            submitter.submit(vk,[&](){
                submitter.cmd_buffer().barrier_span(barriers);
            });
        }

        return {
            .scenes = std::move(scenes),
            .meshes = std::move(meshes),
            .primitives = std::move(primitives),
            .materials = std::move(materials),
            .textures = std::move(textures),
            .texture_views = std::move(texture_views),
            .obj_transforms = obj_transforms,
            .vertices = vertices,
            .indices = indices,
            .albedo_texture_indices = albedo_texture_indices,
            .obj_transform_indices = obj_transform_indices,
        };
    }

    std::vector<VkDrawIndexedIndirectCommand> prepare_to_draw_scene(u32 scene_idx, HostToDeviceUploader &uploader){
        std::vector<VkDrawIndexedIndirectCommand> commands;
        const auto &scene = scenes.at(scene_idx);
        std::vector<u32> indices_to_upload;
        uploader.start_queue_upload(obj_transform_indices);
        for (const u32 mesh_idx : scene.mesh_idx){
            const auto &mesh = meshes[mesh_idx];
            indices_to_upload.resize(mesh.mesh_prim_idx.size());
            for (auto &idx : indices_to_upload) idx = mesh.unique_transform_idx;
            uploader.add_to_last_upload(indices_to_upload);
        }
        uploader.start_queue_upload(albedo_texture_indices);
        for (const u32 mesh_idx : scene.mesh_idx){
            const auto &mesh = meshes[mesh_idx];
            for (const u32 &prim_idx : mesh.mesh_prim_idx){
                const auto &prim = primitives[prim_idx];
                uploader.add_to_last_upload(prim.albedo_texture_idx);
                commands.push_back(VkDrawIndexedIndirectCommand{
                    .indexCount = prim.index_count,
                    .instanceCount = 1,
                    .firstIndex = prim.first_index,
                    .vertexOffset = static_cast<i32>(prim.vertex_offset),
                    .firstInstance = static_cast<u32>(commands.size()),
                });
            }
        }
        uploader.begin_and_finish_uploads();
        return commands;
    }

};
