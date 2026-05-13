module;

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3/SDL_events.h>

#include <VkBootstrap.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/packing.hpp>
#include <glm/trigonometric.hpp>
#include <vulkan/vk_enum_string_helper.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl3.h"
#include "imgui/imgui_impl_vulkan.h"
#include <stb/stb_image.h>

#if !USE_IMPORT_STD
#include <filesystem>
#include <array>
#include <cmath>
#include <print>
#include <functional>
#endif

export module vulkanRenderer;
#if USE_IMPORT_STD
import std;
#endif

import vulkanEngine;
import userInput;
import gltf;
import vulkanUtil;

static glm::ivec2 get_window_size_in_pixels(SDL_Window *window){
    glm::ivec2 size;
    SDL_GetWindowSizeInPixels(window, &size.x, &size.y);
    return size;
}

//Assumes a right handed coordinate system. Output Z ranges from 0 to 1, with close objects at 1.
static glm::mat4 perspective_projection(float horizontal_fov_in_degrees, float horizontal_to_vertical_ratio, float near_z, float far_z){
    // todo numerical precision is there a reason to pass vertical_to_horizontal_ratio or horizontal to vertical? pick whichever has better precision or if it doesn't matter do h/v
    const float half_fov = glm::radians(horizontal_fov_in_degrees)/2;
    float tan = glm::tan(half_fov);// will it help the compiler if i calculate 1/tan once and then multiply it instead of divide it twice?

    // remember glm is column major so the actual matrix will be the transpose of what this notation would suggest
    return glm::mat4{
        1/tan,0,0,0,
        0,-horizontal_to_vertical_ratio/tan,0,0,
        0,0,-near_z/(near_z-far_z),-1,
        0,0,(far_z*near_z)/(far_z-near_z),0,
    };
};

namespace{
struct Texture : public Image{
    Texture(VulkanEngine &vk, uint32_t width, uint32_t height, uint32_t mip_level_count)
    :Image(vk,
           {.width=width,.height=height},
           VK_FORMAT_R8G8B8A8_UNORM,
           VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
           MSAALevel::OFF,
           mip_level_count,
           1){}
    ImageView make_view(VulkanEngine &vk)
    {
        return  {vk, *this, ImageAspects::COLOR, 0, VK_REMAINING_MIP_LEVELS};
    }
};

struct DrawImage : public Image{
    ImageView view;
    DrawImage(VulkanEngine &vk, glm::ivec2 window_size)
    :Image(vk,
           {.width=static_cast<uint32_t>(window_size.x), .height=static_cast<uint32_t>(window_size.y)},
           VK_FORMAT_R16G16B16A16_SFLOAT,
           VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
           | VK_IMAGE_USAGE_STORAGE_BIT
           | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
           MSAALevel::OFF,
           1, 1),
    view(vk, *this, ImageAspects::COLOR, 0, 1)
    {}
};

struct DepthImage : public Image{
    ImageView view;
    DepthImage(VulkanEngine &vk, glm::ivec2 window_size)
    :Image(vk,
           {.width=static_cast<uint32_t>(window_size.x), .height=static_cast<uint32_t>(window_size.y)},
           VK_FORMAT_D32_SFLOAT,
           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
           MSAALevel::OFF,
           1, 1),
    view(vk, *this, ImageAspects::DEPTH, 0, 1)
    {}
};

struct FrameInFlightData{
    GpuSemaphore swapchain_img_ready_sema;
    CommandBuffer main_cmd_buffer;
    GpuFence rendering_done_fence;

    FrameInFlightData(VulkanEngine &vk, CommandPool cmd_pool)
    :
    swapchain_img_ready_sema(vk),
    main_cmd_buffer(vk, cmd_pool),
    rendering_done_fence(vk, true)
    {}
};

struct Scene{
    std::vector<uint32_t> mesh_idx;
};

struct Material {
    glm::vec4 base_color_factor = {1.f, 1.f, 1.f, 1.f};
    float metallic_factor = 1.f;
    float roughness_factor = 1.f;
};

struct Mesh{
    uint32_t unique_transform_idx;
    std::vector<uint32_t> mesh_prim_idx;
};

struct MeshPrimitive{
    uint32_t vertex_offset;
    uint32_t first_index;
    uint32_t index_count;
    uint32_t albedo_texture_idx;
    uint32_t material_idx;
};

struct Scenes{
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
        uint32_t max_instance_count = [&](){
            uint32_t sum = 0;
            for (const auto &mesh : gltf_scenes.meshes) sum += mesh.mesh_prim_idx.size();
            return sum;
        }();
        StorageBuffer albedo_texture_indices(vk, max_instance_count * sizeof(uint32_t), false, true);
        StorageBuffer obj_transform_indices(vk, max_instance_count * sizeof(uint32_t), false, true);
        StorageBuffer obj_transforms(vk, gltf_scenes.meshes.size() * sizeof(glm::mat4), false, true);

        std::vector<Scene> scenes;
        scenes.resize(gltf_scenes.scene_infos.size());
        std::vector<Mesh> meshes;
        meshes.resize(gltf_scenes.meshes.size());
        std::vector<MeshPrimitive> primitives;
        primitives.resize(gltf_scenes.mesh_primitives.size());
        std::vector<Material> materials;
        materials.resize(gltf_scenes.materials.size());

        for (int i = 0; auto &gltf_scene: gltf_scenes.scene_infos){
            scenes[i++].mesh_idx = std::move(gltf_scene.mesh_idx);
        }
        uploader.start_queue_upload(obj_transforms);
        for (int i = 0; auto &gltf_mesh : gltf_scenes.meshes){
            uploader.add_to_last_upload(gltf_mesh.transform);
            meshes[i].mesh_prim_idx = std::move(gltf_mesh.mesh_prim_idx);
            meshes[i].unique_transform_idx = i;
            ++i;
        }
        for (int i = 0; auto &gltf_primitive : gltf_scenes.mesh_primitives){
            primitives[i].material_idx = gltf_primitive.material_idx;
            const auto &material = gltf_scenes.materials[gltf_primitive.material_idx];
            primitives[i].albedo_texture_idx = material.base_color_tex_idx;
            primitives[i].index_count = gltf_scenes.indices[gltf_primitive.indices_idx].size();
            primitives[i].first_index = [&](){
                uint32_t sum = 0;
                for (uint32_t j = 0; j < gltf_primitive.indices_idx; ++j) sum += gltf_scenes.indices[j].size();
                return sum;
            }();
            primitives[i].vertex_offset = [&](){
                uint32_t sum = 0;
                for (uint32_t j = 0; j < gltf_primitive.vertices_idx; ++j) sum += gltf_scenes.vertices[j].size();
                return sum;
            }();
            ++i;
        }
        for (int i = 0; auto &gltf_material : gltf_scenes.materials){
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
            using T = typename std::remove_cvref_t<decltype(gltf_indices)>::value_type;
            uploader.add_to_last_upload(gltf_indices);
        }
        for(auto &gltf_texture : gltf_scenes.textures){
            const auto &img = gltf_scenes.gltf_images[gltf_texture.image_idx];
            int x, y, channels;
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
        for (int i = 0; auto &gltf_texture : gltf_scenes.textures){
            const auto &img = gltf_scenes.gltf_images[gltf_texture.image_idx];
            int x, y, channels;
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

    std::vector<VkDrawIndexedIndirectCommand> prepare_to_draw_scene(uint32_t scene_idx, HostToDeviceUploader &uploader){
        std::vector<VkDrawIndexedIndirectCommand> commands;
        const auto &scene = scenes.at(scene_idx);
        std::vector<uint32_t> indices_to_upload;
        uploader.start_queue_upload(obj_transform_indices);
        for (const uint32_t mesh_idx : scene.mesh_idx){
            const auto &mesh = meshes[mesh_idx];
            indices_to_upload.resize(mesh.mesh_prim_idx.size());
            for (auto &idx : indices_to_upload) idx = mesh.unique_transform_idx;
            uploader.add_to_last_upload(indices_to_upload);
        }
        uploader.start_queue_upload(albedo_texture_indices);
        for (const uint32_t mesh_idx : scene.mesh_idx){
            const auto &mesh = meshes[mesh_idx];
            for (const uint32_t &prim_idx : mesh.mesh_prim_idx){
                const auto &prim = primitives[prim_idx];
                uploader.add_to_last_upload(prim.albedo_texture_idx);
                commands.push_back(VkDrawIndexedIndirectCommand{
                    .indexCount = prim.index_count,
                    .instanceCount = 1,
                    .firstIndex = prim.first_index,
                    .vertexOffset = static_cast<int32_t>(prim.vertex_offset),
                    .firstInstance = static_cast<uint32_t>(commands.size()),
                });
            }
        }
        uploader.begin_and_finish_uploads();
        return commands;
    }

};
}

export class Renderer{
    static constexpr int FRAMES_IN_FLIGHT = 2;

    SDL_Window *window;
    glm::ivec2 window_size;
    VulkanEngine vk;
    CommandPool command_pool; // There must be one command pool for each thread and each thread only touches cmd buffers from its pool
    ImmediateSubmitter immediate_submiter;
    HostToDeviceUploader uploader;

    Swapchain swapchain;
    DrawImage draw_image;
    DepthImage depth_image;
    std::vector<GpuSemaphore> swapchain_render_done_semas;
    std::array<FrameInFlightData, FRAMES_IN_FLIGHT> frame_in_flight_data;
    uint32_t frame_count{};

    SpecializationInfo specialization_info;
    PushConstantsBuilder pc_builder;
    DescriptorSetBuilder ds_builder;

    DescriptorSet compute_desc_set;
    PushConstant<glm::vec4> compute_push_const;
    PipelineLayout compute_pipeline_layout;
    ComputePipeline compute_pipeline;

    Scenes scenes;
    std::vector<VkDrawIndexedIndirectCommand> draw_commands;
    PushConstant<glm::mat4> view_proj_transform_const;
    DescriptorSet graphics_desc_set;
    GraphicsPipeline<Vertex> graphics_pipeline;

    Sampler sampler;

public:
    Renderer(SDL_Window *window)
    :
    window(window),
    window_size(get_window_size_in_pixels(window)),
    vk(window),
    command_pool(vk),
    immediate_submiter(vk, command_pool),
    uploader(&vk, command_pool, 64 * 1024 * 1024),// todo performance 64MiB is small, raise it if we do more later
    swapchain(vk, window, VK_PRESENT_MODE_FIFO_KHR),
    draw_image(vk, window_size),
    depth_image(vk, window_size),
    swapchain_render_done_semas(
        [&] -> std::vector<GpuSemaphore> {
            std::vector<GpuSemaphore> swapchain_render_done_semas;
            int sema_count = (int)swapchain.get_images().size();
            swapchain_render_done_semas.reserve(sema_count);
            for (int i = 0; i <sema_count; ++i){
                swapchain_render_done_semas.emplace_back(vk);
            }
            return swapchain_render_done_semas;
        }()
    ),
    frame_in_flight_data({FrameInFlightData(vk, command_pool), FrameInFlightData(vk, command_pool)}),
    specialization_info((2 * sizeof(int32_t)) + sizeof(double)),
    pc_builder(),
    ds_builder(),
    compute_desc_set(ds_builder.bind(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).build(vk)),
    compute_push_const(pc_builder.add<glm::vec4>(VK_SHADER_STAGE_COMPUTE_BIT)),
    compute_pipeline_layout(vk, compute_desc_set, pc_builder.get_ranges()),
    compute_pipeline(vk, ComputeShader(vk, "shaders/compiled/gradient.comp.spv"), compute_pipeline_layout, &specialization_info.reset().add_entry(0, 16).add_entry(1, 32).add_entry(1234, 34.0)),
    scenes(Scenes::make_from_glb(vk, uploader, immediate_submiter, {"assets/mytests/1.glb", "assets/mytests/1.glb"})),
    draw_commands(scenes.prepare_to_draw_scene(1, uploader)),
    view_proj_transform_const(pc_builder.reset().add<glm::mat4>( VK_SHADER_STAGE_VERTEX_BIT)),
    graphics_desc_set(ds_builder.reset()
                      .bind(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000, VK_SHADER_STAGE_FRAGMENT_BIT)
                      .bind(1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT)
                      .bind(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT)
                      .bind(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT)
                      .bind(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT)
                      .build(vk)
    ),
    graphics_pipeline(vk,
                      VertexShader(vk, "shaders/compiled/colored_triangle_mesh.vert.spv"),
                      FragmentShader(vk, "shaders/compiled/textured.frag.spv"),
                      PipelineLayout(vk, graphics_desc_set, pc_builder.get_ranges()),
                      scenes.vertices,
                      draw_image.get_format(),
                      depth_image.get_format(),
                      MSAALevel::OFF),
    sampler(vk, VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_FALSE, 0)
    {
        vk.init_imgui(window, swapchain.get_format());
        compute_desc_set.update_storage_image(vk, 0, draw_image.view);
        compute_push_const.data += glm::vec4(1, 0, 0, 1);

        graphics_desc_set.update_sampled_images(vk, 0, scenes.texture_views);
        graphics_desc_set.update_sampler(vk, 1, sampler);
        graphics_desc_set.update_storage_buffer(vk, 2, scenes.obj_transform_indices);
        graphics_desc_set.update_storage_buffer(vk, 3, scenes.albedo_texture_indices);
        graphics_desc_set.update_storage_buffer(vk, 4, scenes.obj_transforms);

        immediate_submiter.submit(vk, [&]{
            immediate_submiter.cmd_buffer().barrier(
                BarrierInfo{
                    .img = depth_image, .old_layout_or_undefined_to_discard_current_data=VK_IMAGE_LAYOUT_UNDEFINED, .new_layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    .src_stage_mask = VK_PIPELINE_STAGE_2_NONE,
                    .src_access_mask = 0,
                    .dst_stage_mask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                    .dst_access_mask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                    .aspects = ImageAspects::DEPTH
                }
            );
        });
    }

    FrameInFlightData &get_current_frame_in_flight(){ return frame_in_flight_data[frame_count % FRAMES_IN_FLIGHT]; }

    void draw(glm::mat4 view_transform){
	ImGui::Begin("Background customizer");
        ImGui::InputFloat4("data a",(float*)& compute_push_const.data.a);
        ImGui::End();


        FrameInFlightData &frame_in_flight = get_current_frame_in_flight();

        frame_in_flight.rendering_done_fence.wait(vk);
        uint32_t swapchain_img_idx = swapchain.acquire_next_image(vk, frame_in_flight.swapchain_img_ready_sema);

        CommandBuffer &cmd_buffer = frame_in_flight.main_cmd_buffer;
        cmd_buffer.restart(true);
        cmd_buffer.barrier(BarrierInfo{
                                .img=draw_image,
                                .old_layout_or_undefined_to_discard_current_data=VK_IMAGE_LAYOUT_UNDEFINED,
                                .new_layout=VK_IMAGE_LAYOUT_GENERAL,
                                .src_stage_mask=VK_PIPELINE_STAGE_2_BLIT_BIT,
                                .src_access_mask=0,
                                .dst_stage_mask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                .dst_access_mask=0, // todo: do i really not need to make visible this transition to the compute stage?
                                .aspects=ImageAspects::COLOR}
        );

        cmd_buffer.update_push_constants(compute_pipeline, compute_push_const);

        cmd_buffer.bind_pipeline(compute_pipeline);
        cmd_buffer.bind_descriptor_set(compute_pipeline, compute_desc_set);
        cmd_buffer.dispatch(std::ceil(window_size.x / 16.0), std::ceil(window_size.y / 16.0), 1);

        cmd_buffer.barrier(BarrierInfo{
                               .img=draw_image,
                               .old_layout_or_undefined_to_discard_current_data=VK_IMAGE_LAYOUT_GENERAL,
                               .new_layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                               .src_stage_mask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               .src_access_mask=VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                               .dst_stage_mask=VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                               .dst_access_mask=VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                               .aspects=ImageAspects::COLOR}
        );

        cmd_buffer.bind_pipeline(graphics_pipeline);
        cmd_buffer.bind_descriptor_set(graphics_pipeline, graphics_desc_set);
        view_proj_transform_const.data = perspective_projection(80.0f, (float)window_size.x / (float)window_size.y, 0.001f, 10000.0f) * view_transform;
        cmd_buffer.update_push_constants(graphics_pipeline, view_proj_transform_const);
        cmd_buffer.draw_indexed(draw_image.view, depth_image.view, draw_image.extent,  scenes.vertices, scenes.indices, draw_commands);

        cmd_buffer.barrier(BarrierInfo{
                                .img=draw_image,
                                .old_layout_or_undefined_to_discard_current_data=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                .new_layout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                .src_stage_mask=VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .src_access_mask=VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                .dst_stage_mask=VK_PIPELINE_STAGE_2_BLIT_BIT,
                                .dst_access_mask=VK_ACCESS_2_TRANSFER_READ_BIT,
                                .aspects=ImageAspects::COLOR
                           },
                           BarrierInfo{
                                .img=swapchain.get_images()[swapchain_img_idx],
                                .old_layout_or_undefined_to_discard_current_data=VK_IMAGE_LAYOUT_UNDEFINED,
                                .new_layout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                .src_stage_mask=VK_PIPELINE_STAGE_2_BLIT_BIT, // Here to prevent the image transition write from happening before acquire_next_image can read the image. should i make a seperate execution barrier?
                                .src_access_mask=0,
                                .dst_stage_mask=VK_PIPELINE_STAGE_2_BLIT_BIT,
                                .dst_access_mask=VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .aspects=ImageAspects::COLOR
                           }
        );

        cmd_buffer.blit_entire_images(draw_image,
                                      swapchain.get_images()[swapchain_img_idx],
                                      ImageAspects::COLOR
        );

        cmd_buffer.barrier(BarrierInfo{
                                .img=swapchain.get_images()[swapchain_img_idx],
                                .old_layout_or_undefined_to_discard_current_data=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                .new_layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                .src_stage_mask=VK_PIPELINE_STAGE_2_BLIT_BIT,
                                .src_access_mask=VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .dst_stage_mask=VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .dst_access_mask=VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                                .aspects=ImageAspects::COLOR}
        );

        cmd_buffer.draw_imgui(swapchain.get_image_views()[swapchain_img_idx], swapchain.get_extent());

        cmd_buffer.barrier(BarrierInfo{
                                .img=swapchain.get_images()[swapchain_img_idx],
                                .old_layout_or_undefined_to_discard_current_data=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                .new_layout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                .src_stage_mask=VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .src_access_mask=VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                .dst_stage_mask=VK_PIPELINE_STAGE_2_NONE,
                                .dst_access_mask=0,
                                .aspects=ImageAspects::COLOR}
        );

        cmd_buffer.end();

        cmd_buffer.submit(vk,
                           frame_in_flight.swapchain_img_ready_sema,
                           VK_PIPELINE_STAGE_2_BLIT_BIT,
                           swapchain_render_done_semas[swapchain_img_idx],
                           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, // All commands to make sure transitions are complete when we signal to the present engine that it can take the image, can we make this scope smaller?
                           frame_in_flight.rendering_done_fence
        );

        swapchain.present(vk, swapchain_render_done_semas[swapchain_img_idx], swapchain_img_idx);

        ++frame_count;
    }
};
