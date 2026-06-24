module;

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3/SDL_events.h>

#include <VkBootstrap.h>
#include <glm/ext/matrix_float4x4.hpp>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl3.h"
#include <stb/stb_image.h>

#if !USE_IMPORT_STD
#include <array>
#include <print>
#endif

export module vulkanRenderer;
#if USE_IMPORT_STD
import std;
#endif

import vulkanEngine;
import userInput;
import vulkanUtil;
import assets;
import util;
import types;

using namespace util;

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

export class Renderer{
    static constexpr int FRAMES_IN_FLIGHT = 2;
    const VkPresentModeKHR present_mode = VK_PRESENT_MODE_MAILBOX_KHR;

    VulkanEngine &vk;
    SDL_Window *window;
    ivec2 window_size;// in pixels
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
    PushConstant<fvec4> compute_push_const;
    PipelineLayout compute_pipeline_layout;
    ComputePipeline compute_pipeline;

    PushConstant<fmat4> view_proj_transform_const;
    GraphicsPipeline<Vertex> graphics_pipeline;

    bool should_rebuild_swapchain = false;
    void update_drawing_surfaces(bool swapchain_out_of_date){
        ivec2 new_window_size = get_window_size_in_pixels(window);
        bool window_changed_size = new_window_size != window_size;

        const auto update_swapchain = [&](){
            swapchain.rebuild_swapchain(vk, window, present_mode);
            should_rebuild_swapchain = false;
        };
        const auto update_images = [&](){
            draw_image.resize(vk, new_window_size);
            depth_image.resize(vk, new_window_size);
            immediate_submiter.submit(vk, [&](){depth_image.set_layout(immediate_submiter.cmd_buffer());});
            compute_desc_set.update_storage_image(vk, 0, draw_image.view);
        };

        if (window_changed_size || swapchain_out_of_date || should_rebuild_swapchain){
            window_size = new_window_size;
            vk.wait_idle();
            update_images();
            update_swapchain();
        }
    }

public:
    Renderer(VulkanEngine &vk, ResourceLoader &loader, SDL_Window *window)
    :
    vk(vk),
    window(window),
    window_size(get_window_size_in_pixels(window)),
    command_pool(vk),
    immediate_submiter(vk, command_pool),
    uploader(&vk, command_pool, 64 * 1024 * 1024),// todo performance 64MiB is small, raise it if we do more later
    swapchain(vk, window, present_mode),
    draw_image(vk, get_window_size_in_pixels(window)),
    depth_image(vk, get_window_size_in_pixels(window)),
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
    specialization_info((2 * sizeof(int32_t))),
    pc_builder(),
    ds_builder(),
    compute_desc_set(ds_builder.bind(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).build(vk)),
    compute_push_const(pc_builder.add<fvec4>(VK_SHADER_STAGE_COMPUTE_BIT)),
    compute_pipeline_layout(vk, compute_desc_set, pc_builder.get_ranges()),
    compute_pipeline(vk, ComputeShader(vk, "shaders/compiled/gradient.comp.spv"), compute_pipeline_layout, &specialization_info.reset().add_entry(0, 16).add_entry(1, 32)),
    view_proj_transform_const(pc_builder.reset().add<fmat4>( VK_SHADER_STAGE_VERTEX_BIT)),
    graphics_pipeline(vk,
                      VertexShader(vk, "shaders/compiled/colored_triangle_mesh.vert.spv"),
                      FragmentShader(vk, "shaders/compiled/textured.frag.spv"),
                      PipelineLayout(vk, loader.get_descriptor_set(), pc_builder.get_ranges()),
                      loader.get_vertex_buffer(),
                      draw_image.img.get_format(),
                      depth_image.img.get_format(),
                      MSAALevel::OFF)
    {
        vk.init_imgui(window, swapchain.get_format());
        compute_desc_set.update_storage_image(vk, 0, draw_image.view);
        compute_push_const.data += fvec4(1, 0, 0, 1);

        immediate_submiter.submit(vk, [&]{
            depth_image.set_layout(immediate_submiter.cmd_buffer());
        });
    }

    FrameInFlightData &get_current_frame_in_flight(){ return frame_in_flight_data[frame_count % FRAMES_IN_FLIGHT]; }

    void draw_gui(){
	ImGui::Begin("Background customizer");
        ImGui::InputFloat4("data a",(float*)& compute_push_const.data.a);
        ImGui::End();
    }

    void draw_scene(const fmat4 &view_transform, ResourceLoader &loader, u64 scene_idx){
        FrameInFlightData &frame_in_flight = get_current_frame_in_flight();

        frame_in_flight.rendering_done_fence.wait_and_reset(vk);
        CommandBuffer &cmd_buffer = frame_in_flight.main_cmd_buffer;
        cmd_buffer.restart(true);

        uint32_t swapchain_img_idx = [&] {
            while (true){
                auto [idx, result] = swapchain.acquire_next_image(vk, frame_in_flight.swapchain_img_ready_sema);
                if (result == VK_ERROR_OUT_OF_DATE_KHR) update_drawing_surfaces(true);
                if (result == VK_SUBOPTIMAL_KHR) { should_rebuild_swapchain = true; return idx; }
                if (result == VK_SUCCESS) return idx;
            }
        }();

        cmd_buffer.barrier(BarrierInfo{
                                .img=draw_image.img,
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
                               .img=draw_image.img,
                               .old_layout_or_undefined_to_discard_current_data=VK_IMAGE_LAYOUT_GENERAL,
                               .new_layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                               .src_stage_mask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               .src_access_mask=VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                               .dst_stage_mask=VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                               .dst_access_mask=VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                               .aspects=ImageAspects::COLOR}
        );

        cmd_buffer.bind_pipeline(graphics_pipeline);
        cmd_buffer.bind_descriptor_set(graphics_pipeline, loader.get_descriptor_set());
        view_proj_transform_const.data = perspective_projection(80.0f, (float)window_size.x / (float)window_size.y, 0.001f, 10000.0f) * view_transform;
        cmd_buffer.update_push_constants(graphics_pipeline, view_proj_transform_const);
        auto draw_commands = loader.prepare_to_draw_scene(scene_idx);
        cmd_buffer.draw_indexed(draw_image.view, depth_image.view, {.width=static_cast<uint32_t>(window_size.x), .height=static_cast<uint32_t>(window_size.y)},
                                loader.get_vertex_buffer(), loader.get_index_buffer(), draw_commands);

        cmd_buffer.barrier(BarrierInfo{
                                .img=draw_image.img,
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

        cmd_buffer.blit(draw_image.img,
                        swapchain.get_images()[swapchain_img_idx],
                        {0,0},
                        window_size,
                        {0,0},
                        {swapchain.get_images()[swapchain_img_idx].extent.width, swapchain.get_images()[swapchain_img_idx].extent.height},
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

        VkResult result = swapchain.present(vk, swapchain_render_done_semas[swapchain_img_idx], swapchain_img_idx);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) update_drawing_surfaces(true);
        if (result == VK_SUBOPTIMAL_KHR) should_rebuild_swapchain = true;

        ++frame_count;
    }

    void draw(const fmat4 &view_transform, ResourceLoader &loader, u64 scene_idx){
        if (!Swapchain::presentable_swapchain_exists(vk, window)){
            std::println("Can't build swapchain");
            return;
        }
        update_drawing_surfaces(false);
        draw_gui();
        draw_scene(view_transform, loader, scene_idx);
    }
};
