module;

#include <functional>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3/SDL_events.h>

#include <VkBootstrap.h>
#include <vulkan/vk_enum_string_helper.h>

#include <array>
#include <cmath>
#include <print>

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"

export module vulkanRenderer2;

import vulkanUtil;

static glm::ivec2 get_window_size_in_pixels(SDL_Window *window){
    glm::ivec2 size;
    SDL_GetWindowSizeInPixels(window, &size.x, &size.y);
    return size;
}

struct FrameInFlightData{
    GpuSemaphore swapchain_img_ready_sema;
    CommandPool cmd_pool;
    CommandBuffer main_cmd_buffer;
    GpuFence rendering_done_fence;

    FrameInFlightData(VulkanEngine &engine)
    :swapchain_img_ready_sema(engine.create_semaphore()),
     cmd_pool(engine.create_command_pool()),
     main_cmd_buffer(engine.create_command_buffer(this->cmd_pool)),
     rendering_done_fence(engine.create_fence(true))
    { }
};

export class Renderer2{
public:
    static constexpr int FRAMES_IN_FLIGHT = 2;

    SDL_Window *window;
    glm::ivec2 window_size;
    VulkanEngine vk;

    Swapchain swapchain;
    Image draw_image;
    ImageView draw_image_view;
    std::vector<GpuSemaphore> swapchain_render_done_semas;
    std::array<FrameInFlightData, FRAMES_IN_FLIGHT> frame_in_flight_data;

    DescriptorSetBuilder ds_builder;
    DescriptorSet desc_set;

    ComputePipeline gradient_pipeline;

    GpuFence immediate_submit_fence;
    CommandPool immediate_cmd_pool;
    CommandBuffer immediate_cmd_buffer;

    uint32_t frame_count{};

    struct PushConstData{
        glm::vec4 a;
        glm::vec4 b;
        glm::vec4 c;
        glm::vec4 d;
    };

    Renderer2(SDL_Window *window)
    :
    window(window),
    window_size(get_window_size_in_pixels(window)),
    vk(window),
    swapchain(vk.create_swapchain(window, VK_PRESENT_MODE_FIFO_KHR)),
    draw_image(vk.create_image(window_size,
                               VK_FORMAT_R16G16B16A16_SFLOAT,
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                               | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                               | VK_IMAGE_USAGE_STORAGE_BIT
                               | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)),
    draw_image_view(vk.create_image_view(draw_image, ImageAspects::COLOR, 0, 1)),
    swapchain_render_done_semas(
        [&] -> std::vector<GpuSemaphore> {
            std::vector<GpuSemaphore> semas;
            int sema_count = (int)swapchain.get_images().size();
            semas.reserve(sema_count);
            for (int i = 0; i <sema_count; ++i){
                semas.push_back(vk.create_semaphore());
            }
            return semas;
        }()
    ),
    frame_in_flight_data({FrameInFlightData(vk), FrameInFlightData(vk)}),
    ds_builder(
        [&] -> DescriptorSetBuilder{
            DescriptorSetBuilder b;
            b.bind(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            return b;
        }()
    ),
    desc_set(ds_builder.build(vk)),
    gradient_pipeline(vk.create_compute_pipeline(desc_set,
                                                 // std::vector<VkPushConstantRange>{{
                                                 //     .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                 //     .offset = 0,
                                                 //     .size = sizeof(PushConstData),
                                                 // }},
                                                 std::nullopt,
                                                 "shaders/compiled/gradient.comp.spv")),
    immediate_submit_fence(vk.create_fence(true)),
    immediate_cmd_pool(vk.create_command_pool()),
    immediate_cmd_buffer(vk.create_command_buffer(immediate_cmd_pool))
    {
        vk.update_storage_image_descriptor(desc_set, draw_image_view, 0);
        vk.init_imgui(window, swapchain);
    }

    FrameInFlightData &get_current_frame_in_flight(){ return frame_in_flight_data[frame_count % FRAMES_IN_FLIGHT]; }

    void immediate_submit(std::function<void()>& function){
        immediate_cmd_buffer.restart(true);

        function();

        immediate_cmd_buffer.end();

        vk.submit_commands(immediate_cmd_buffer, immediate_submit_fence);

        vk.wait(immediate_submit_fence);
    }

    void draw(){
        FrameInFlightData &frame_in_flight = get_current_frame_in_flight();

        vk.wait(frame_in_flight.rendering_done_fence);
        uint32_t swapchain_img_idx = vk.acquire_next_image(swapchain, frame_in_flight.swapchain_img_ready_sema);

        CommandBuffer &cmd_buffer = frame_in_flight.main_cmd_buffer;
        cmd_buffer.restart(true);
        cmd_buffer.barrier(draw_image,
                           true,
                           VK_IMAGE_LAYOUT_GENERAL,
                           VK_PIPELINE_STAGE_2_NONE,
                           0,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           0,
                           ImageAspects::COLOR
        );

        cmd_buffer.bind_pipeline(gradient_pipeline);
        cmd_buffer.bind_descriptor_sets(gradient_pipeline, desc_set);
        cmd_buffer.dispatch(std::ceil(window_size.x / 16.0), std::ceil(window_size.y / 16.0), 1);

        cmd_buffer.barrier(draw_image,
                           false,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_BLIT_BIT,
                           VK_ACCESS_2_TRANSFER_READ_BIT,
                           ImageAspects::COLOR
        );

        cmd_buffer.barrier(swapchain.get_images()[swapchain_img_idx],
                           true,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_PIPELINE_STAGE_2_NONE,
                           0,
                           VK_PIPELINE_STAGE_2_BLIT_BIT,
                           0,
                           ImageAspects::COLOR
        );

        cmd_buffer.blit_entire_images(draw_image,
                                      swapchain.get_images()[swapchain_img_idx],
                                      ImageAspects::COLOR
        );

        cmd_buffer.barrier(swapchain.get_images()[swapchain_img_idx],
                           false,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_PIPELINE_STAGE_2_BLIT_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                           ImageAspects::COLOR);

        cmd_buffer.draw_imgui(swapchain.get_image_views()[swapchain_img_idx], swapchain.get_extent());

        cmd_buffer.barrier(swapchain.get_images()[swapchain_img_idx],
                           false,
                           VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_NONE,
                           0,
                           ImageAspects::COLOR
        );

        cmd_buffer.end();

        vk.submit_commands(cmd_buffer,
                           frame_in_flight.swapchain_img_ready_sema,
                           VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           swapchain_render_done_semas[swapchain_img_idx],
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                           frame_in_flight.rendering_done_fence
        );

        vk.present(swapchain, swapchain_render_done_semas[swapchain_img_idx], swapchain_img_idx);

        ++frame_count;
    }
};
