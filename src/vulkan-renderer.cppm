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
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"

export module vulkanRenderer;

import vulkanUtil;

static glm::ivec2 get_window_size_in_pixels(SDL_Window *window){
    glm::ivec2 size;
    SDL_GetWindowSizeInPixels(window, &size.x, &size.y);
    return size;
}

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

struct ShaderData{
    glm::vec4 a;
    glm::vec4 b;
    glm::vec4 c;
    glm::vec4 d;
};

struct Vertex{
    glm::vec3 pos;
    glm::vec3 color;
};

export class Renderer{
public:
    static constexpr int FRAMES_IN_FLIGHT = 2;

    SDL_Window *window;
    glm::ivec2 window_size;
    VulkanEngine vk;

    Swapchain swapchain;
    Image draw_image;
    ImageView draw_image_view;
    std::vector<GpuSemaphore> swapchain_render_done_semas;
    CommandPool command_pool; // There must be one command pool for each thread and each thread only touches cmd buffers from its pool
    std::array<FrameInFlightData, FRAMES_IN_FLIGHT> frame_in_flight_data;

    DescriptorSetBuilder ds_builder;
    DescriptorSet desc_set;

    PushConstantsBuilder pc_builder;
    PushConstant<ShaderData> push_const;

    int selected_compute_pipeline = 0;
    SpecializationInfo specialization_info;
    PipelineLayout compute_pipeline_layout;
    ComputePipeline gradient_pipeline;
    ComputePipeline sky_pipeline;

    VertexShader vert_shader;
    FragmentShader frag_shader;
    DescriptorSet graphics_desc_set;
    VertexBuffer vertex_buffer;
    StagingBuffer staging_buffer;
    GraphicsPipeline graphics_pipeline;

    GpuFence immediate_submit_fence;
    CommandPool immediate_cmd_pool;
    CommandBuffer immediate_cmd_buffer;

    uint32_t frame_count{};

    void immediate_submit(const std::function<void()> &function){
        immediate_cmd_buffer.restart(true);

        function();

        immediate_cmd_buffer.end();
        immediate_cmd_buffer.submit(vk, immediate_submit_fence);
        immediate_submit_fence.wait(vk);
    }

    Renderer(SDL_Window *window)
    :
    window(window),
    window_size(get_window_size_in_pixels(window)),
    vk(window),
    swapchain(vk, window, VK_PRESENT_MODE_FIFO_KHR),
    draw_image(vk,
               {static_cast<uint32_t>(window_size.x), static_cast<uint32_t>(window_size.y)},
               VK_FORMAT_R16G16B16A16_SFLOAT,
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT
               | VK_IMAGE_USAGE_TRANSFER_DST_BIT
               | VK_IMAGE_USAGE_STORAGE_BIT
               | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    draw_image_view(vk, draw_image, ImageAspects::COLOR, 0, 1),
    swapchain_render_done_semas(
        [&] -> std::vector<GpuSemaphore> {
            std::vector<GpuSemaphore> semas;
            int sema_count = (int)swapchain.get_images().size();
            semas.reserve(sema_count);
            for (int i = 0; i <sema_count; ++i){
                semas.emplace_back(vk);
            }
            return semas;
        }()
    ),
    command_pool(vk),
    frame_in_flight_data({FrameInFlightData(vk, command_pool), FrameInFlightData(vk, command_pool)}),
    ds_builder(),
    desc_set(ds_builder.bind(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).build(vk)),
    push_const(pc_builder.add<ShaderData>(VK_SHADER_STAGE_COMPUTE_BIT)),
    specialization_info((2 * sizeof(int32_t)) + sizeof(double)),
    compute_pipeline_layout(vk, desc_set, pc_builder.get_ranges()),
    gradient_pipeline(vk, ComputeShader(vk, "shaders/compiled/gradient.comp.spv"), compute_pipeline_layout, &specialization_info.reset().add_entry(0, 16).add_entry(1, 32).add_entry(1234, 34.0)),
    sky_pipeline(vk, ComputeShader(vk, "shaders/compiled/sky.comp.spv"), compute_pipeline_layout, &specialization_info.reset()),
    vert_shader(vk, "shaders/compiled/colored-triangle.vert.spv"),
    frag_shader(vk, "shaders/compiled/colored-triangle.frag.spv"),
    graphics_desc_set(ds_builder.reset().build(vk)),
    vertex_buffer(vk, {VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT}, 16),
    staging_buffer(vk, sizeof(Vertex) * 16),
    graphics_pipeline(vk, vert_shader, frag_shader, PipelineLayout(vk, graphics_desc_set, std::nullopt), vertex_buffer, draw_image.get_format(), VK_FORMAT_UNDEFINED, MSAALevel::OFF),
    immediate_submit_fence(vk, false),
    immediate_cmd_pool(vk),
    immediate_cmd_buffer(vk, immediate_cmd_pool)
    {
        desc_set.update(vk, 0, draw_image_view);
        vk.init_imgui(window, swapchain.get_format());
        push_const.data.a += glm::vec4(1, 0, 0, 1);
        push_const.data.b += glm::vec4(0, 0, 1, 1);
        std::array<Vertex, 3> vertices = {{
            {.pos={1.0f, 1.0f, 0.5f}, .color={1.0f, 0.0f, 0.0f}},
            {.pos={0.0f, -1.0f, 0.5f}, .color={0.0f, 0.0f, 1.0f}},
            {.pos={-1.0f, 1.0f, 0.5f}, .color={0.0f, 1.0f, 0.0f}},
        }};
        memcpy(staging_buffer.get_mapped_data(), vertices.data(), vertices.size() * sizeof(Vertex));
        immediate_submit([&]{immediate_cmd_buffer.copy_entire_buffer(staging_buffer, vertex_buffer);});
    }

    FrameInFlightData &get_current_frame_in_flight(){ return frame_in_flight_data[frame_count % FRAMES_IN_FLIGHT]; }

    ComputePipeline &get_selected_compute_pipeline(){
        switch (selected_compute_pipeline){
            case 0: return this->gradient_pipeline;
            case 1: return this->sky_pipeline;
            default: {
                std::println("Error: Invalid compute pipeline of selected_compute_pipeline={}, aborting", selected_compute_pipeline);
                abort();
            }
        }
    }

    void draw(){
	ImGui::Begin("Background customizer");

        const char *comp_pipeline_names[] = {"Gradient", "Sky"};
        ImGui::Combo("Compute Pipeline", &selected_compute_pipeline, comp_pipeline_names, IM_ARRAYSIZE(comp_pipeline_names));
        ImGui::InputFloat4("data a",(float*)& push_const.data.a);
        ImGui::InputFloat4("data b",(float*)& push_const.data.b);
        ImGui::InputFloat4("data c",(float*)& push_const.data.c);
        ImGui::InputFloat4("data d",(float*)& push_const.data.d);
	
        ImGui::End();


        FrameInFlightData &frame_in_flight = get_current_frame_in_flight();

        frame_in_flight.rendering_done_fence.wait(vk);
        uint32_t swapchain_img_idx = swapchain.acquire_next_image(vk, frame_in_flight.swapchain_img_ready_sema);

        CommandBuffer &cmd_buffer = frame_in_flight.main_cmd_buffer;
        cmd_buffer.restart(true);
        cmd_buffer.barrier(CommandBuffer::BarrierInfo{
                                .img=draw_image,
                                .discard_current_data=true,
                                .new_layout=VK_IMAGE_LAYOUT_GENERAL,
                                .src_stage_mask=VK_PIPELINE_STAGE_2_BLIT_BIT,
                                .src_access_mask=0,
                                .dst_stage_mask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                .dst_access_mask=0, // todo: do i really not need to make visible this transition to the compute stage?
                                .aspects=ImageAspects::COLOR}
        );

        cmd_buffer.update_push_constants(get_selected_compute_pipeline(), push_const);

        cmd_buffer.bind_pipeline(get_selected_compute_pipeline());
        cmd_buffer.bind_descriptor_sets(get_selected_compute_pipeline(), desc_set);
        cmd_buffer.dispatch(std::ceil(window_size.x / 16.0), std::ceil(window_size.y / 16.0), 1);

        cmd_buffer.barrier(CommandBuffer::BarrierInfo{
                               .img=draw_image,
                               .discard_current_data=false,
                               .new_layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                               .src_stage_mask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               .src_access_mask=VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                               .dst_stage_mask=VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                               .dst_access_mask=VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                               .aspects=ImageAspects::COLOR}
        );

        cmd_buffer.draw(draw_image_view, draw_image.extent, graphics_pipeline, vertex_buffer);

        cmd_buffer.barrier(CommandBuffer::BarrierInfo{
                                .img=draw_image,
                                .discard_current_data=false,
                                .new_layout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                .src_stage_mask=VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .src_access_mask=VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                .dst_stage_mask=VK_PIPELINE_STAGE_2_BLIT_BIT,
                                .dst_access_mask=VK_ACCESS_2_TRANSFER_READ_BIT,
                                .aspects=ImageAspects::COLOR
                           },
                           CommandBuffer::BarrierInfo{
                                .img=swapchain.get_images()[swapchain_img_idx],
                                .discard_current_data=true,
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

        cmd_buffer.barrier(CommandBuffer::BarrierInfo{
                                .img=swapchain.get_images()[swapchain_img_idx],
                                .discard_current_data=false,
                                .new_layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                .src_stage_mask=VK_PIPELINE_STAGE_2_BLIT_BIT,
                                .src_access_mask=VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .dst_stage_mask=VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .dst_access_mask=VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                                .aspects=ImageAspects::COLOR}
        );

        cmd_buffer.draw_imgui(swapchain.get_image_views()[swapchain_img_idx], swapchain.get_extent());

        cmd_buffer.barrier(CommandBuffer::BarrierInfo{
                                .img=swapchain.get_images()[swapchain_img_idx],
                                .discard_current_data=false,
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
