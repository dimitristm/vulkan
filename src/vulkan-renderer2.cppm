module;


#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3/SDL_events.h>

#include <VkBootstrap.h>
#include <vulkan/vk_enum_string_helper.h>

#include <array>
#include <thread>
#include <chrono>
#include <cmath>
#include <print>
#include <fstream>
#include <functional>

#include <glm/vec2.hpp>

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"


// Disable harmless VMA warnings
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Weverything"
#elif defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wall"
#  pragma GCC diagnostic ignored "-Wextra"
#endif

#include <vk_mem_alloc.h>

#if defined(__clang__)
#  pragma clang diagnostic pop
#elif defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

//TODO: enable VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT

export module vulkanRenderer2;

import vulkanUtil;

static glm::ivec2 get_window_size_in_pixels(SDL_Window *window){
    glm::ivec2 size;
    SDL_GetWindowSizeInPixels(window, &size.x, &size.y);
    return size;
}

DescriptorSetBuilder asdf(){
    DescriptorSetBuilder b;
    b.bind(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    return b;
}

export class Renderer2{
public:
    static constexpr int FRAMES_IN_FLIGHT = 2;

    SDL_Window *window;
    glm::ivec2 window_size;
    VulkanEngine vk;
    Swapchain swapchain;
    Image draw_image;
    ImageView draw_image_view;
    std::array<FrameData, FRAMES_IN_FLIGHT> frames;
    DescriptorSetBuilder ds_builder;
    DescriptorSet set;

    Renderer2(SDL_Window *window)
    :window(window),
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
     frames({FrameData(vk), FrameData(vk)}),
     ds_builder([]{DescriptorSetBuilder b; b.bind(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE); return b;}),
    {}
};
