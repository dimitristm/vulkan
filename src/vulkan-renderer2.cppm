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

export class Renderer2{
public:
    SDL_Window *window;
    glm::ivec2 window_size;
    VulkanEngine vk;

    Image draw_image;

    Renderer2(SDL_Window *window)
    :window(window),
     window_size(get_window_size_in_pixels(window)),
     vk(window),
     draw_image(vk.create_image({window_size.x, window_size.y, 1}, ))
     
    {
        SDL_GetWindowSizeInPixels(window, &window_size.x, &window_size.y);
    }
};
