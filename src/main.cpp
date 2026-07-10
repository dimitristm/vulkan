#include <SDL3/SDL_init.h>
#include <Tracy.hpp>

#if !VK_PROJ_USE_IMPORT_STD
#include <print>
#endif

#if VK_PROJ_USE_IMPORT_STD
import std;
#endif

import vulkanEngine;
import vulkanRenderer;
import userInput;
import assets;
import vulkanUtil;
import util;
import types;

import imgui;

int main(){
    TracyNoop;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window{};

    std::string a("assets/mytests/main.assetpack");
    // Assetpack::Builder b(a, BCnQuality::MIN, MipQuality::MIN);
    // b.add_from_gltf("assets/mytests/BoxTextured.glb")
    //     .add_from_gltf("assets/mytests/BoxTexturedNonPowerOfTwo.glb")
    // //.add_from_gltf("assets/flight-helmet.glb")
    // .build();

    {
        ResourceLoader loader;
        loader.load_assetpack_table_of_contents(a);
        SDL_WindowFlags window_flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;
        i32 window_width = 1700;
        i32 window_height = 900;
        window = SDL_CreateWindow(
            "Vulkan app",
            window_width,
            window_height,
            window_flags
        );

        VulkanEngine vk{window};
        vk.init_imgui(window, DrawImage::format);
        loader.init_vulkan_resources(vk);
        loader.load_everything_to_gpu();
        UserInputHandler input_handler{window};
        util::FrameTimer frame_timer;

        int scene_idx{};
        Renderer renderer{vk, loader, window};
        while (!input_handler.should_quit){
            frame_timer.begin_frame();
            input_handler.handle_input();
            //ImGui::ShowDemoWindow();
            ImGui::InputInt("Value", &scene_idx);

            frame_timer.imgui();
            renderer.draw(input_handler.get_camera().get_view_transform(), loader, scene_idx);
            frame_timer.end_frame();
        }
    }

    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
