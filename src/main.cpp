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

class ScenePicker{
    i32 scene_idx{};
    const ResourceLoader &loader;
public:
    ScenePicker(const ResourceLoader &loader):loader(loader){}

    i32 pick(i32 new_scene_idx){
        i32 scene_count = static_cast<i32>(loader.scene_count());
        scene_idx = std::clamp(new_scene_idx, 0, scene_count-1);
        return scene_idx;
    }

    void imgui_content(){
        i32 scene_count = static_cast<i32>(loader.scene_count());
        ImGui::Text("Scene count: %d (max idx: %d)", scene_count, scene_count-1);
        ImGui::InputInt("Scene idx", &scene_idx);
        scene_idx = std::clamp(scene_idx, 0, scene_count-1);
    }

    [[nodiscard]] i32 get_scene_idx() const { return scene_idx; }
};

int main(){
    TracyNoop;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window{};

    std::string a("assets/mytests/main.assetpack");
    // {
    //     Assetpack::Builder b(a, BCnQuality::MIN, MipQuality::MIN);
    //     b.add_from_gltf("assets/mytests/BoxTextured.glb")
    //         .add_from_gltf("assets/mytests/BoxTexturedNonPowerOfTwo.glb")
    //     // .add_from_gltf("assets/flight-helmet.glb")
    //     .build();
    // }

    {
        ResourceLoader loader;
        loader.load_assetpack_table_of_contents(a);
        ScenePicker scene_picker{loader};
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

        Renderer renderer{vk, loader, window};
        while (!input_handler.should_quit){
            frame_timer.begin_frame();
            input_handler.handle_input();
            //ImGui::ShowDemoWindow();

            ImGui::Begin("General");
            ImGui::SeparatorText("FPS");
            frame_timer.imgui_content();
            ImGui::SeparatorText("Scene Changer");
            scene_picker.imgui_content();
            ImGui::End();

            renderer.draw(input_handler.get_camera().get_view_transform(), loader, scene_picker.get_scene_idx());
            frame_timer.end_frame();
        }
    }

    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
