#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl3.h"
#include <SDL3/SDL_init.h>
#include <Tracy.hpp>

import vulkanRenderer;
import userInput;
import util;

int main(){
    TracyNoop;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window{};
    SDL_WindowFlags window_flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;
    int window_width = 1700;
    int window_height = 900;
    window = SDL_CreateWindow(
        "Vulkan app",
        window_width,
        window_height,
        window_flags
    );

    UserInputHandler input_handler(window);
    util::FrameTimer frame_timer;

    {
        Renderer renderer{window};
        while (!input_handler.should_quit){
            frame_timer.begin_frame();
            input_handler.handle_input();
            //ImGui::ShowDemoWindow();

            frame_timer.imgui();
            renderer.draw(input_handler.get_camera().get_view_transform());
            frame_timer.end_frame();
        }
    }

    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
