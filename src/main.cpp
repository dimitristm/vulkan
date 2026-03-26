#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include <SDL3/SDL_init.h>

import vulkanRenderer;
import userInput;

int main(){
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window{};
    SDL_WindowFlags window_flags = SDL_WINDOW_VULKAN;
    int window_width = 1700;
    int window_height = 900;
    window = SDL_CreateWindow(
        "Vulkan app",
        window_width,
        window_height,
        window_flags
    );

    UserInputHandler input_handler(window);

    {
        Renderer renderer{window};
        while (!input_handler.should_quit){
            input_handler.handle_input();

            ImGui::ShowDemoWindow();
            renderer.draw(input_handler.get_camera().get_view_transform());
        }
    }

    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
