#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include <SDL3/SDL_init.h>
//import vulkanRenderer;
import vulkanRenderer2;
//import vulkanUtil;

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

    Renderer2 renderer{window};
    SDL_Event event;
    bool quit = false;
    while (!quit){
        while(SDL_PollEvent(&event)){
            if (event.type == SDL_EVENT_QUIT) { quit = true; }
        }
        ImGui_ImplSDL3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        ImGui::ShowDemoWindow();

        ImGui::Render();

        renderer.draw();
    }
    return 0;
}
