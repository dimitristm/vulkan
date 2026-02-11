#include <SDL3/SDL_init.h>
import vulkanRenderer;
import vulkanRenderer2;
//import vulkanUtil;

#include <print>
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

    Renderer renderer{};
    renderer.run();
    return 0;
}
