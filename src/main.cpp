#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include <SDL3/SDL_init.h>
#include <boost/pfr.hpp>
#include <cstdint>
#include <print>
#include <glm/vec2.hpp>

import vulkanRenderer;
template <typename T, typename Tag>
struct strong_type_with_implicit_cast
{
    using underlying_type = T;
    using tag_type        = Tag;

    T value;

    constexpr strong_type_with_implicit_cast() = default;

    // Implicit casting:
    constexpr strong_type_with_implicit_cast(T v) : value(v) {}
    constexpr operator T&() noexcept { return value; }
    constexpr operator T const&() const noexcept { return value; }
    constexpr operator T() const noexcept { return value; }
};

#define STRONG_TYPE_WITH_IMPLICIT_CAST(name, underlying)                        \
    namespace {struct name##___tag_mangle {};};                                 \
    using name = strong_type_with_implicit_cast<underlying, name##___tag_mangle>\

STRONG_TYPE_WITH_IMPLICIT_CAST(int32_norm_t, int32_t);

// Tells std::print to print it like it would print the underlying_type
template <typename T, typename Tag, typename CharT>
struct std::formatter<strong_type_with_implicit_cast<T, Tag>, CharT> : std::formatter<T, CharT>{
};

template <typename T>
static void get_attribute_formats(const T &value){

    boost::pfr::for_each_field_with_name(value, [](std::string_view name, const auto &value){
        //static_assert(std::is_same_v<std::remove_cvref_t<decltype(value)>, int32_t>, "above field wasn't int32_t");
    });
    boost::pfr::tuple_element_t<0, T>;
}

struct vec2_norm : public glm::vec2{
public:
    using glm::vec2::vec2;
    constexpr vec2_norm(glm::vec2 value) : glm::vec2(value){}
};

int main(){
    glm::vec2 a{2,3};
    vec2_norm b{3,45};
    auto sdf = b.length();
    a = b;
    b = a;


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

    {
        Renderer renderer{window};
        SDL_Event event;
        bool quit = false;
        while (!quit){
            while(SDL_PollEvent(&event)){
                if (event.type == SDL_EVENT_QUIT) { quit = true; }
                ImGui_ImplSDL3_ProcessEvent(&event);
            }

            ImGui::ShowDemoWindow();
            renderer.draw();
        }
    }

    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
