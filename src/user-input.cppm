module;

#include "imgui/imgui_impl_sdl3.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_events.h>
#if !USING_IMPORT_STD
#include <print>
#endif

export module userInput;
#if USING_IMPORT_STD
import std;
#endif

static bool is_fullscreen(SDL_Window *window) {
    SDL_WindowFlags flags = SDL_GetWindowFlags(window);
    return (flags & SDL_WINDOW_FULLSCREEN) != 0;
}

static void toggle_fullscreen(SDL_Window *window) {
    bool fullscreen = is_fullscreen(window);
    if (!SDL_SetWindowFullscreen(window, !fullscreen)) {
        std::println("Failed to toggle fullscreen: {}", SDL_GetError());
    }
}

class Camera{
public:
    glm::vec3 pos = glm::vec3(0.0f, 0.0f, 3.0f);
    float mouse_sensitivity = 0.1f;
private:
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 direction = glm::vec3(0.0f, 0.0f, -1.0f);
    float yaw = -90.f;
    float pitch = 0.0f;
    float speed = 0.02;

public:
    void update_direction(float mouse_movement_x, float mouse_movement_y){
        yaw += mouse_movement_x * mouse_sensitivity;
        pitch -= mouse_movement_y * mouse_sensitivity;

        if(pitch > 89.0f) pitch = 89.0f;
        if(pitch < -89.0f) pitch = -89.0f;

        direction.x = glm::cos(glm::radians(yaw)) * glm::cos(glm::radians(pitch));
        direction.y = glm::sin(glm::radians(pitch));
        direction.z = glm::sin(glm::radians(yaw)) * glm::cos(glm::radians(pitch));
        direction = glm::normalize(direction);
    }

    [[nodiscard]] glm::mat4 get_view_transform() const {
        return glm::lookAt(pos, pos + get_direction(), up);
    }

    [[nodiscard]] float get_yaw() const { return yaw; }
    [[nodiscard]] float get_pitch() const { return pitch; }
    [[nodiscard]] const glm::vec3 &get_direction() const { return direction; }
    [[nodiscard]] const glm::vec3 &get_upwards_vector() const {return up; }

    void move_forward(){ pos += speed * direction; }
    void move_left()   { pos -= glm::normalize(glm::cross(direction, up)) * speed; }
    void move_back()   { pos -= speed * direction; }
    void move_right()  { pos += glm::normalize(glm::cross(direction, up)) * speed; }
};


export class UserInputHandler{
public:
    bool should_quit{};

    UserInputHandler(SDL_Window *window)
    :window(window)
    {
        assert(!SDL_GetWindowRelativeMouseMode(window) && "Do not set relative mouse mode outside of UserInputHandler. Getting that right is its responsibility. Instead use set_control_mode and possibly other UserInputHandler functions.");
        control_mode = ControlMode::USER_CONTROLLING_THE_GUI;
    }

    const Camera &get_camera() { return camera; }

    void handle_input(){
        assert(!should_quit && "You were supposed to quit, but didn't.");
        assert(control_mode_and_relative_mode_are_synced() && "Error: Either you set the relative mode outside of UserInput (don't do that, UserInput decides whether relative mode should be enabled) or there is an issue with UserInput");

        handle_key_hold();
        SDL_Event event;
        while (SDL_PollEvent(&event) && !should_quit){
            ImGui_ImplSDL3_ProcessEvent(&event);
            handle_event(event);
        }
    }

private:
    SDL_Window *window;
    Camera camera;
    enum class ControlMode : char{
        USER_CONTROLLING_THE_CAMERA,
        USER_CONTROLLING_THE_GUI,
    } control_mode{};
    // todo: add the time here, pass time to camera move functions

    bool control_mode_and_relative_mode_are_synced(){
        switch (control_mode){
            case ControlMode::USER_CONTROLLING_THE_CAMERA: return SDL_GetWindowRelativeMouseMode(window);
            case ControlMode::USER_CONTROLLING_THE_GUI: return !SDL_GetWindowRelativeMouseMode(window);
            return false;
        }
    }

    void set_control_mode(ControlMode new_control_mode){
        assert(new_control_mode != control_mode && "You set the UserInputHandler::control_mode to a value it already had. This is probably a bug.");
        control_mode = new_control_mode;
        switch (new_control_mode){
            case ControlMode::USER_CONTROLLING_THE_CAMERA:
                ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouse;
                SDL_SetWindowRelativeMouseMode(window, true);
                return;
            case ControlMode::USER_CONTROLLING_THE_GUI:
                ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
                glm::ivec2 window_size;
                SDL_GetWindowSize(window, &window_size.x, &window_size.y);
                SDL_WarpMouseInWindow(window, window_size.x/2, window_size.y/2);
                SDL_SetWindowRelativeMouseMode(window, false);
                return;
        }
        assert(false && "Unhandled ControlMode value passed to UserInput::set_control_mode");
    }

    void handle_event(const SDL_Event &event){
        const auto handle_key_press = [this](const SDL_KeyboardEvent &key){
            if (key.scancode == SDL_SCANCODE_F11) toggle_fullscreen(window);

            switch (control_mode){
            case ControlMode::USER_CONTROLLING_THE_GUI: break;
            case ControlMode::USER_CONTROLLING_THE_CAMERA:
                if (key.scancode == SDL_SCANCODE_ESCAPE) set_control_mode(ControlMode::USER_CONTROLLING_THE_GUI);
                break;
            }
        };
        const auto handle_mouse_motion = [this](const SDL_MouseMotionEvent &motion){
            switch (control_mode){
            case ControlMode::USER_CONTROLLING_THE_GUI: break;
            case ControlMode::USER_CONTROLLING_THE_CAMERA:
                if (SDL_GetWindowRelativeMouseMode(window)) camera.update_direction(motion.xrel, motion.yrel);
                break;
            }
        };
        const auto handle_mouse_press = [this](const SDL_MouseButtonEvent &button){
            switch (control_mode){
            case ControlMode::USER_CONTROLLING_THE_GUI:
                if (button.button == SDL_BUTTON_LEFT && !ImGui::GetIO().WantCaptureMouse) {
                    set_control_mode(ControlMode::USER_CONTROLLING_THE_CAMERA);
                }
                break;
            case ControlMode::USER_CONTROLLING_THE_CAMERA: break;
            }
        };
        if (event.type == SDL_EVENT_QUIT) should_quit = true;
        else if (event.type == SDL_EVENT_KEY_DOWN) handle_key_press(event.key);
        else if (event.type == SDL_EVENT_MOUSE_MOTION) handle_mouse_motion(event.motion);
        else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) handle_mouse_press(event.button);
    }


    void handle_key_hold(){
        const bool *keys = SDL_GetKeyboardState(nullptr);

        if (keys[SDL_SCANCODE_W]) { camera.move_forward(); } // might want to try lowering the latency by checking the timestamp in handle_key_press
        if (keys[SDL_SCANCODE_A]) { camera.move_left(); }
        if (keys[SDL_SCANCODE_S]) { camera.move_back(); }
        if (keys[SDL_SCANCODE_D]) { camera.move_right(); }
    }
};
