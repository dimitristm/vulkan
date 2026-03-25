module;

#include "imgui/imgui_impl_sdl3.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_events.h>

export module userInput;


class Camera{
public:
    glm::vec3 pos = glm::vec3(0.0f, 0.0f, 3.0f);
    float mouse_sensitivity = 0.1f;
private:
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 direction = glm::vec3(0.0f, 0.0f, -1.0f);
    float yaw = -90.f;
    float pitch = 0.0f;
    float speed = 0.35;

public:
    void update_direction(float mouse_movement_x, float mouse_movement_y){
        yaw += mouse_movement_x * mouse_sensitivity;
        pitch += mouse_movement_y * mouse_sensitivity;

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
    Camera camera;
    SDL_Window *window;
    // todo: add the time here, pass time to camera move functions

public:
    bool should_quit{};

    UserInputHandler(SDL_Window *window)
    :window(window)
    {}

    const Camera &get_camera() { return camera; }

    void handle_key_press(const SDL_KeyboardEvent &key){
        if (key.scancode == SDL_SCANCODE_R){
            glm::ivec2 window_size;
            SDL_GetWindowSize(window, &window_size.x, &window_size.y);
            SDL_WarpMouseInWindow(window, window_size.x/2, window_size.y/2);
            SDL_SetWindowRelativeMouseMode(window, !SDL_GetWindowRelativeMouseMode(window));
        }
        else if (key.scancode == SDL_SCANCODE_W){ camera.move_forward(); }
        else if (key.scancode == SDL_SCANCODE_A){ camera.move_left(); }
        else if (key.scancode == SDL_SCANCODE_S){ camera.move_back(); }
        else if (key.scancode == SDL_SCANCODE_D){ camera.move_right(); }
    }

    void handle_input(){
        assert(!should_quit && "You were supposed to quit, but didn't.");

        SDL_Event event;
        while(SDL_PollEvent(&event)){
            if (event.type == SDL_EVENT_QUIT){
                should_quit = true;
                return;
            }
            else if (event.type == SDL_EVENT_MOUSE_MOTION){ camera.update_direction(event.motion.xrel, event.motion.yrel); }
            else if (event.type == SDL_EVENT_KEY_DOWN){ handle_key_press(event.key); }
            ImGui_ImplSDL3_ProcessEvent(&event);
        }
    }
};
