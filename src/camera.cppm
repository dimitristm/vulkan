module;

#if !VK_PROJ_USE_IMPORT_STD
#include <print>
#endif

export module camera;

#if VK_PROJ_USE_IMPORT_STD
import std;
#endif
import types;
import glm;

export class Camera{
public:
    fvec3 pos{0.0f, 0.0f, 3.0f};
    f32 mouse_sensitivity = 0.1f;
private:
    fvec3 up{0.0f, 1.0f, 0.0f};
    fvec3 direction{0.0f, 0.0f, -1.0f};
    f32 yaw = -90.f;
    f32 pitch = 0.0f;
    f32 speed = 0.02;

public:
    void update_direction(f32 mouse_movement_x, f32 mouse_movement_y){
        yaw += mouse_movement_x * mouse_sensitivity;
        pitch -= mouse_movement_y * mouse_sensitivity;

        if(pitch > 89.0f) pitch = 89.0f;
        if(pitch < -89.0f) pitch = -89.0f;

        direction.x = glm::cos(glm::radians(yaw)) * glm::cos(glm::radians(pitch));
        direction.y = glm::sin(glm::radians(pitch));
        direction.z = glm::sin(glm::radians(yaw)) * glm::cos(glm::radians(pitch));
        direction = glm::normalize(direction);
    }

    [[nodiscard]] fmat4 get_view_transform() const {
        return glm::gtc::lookAt(pos, pos + get_direction(), up);
    }

    [[nodiscard]] f32 get_yaw() const { return yaw; }
    [[nodiscard]] f32 get_pitch() const { return pitch; }
    [[nodiscard]] const fvec3 &get_direction() const { return direction; }
    [[nodiscard]] const fvec3 &get_upwards_vector() const { return up; }
    [[nodiscard]] const fvec3 &get_position() const { return pos; }

    void move_forward(){ pos += speed * direction; }
    void move_left()   { pos -= glm::normalize(glm::cross(direction, up)) * speed; }
    void move_back()   { pos -= speed * direction; }
    void move_right()  { pos += glm::normalize(glm::cross(direction, up)) * speed; }
};

