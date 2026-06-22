module;
#include "imgui/imgui.h"
#include <glm/vec2.hpp>
#include <glm/trigonometric.hpp>
#include <glm/mat4x4.hpp>
#include <SDL3/SDL_video.h>
#include <emmintrin.h>
#if !USE_IMPORT_STD
#include <thread>
#include <chrono>
#include <print>
#include <cmath>
#endif

export module util;
#if USE_IMPORT_STD
import std;
#endif
import types;


namespace util{
export class Duration;
export class Time;

export ivec2 get_window_size_in_pixels(SDL_Window *window){
    ivec2 size;
    SDL_GetWindowSizeInPixels(window, &size.x, &size.y);
    return size;
}

export ivec2 get_monitor_size_in_pixels(SDL_Window *window){
    SDL_DisplayID display = SDL_GetDisplayForWindow(window);
    const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(display);

    if (mode == nullptr) {
        std::println("Error: couldn't get display mode. SDL error: {}", SDL_GetError());
        std::println("Falling back to window size.");
        return get_window_size_in_pixels(window);
    }
    return{
        std::ceil(mode->w * mode->pixel_density),
        std::ceil(mode->h * mode->pixel_density),
    };
}

//Assumes a right handed coordinate system. Output Z ranges from 0 to 1, with close objects at 1.
export fmat4 perspective_projection(f32 horizontal_fov_in_degrees, f32 horizontal_to_vertical_ratio, f32 near_z, f32 far_z){
    // todo numerical precision is there a reason to pass vertical_to_horizontal_ratio or horizontal to vertical? pick whichever has better precision or if it doesn't matter do h/v
    const f32 half_fov = glm::radians(horizontal_fov_in_degrees)/2;
    f32 tan = glm::tan(half_fov);// will it help the compiler if i calculate 1/tan once and then multiply it instead of divide it twice?

    // remember glm is column major so the actual matrix will be the transpose of what this notation would suggest
    return fmat4{
        1/tan,0,0,0,
        0,-horizontal_to_vertical_ratio/tan,0,0,
        0,0,-near_z/(near_z-far_z),-1,
        0,0,(far_z*near_z)/(far_z-near_z),0,
    };
};


class Duration {
    f64 ms = 0.0;
public:
    constexpr Duration() = default;
    explicit constexpr Duration(f64 ms):ms(ms){}

    static constexpr Duration from_ms(f64 ms){
        return Duration(ms);
    }

    static constexpr Duration from_sec(f64 sec){
        return Duration(sec * 1000.0);
    }

    static constexpr Duration from_min(f64 min){
        return Duration(min * 60.0 * 1000.0);
    }

    [[nodiscard]] constexpr f64 to_ms() const { return ms; }
    [[nodiscard]] constexpr f64 to_sec() const { return ms / 1000.0; }
    [[nodiscard]] constexpr f64 to_min() const { return ms / (60.0 * 1000.0); }

    [[nodiscard]] constexpr auto chrono() const{
        return std::chrono::duration<f64, std::milli>(ms);
    }

    constexpr auto operator<=>(const Duration&) const = default;

    constexpr Duration operator+(const Duration& other) const{
        return Duration(ms + other.ms);
    }

    constexpr Duration operator-(const Duration& other) const{
        return Duration(ms - other.ms);
    }

    constexpr Duration& operator+=(const Duration& other){
        ms += other.ms;
        return *this;
    }

    constexpr Duration& operator-=(const Duration& other){
        ms -= other.ms;
        return *this;
    }
};

class Time{
    using Clock = std::chrono::steady_clock;
    Clock::time_point time_point;

public:
    constexpr Time() = default;

    explicit Time(Clock::time_point tp):time_point(tp){}

    [[nodiscard]] static Time now(){
        return Time(Clock::now());
    }

    constexpr auto operator<=>(const Time&) const = default;

    friend Duration operator-(const Time& a, const Time& b){
        return Duration::from_ms(std::chrono::duration<f64, std::milli>(a.time_point - b.time_point) .count());
    }

    friend Time operator+(const Time& t, const Duration& d){
        return Time(std::chrono::time_point_cast<Clock::duration>(t.time_point + d.chrono()));
    }

    friend Time operator-(const Time& t, const Duration& d){
        return Time(std::chrono::time_point_cast<Clock::duration>(t.time_point - d.chrono()));
    }
};

export class Timer {
    Time start_time{};
    std::optional<Time> end_time;

public:
    void start(){
        start_time = Time::now();
        end_time.reset();
    }

    void end(){
        end_time = Time::now();
    }

    [[nodiscard]] bool ended() const{
        return end_time.has_value();
    }

    [[nodiscard]] Duration elapsed() const{
        if (!ended()) return Time::now() - start_time;

        return *end_time - start_time;
    }
};
}

template <>
struct std::formatter<util::Duration> : std::formatter<std::string>{
    auto format(const util::Duration& d, format_context& ctx) const{
        std::string str;

        if (d.to_ms() < 10000.0){
            str = std::format("{:.1f}ms", d.to_ms());
        }
        else if (d.to_sec() < 120.0){
            str = std::format("{:.2f}s", d.to_sec());
        }
        else{
            str = std::format("{:.2f}min", d.to_min());
        }

        return std::formatter<std::string>::format(str, ctx);
    }
};

template <>
struct std::formatter<util::Timer> : std::formatter<std::string>{
    auto format(const util::Timer& t, format_context& ctx) const{
        if (!t.ended()){
            return std::formatter<std::string>::format("Timer not ended",ctx);
        }

        return std::formatter<std::string>::format(std::format("{}", t.elapsed()),ctx);
    }
};

namespace util{
export class LagDetect {
    Duration threshold;
    Timer timer;

public:
    explicit LagDetect(Duration threshold):threshold(threshold){}

    void start(){
        timer.start();
    }

    void end(std::string_view msg){
        timer.end();

        const Duration elapsed = timer.elapsed();

        if (elapsed > threshold){
            std::println("{} ({})", msg, elapsed);
        }
    }
};

export class FrameTimer{
    Time frame_start{};
    Timer frame_timer{};
    Duration frame_duration{};

    f64 ema_fps = 0.0; // EMA = exponential moving average
    bool frame_limit_enabled = true;
    i32 max_fps = 120.0f;

public:
    void begin_frame(){
        frame_start = Time::now();
        frame_timer.start();
    }

    void end_frame(){
        if (Duration target = Duration::from_sec(1.0 / static_cast<f64>(max_fps));
            frame_limit_enabled && frame_timer.elapsed() < target)
        {
            Duration remaining = target - frame_timer.elapsed();

            if (remaining > Duration::from_ms(1.0)){
                std::this_thread::sleep_for((remaining - Duration::from_ms(1.0)).chrono());
            }

            while (Time::now() < frame_start + target) _mm_pause();
        }
        frame_timer.end();
        frame_duration = frame_timer.elapsed();

        f64 instant_fps = 1.0 / frame_duration.to_sec();
        ema_fps = ema_fps * 0.9 + instant_fps * 0.1;
    }

    void imgui(){
        ImGui::Begin("Performance");
        ImGui::Text("%s Frame time", std::format("{}", frame_duration).c_str());
        ImGui::Text("%.1f Instant FPS", 1.0 / frame_duration.to_sec());
        ImGui::Text("%.1f Exponential moving average FPS", ema_fps);
        ImGui::Checkbox("Limit FPS", &frame_limit_enabled);
        ImGui::SliderInt("Max FPS", &max_fps, 5, 1000);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::InputInt("##input", &max_fps);
        ImGui::End();
    }
};

export struct ByteRange{
    u64 offset;
    u64 size;
};

export template<typename T>
u64 get_data_size(const std::vector<T> &vec){
    return std::size(vec) * sizeof(T);
}
}
