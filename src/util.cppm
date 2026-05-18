module;
#include <glm/vec2.hpp>
#include <glm/trigonometric.hpp>
#include <glm/mat4x4.hpp>
#include <SDL3/SDL_video.h>
#if !USE_IMPORT_STD
#include <chrono>
#include <print>
#include <cmath>
#endif

export module util;
#if USE_IMPORT_STD
import std;
#endif


namespace util{
export class Duration;
export class Time;

export glm::ivec2 get_window_size_in_pixels(SDL_Window *window){
    glm::ivec2 size;
    SDL_GetWindowSizeInPixels(window, &size.x, &size.y);
    return size;
}

export glm::ivec2 get_monitor_size_in_pixels(SDL_Window *window){
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
export glm::mat4 perspective_projection(float horizontal_fov_in_degrees, float horizontal_to_vertical_ratio, float near_z, float far_z){
    // todo numerical precision is there a reason to pass vertical_to_horizontal_ratio or horizontal to vertical? pick whichever has better precision or if it doesn't matter do h/v
    const float half_fov = glm::radians(horizontal_fov_in_degrees)/2;
    float tan = glm::tan(half_fov);// will it help the compiler if i calculate 1/tan once and then multiply it instead of divide it twice?

    // remember glm is column major so the actual matrix will be the transpose of what this notation would suggest
    return glm::mat4{
        1/tan,0,0,0,
        0,-horizontal_to_vertical_ratio/tan,0,0,
        0,0,-near_z/(near_z-far_z),-1,
        0,0,(far_z*near_z)/(far_z-near_z),0,
    };
};


class Duration {
public:
    constexpr Duration() = default;
    explicit constexpr Duration(double ms):ms(ms){}

    static constexpr Duration from_ms(double ms){
        return Duration(ms);
    }

    static constexpr Duration from_sec(double sec){
        return Duration(sec * 1000.0);
    }

    static constexpr Duration from_min(double min){
        return Duration(min * 60.0 * 1000.0);
    }

    [[nodiscard]] constexpr double to_ms() const { return ms; }
    [[nodiscard]] constexpr double to_sec() const { return ms / 1000.0; }
    [[nodiscard]] constexpr double to_min() const { return ms / (60.0 * 1000.0); }

    [[nodiscard]] constexpr auto chrono() const{
        return std::chrono::duration<double, std::milli>(ms);
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

private:
    double ms = 0.0;
};

class Time{
public:
    using Clock = std::chrono::steady_clock;

    constexpr Time() = default;

    explicit Time(Clock::time_point tp):time_point(tp){}

    [[nodiscard]] static Time now(){
        return Time(Clock::now());
    }

    constexpr auto operator<=>(const Time&) const = default;

    friend Duration operator-(const Time& a, const Time& b){
        return Duration::from_ms(std::chrono::duration<double, std::milli>(a.time_point - b.time_point) .count());
    }

    friend Time operator+(const Time& t, const Duration& d){
        return Time(std::chrono::time_point_cast<Clock::duration>(t.time_point + d.chrono()));
    }

    friend Time operator-(const Time& t, const Duration& d){
        return Time(std::chrono::time_point_cast<Clock::duration>(t.time_point - d.chrono()));
    }

private:
    Clock::time_point time_point;
};

export class Timer {
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
        if (!end_time) return Time::now() - start_time;

        return *end_time - start_time;
    }

private:
    Time start_time{};
    std::optional<Time> end_time;
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

private:
    Duration threshold;
    Timer timer;
};
}
