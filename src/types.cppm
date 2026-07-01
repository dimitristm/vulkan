module;

#include <vulkan/vulkan_core.h>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#if !VK_PROJ_USE_IMPORT_STD
#include <cassert>
#include <format>
#endif

export module types;
#if VK_PROJ_USE_IMPORT_STD
import std;
#endif

// Integers, floating point
export using i8  = std::int8_t;
export using i16 = std::int16_t;
export using i32 = std::int32_t;
export using i64 = std::int64_t;

export using u8  = std::uint8_t;
export using u16 = std::uint16_t;
export using u32 = std::uint32_t;
export using u64 = std::uint64_t;

export using f32 = float;
export using f64 = double;

// Vectors and matrices
export using fvec2 = glm::vec<2, f32>;
export using fvec3 = glm::vec<3, f32>;
export using fvec4 = glm::vec<4, f32>;

export using dvec2 = glm::vec<2, f64>;
export using dvec3 = glm::vec<3, f64>;
export using dvec4 = glm::vec<4, f64>;

export using i8vec2  = glm::vec<2, i8>;
export using i8vec3  = glm::vec<3, i8>;
export using i8vec4  = glm::vec<4, i8>;

export using i16vec2 = glm::vec<2, i16>;
export using i16vec3 = glm::vec<3, i16>;
export using i16vec4 = glm::vec<4, i16>;

export using ivec2   = glm::vec<2, i32>;
export using ivec3   = glm::vec<3, i32>;
export using ivec4   = glm::vec<4, i32>;

export using i64vec2 = glm::vec<2, i64>;
export using i64vec3 = glm::vec<3, i64>;
export using i64vec4 = glm::vec<4, i64>;

export using u8vec2  = glm::vec<2, u8>;
export using u8vec3  = glm::vec<3, u8>;
export using u8vec4  = glm::vec<4, u8>;

export using u16vec2 = glm::vec<2, u16>;
export using u16vec3 = glm::vec<3, u16>;
export using u16vec4 = glm::vec<4, u16>;

export using uvec2   = glm::vec<2, u32>;
export using uvec3   = glm::vec<3, u32>;
export using uvec4   = glm::vec<4, u32>;

export using u64vec2 = glm::vec<2, u64>;
export using u64vec3 = glm::vec<3, u64>;
export using u64vec4 = glm::vec<4, u64>;

export using bvec2 = glm::bvec2;
export using bvec3 = glm::bvec3;
export using bvec4 = glm::bvec4;

export using fmat2 = glm::mat<2, 2, f32>;
export using fmat3 = glm::mat<3, 3, f32>;
export using fmat4 = glm::mat<4, 4, f32>;

export using fmat2x3 = glm::mat<2, 3, f32>;
export using fmat2x4 = glm::mat<2, 4, f32>;
export using fmat3x2 = glm::mat<3, 2, f32>;
export using fmat3x4 = glm::mat<3, 4, f32>;
export using fmat4x2 = glm::mat<4, 2, f32>;
export using fmat4x3 = glm::mat<4, 3, f32>;

export using dmat2 = glm::mat<2, 2, f64>;
export using dmat3 = glm::mat<3, 3, f64>;
export using dmat4 = glm::mat<4, 4, f64>;

namespace{
template <typename T, typename Tag>
struct strong_type_with_implicit_cast {
    using underlying_type = T;
    using tag_type        = Tag;
    T value;
    constexpr strong_type_with_implicit_cast() = default;

    // Implicit casting:
    constexpr strong_type_with_implicit_cast(T v) : value(v) {}
    constexpr operator T&() noexcept { return value; }
    // constexpr operator T const&() const noexcept { return value; } this line seems redundant
    constexpr operator T() const noexcept { return value; }
};
}

// Tells std::print to print it like it would print the underlying_type
template <typename T, typename Tag, typename CharT>
struct std::formatter<strong_type_with_implicit_cast<T, Tag>, CharT> : std::formatter<T, CharT>{
};

#define STRONG_TYPE_WITH_IMPLICIT_CAST(name, underlying)                               \
    namespace {struct name##___tag_mangle {};};                                        \
    export using name = strong_type_with_implicit_cast<underlying, name##___tag_mangle>\

STRONG_TYPE_WITH_IMPLICIT_CAST(i8_norm_t, i8);
STRONG_TYPE_WITH_IMPLICIT_CAST(i16_norm_t, i16);
STRONG_TYPE_WITH_IMPLICIT_CAST(i32_norm_t, i32);
STRONG_TYPE_WITH_IMPLICIT_CAST(u8_norm_t, u8);
STRONG_TYPE_WITH_IMPLICIT_CAST(u16_norm_t, u16);
STRONG_TYPE_WITH_IMPLICIT_CAST(u32_norm_t, u32);

STRONG_TYPE_WITH_IMPLICIT_CAST(i32_A2R10G10B10_t, i32);
STRONG_TYPE_WITH_IMPLICIT_CAST(i32_A2R10G10B10_norm_t, i32);
STRONG_TYPE_WITH_IMPLICIT_CAST(u32_A2R10G10B10_t, u32);
STRONG_TYPE_WITH_IMPLICIT_CAST(u32_A2R10G10B10_norm_t, u32);


namespace{
template <typename T>
struct norm_vec : public T{
public:
    using T::T;
    constexpr norm_vec(T value) : T(value){}
};
}

// Vertex attribute types
export using ivec2_norm8  = norm_vec<i8vec2>;
export using ivec3_norm8  = norm_vec<i8vec3>;
export using ivec4_norm8  = norm_vec<i8vec4>;
export using ivec2_norm16 = norm_vec<i16vec2>;
export using ivec3_norm16 = norm_vec<i16vec3>;
export using ivec4_norm16 = norm_vec<i16vec4>;
export using uvec2_norm8  = norm_vec<u8vec2>;
export using uvec3_norm8  = norm_vec<u8vec3>;
export using uvec4_norm8  = norm_vec<u8vec4>;
export using uvec2_norm16 = norm_vec<u16vec2>;
export using uvec3_norm16 = norm_vec<u16vec3>;
export using uvec4_norm16 = norm_vec<u16vec4>;
