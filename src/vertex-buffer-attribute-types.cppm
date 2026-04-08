module;

#include <vulkan/vulkan_core.h>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#if !USE_IMPORT_STD
#include <cassert>
#include <format>
#endif

export module vertexBufferAttributeTypes;
#if USE_IMPORT_STD
import std;
#endif

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

STRONG_TYPE_WITH_IMPLICIT_CAST(int8_norm_t, int8_t);
STRONG_TYPE_WITH_IMPLICIT_CAST(int16_norm_t, int16_t);
STRONG_TYPE_WITH_IMPLICIT_CAST(int32_norm_t, int32_t);
STRONG_TYPE_WITH_IMPLICIT_CAST(uint8_norm_t, uint8_t);
STRONG_TYPE_WITH_IMPLICIT_CAST(uint16_norm_t, uint16_t);
STRONG_TYPE_WITH_IMPLICIT_CAST(uint32_norm_t, uint32_t);

STRONG_TYPE_WITH_IMPLICIT_CAST(int32_A2R10G10B10_t, int32_t);
STRONG_TYPE_WITH_IMPLICIT_CAST(int32_A2R10G10B10_norm_t, int32_t);
STRONG_TYPE_WITH_IMPLICIT_CAST(uint32_A2R10G10B10_t, uint32_t);
STRONG_TYPE_WITH_IMPLICIT_CAST(uint32_A2R10G10B10_norm_t, uint32_t);


namespace{
template <typename T>
struct norm_vec : public T{
public:
    using T::T;
    constexpr norm_vec(T value) : T(value){}
};
}
export using ivec2_norm8  = norm_vec<glm::i8vec2>;
export using ivec3_norm8  = norm_vec<glm::i8vec3>;
export using ivec4_norm8  = norm_vec<glm::i8vec4>;
export using ivec2_norm16 = norm_vec<glm::i16vec2>;
export using ivec3_norm16 = norm_vec<glm::i16vec3>;
export using ivec4_norm16 = norm_vec<glm::i16vec4>;
export using uvec2_norm8  = norm_vec<glm::u8vec2>;
export using uvec3_norm8  = norm_vec<glm::u8vec3>;
export using uvec4_norm8  = norm_vec<glm::u8vec4>;
export using uvec2_norm16 = norm_vec<glm::u16vec2>;
export using uvec3_norm16 = norm_vec<glm::u16vec3>;
export using uvec4_norm16 = norm_vec<glm::u16vec4>;

