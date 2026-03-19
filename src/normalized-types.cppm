module;

#include "boost/pfr.hpp"
#include <cassert>
#include <format>
#include <vulkan/vulkan_core.h>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <type_traits>
#include <print>

export module normalizedTypes;

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
    constexpr operator T const&() const noexcept { return value; }
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


template <typename T>
static constexpr VkFormat get_format(const std::string_view name, const T& value) {
    using U = std::remove_cvref_t<T>;

    // This list contains only widely supported formats.
    if constexpr      (std::is_same_v<U, int8_norm_t>)   return VK_FORMAT_R8_SNORM;
    else if constexpr (std::is_same_v<U, int16_norm_t>)  return VK_FORMAT_R16_SNORM;
    else if constexpr (std::is_same_v<U, int8_t>)        return VK_FORMAT_R8_SINT;
    else if constexpr (std::is_same_v<U, int16_t>)       return VK_FORMAT_R16_SINT;
    else if constexpr (std::is_same_v<U, int32_t>)       return VK_FORMAT_R32_SINT;
    else if constexpr (std::is_same_v<U, uint8_norm_t>)  return VK_FORMAT_R8_UNORM;
    else if constexpr (std::is_same_v<U, uint16_norm_t>) return VK_FORMAT_R16_UNORM;
    else if constexpr (std::is_same_v<U, uint8_t>)       return VK_FORMAT_R8_UINT;
    else if constexpr (std::is_same_v<U, uint16_t>)      return VK_FORMAT_R16_UINT;
    else if constexpr (std::is_same_v<U, uint32_t>)      return VK_FORMAT_R32_UINT;
    else if constexpr (std::is_same_v<U, float>)         return VK_FORMAT_R32_SFLOAT;
    //else if constexpr (std::is_same_v<U, double>)        return VK_FORMAT_R64_SFLOAT; unsupported by most hardware (~27% support on gpuinfo.org)

    else if constexpr (std::is_same_v<U, ivec2_norm8>)   return VK_FORMAT_R8G8_SNORM;
    else if constexpr (std::is_same_v<U, ivec2_norm16>)  return VK_FORMAT_R16G16_SNORM;
    else if constexpr (std::is_same_v<U, ivec3_norm8>)   return VK_FORMAT_R8G8B8_SNORM; // ~82% support on gpuinfo.org
    else if constexpr (std::is_same_v<U, ivec3_norm16>)  return VK_FORMAT_R16G16B16_SNORM; // ~82.5%
    else if constexpr (std::is_same_v<U, ivec4_norm8>)   return VK_FORMAT_R8G8B8A8_SNORM;
    else if constexpr (std::is_same_v<U, ivec4_norm16>)  return VK_FORMAT_R16G16B16A16_SNORM;
    else if constexpr (std::is_same_v<U, uvec2_norm8>)   return VK_FORMAT_R8G8_UNORM;
    else if constexpr (std::is_same_v<U, uvec2_norm16>)  return VK_FORMAT_R16G16_UNORM;
    else if constexpr (std::is_same_v<U, uvec3_norm8>)   return VK_FORMAT_R8G8B8_UNORM; // ~82%
    else if constexpr (std::is_same_v<U, uvec3_norm16>)  return VK_FORMAT_R16G16B16_UNORM; // ~82%
    else if constexpr (std::is_same_v<U, uvec4_norm8>)   return VK_FORMAT_R8G8B8A8_UNORM;
    else if constexpr (std::is_same_v<U, uvec4_norm16>)  return VK_FORMAT_R16G16B16A16_UNORM;

    else if constexpr (std::is_same_v<U, glm::i8vec2>)   return VK_FORMAT_R8G8_SINT;
    else if constexpr (std::is_same_v<U, glm::i8vec3>)   return VK_FORMAT_R8G8B8_SINT; // ~82%
    else if constexpr (std::is_same_v<U, glm::i8vec4>)   return VK_FORMAT_R8G8B8A8_SINT;
    else if constexpr (std::is_same_v<U, glm::i16vec2>)  return VK_FORMAT_R16G16_SINT;
    else if constexpr (std::is_same_v<U, glm::i16vec3>)  return VK_FORMAT_R16G16B16_SINT; // ~78%
    else if constexpr (std::is_same_v<U, glm::i16vec4>)  return VK_FORMAT_R16G16B16A16_SINT;
    else if constexpr (std::is_same_v<U, glm::ivec2>)    return VK_FORMAT_R32G32_SINT;
    else if constexpr (std::is_same_v<U, glm::ivec3>)    return VK_FORMAT_R32G32B32_SINT;
    else if constexpr (std::is_same_v<U, glm::ivec4>)    return VK_FORMAT_R32G32B32A32_SINT;
    else if constexpr (std::is_same_v<U, glm::u8vec2>)   return VK_FORMAT_R8G8_UINT;
    else if constexpr (std::is_same_v<U, glm::u8vec3>)   return VK_FORMAT_R8G8B8_UINT; // ~82%
    else if constexpr (std::is_same_v<U, glm::u8vec4>)   return VK_FORMAT_R8G8B8A8_UINT;
    else if constexpr (std::is_same_v<U, glm::u16vec2>)  return VK_FORMAT_R16G16_UINT;
    else if constexpr (std::is_same_v<U, glm::u16vec3>)  return VK_FORMAT_R16G16B16_UINT; // ~78%
    else if constexpr (std::is_same_v<U, glm::u16vec4>)  return VK_FORMAT_R16G16B16A16_UINT;
    else if constexpr (std::is_same_v<U, glm::uvec2>)    return VK_FORMAT_R32G32_UINT;
    else if constexpr (std::is_same_v<U, glm::uvec3>)    return VK_FORMAT_R32G32B32_UINT;
    else if constexpr (std::is_same_v<U, glm::uvec4>)    return VK_FORMAT_R32G32B32A32_UINT;

    else if constexpr (std::is_same_v<U, glm::vec2>)     return VK_FORMAT_R32G32_SFLOAT;
    else if constexpr (std::is_same_v<U, glm::vec3>)     return VK_FORMAT_R32G32B32_SFLOAT;
    else if constexpr (std::is_same_v<U, glm::vec4>)     return VK_FORMAT_R32G32B32A32_SFLOAT;


    std::println("Your vertex struct conainted a field named '{}', which has a type that isn't supported as a vertex attribute.", name);
    assert(false);
    abort();
    return VK_FORMAT_UNDEFINED;
}
template <typename T>
static void get_attribute_formats(const T &value){
    std::vector<VkFormat> formats;
    boost::pfr::for_each_field_with_name(value, [](std::string_view name, const auto &value){
        get_format(name, value);
    });
}
