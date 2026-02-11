cmake_minimum_required(VERSION 3.31.6)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(vulkan-project
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(WARNING
        "\n\033[1;31m"
        "============================================================\n"
        "  WARNING: This project is only tested with Clang.\n"
        "  Compiler in use: ${CMAKE_CXX_COMPILER_ID}\n"
        "  Compiler-specific breakage is possible.\n"
        "============================================================"
        "\033[0m\n"
    )
endif()

# Enable LTO on release builds
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED)

if (LTO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# Dependencies
include(FetchContent)

FetchContent_Declare(
    fetch_vk_bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
    GIT_TAG        v1.4.338
)
FetchContent_MakeAvailable(fetch_vk_bootstrap)

FetchContent_Declare(
    VMA
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    GIT_TAG        v3.3.0
)
set(VMA_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(VMA_BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(VMA)

find_package(Vulkan REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# ---- Executable ----
add_executable(vulkan-prog
    src/main.cpp
    src/vulkan-renderer.cppm
    src/vulkan-renderer2.cppm
    src/vulkan-util.cppm
    vendored/vma_compilation_unit.cpp
    vendored/imgui.cpp
    vendored/imgui_impl_sdl3.cpp
    vendored/imgui_impl_vulkan.cpp
    vendored/imgui_demo.cpp
    vendored/imgui_draw.cpp
    vendored/imgui_tables.cpp
    vendored/imgui_widgets.cpp
)

target_include_directories(vulkan-prog PRIVATE ${CMAKE_SOURCE_DIR}/vendored)

target_link_libraries(vulkan-prog
    PRIVATE
        Vulkan::Vulkan
        SDL3::SDL3
        vk-bootstrap::vk-bootstrap
        GPUOpen::VulkanMemoryAllocator
        glm::glm
)

target_sources(vulkan-prog
        PRIVATE
        FILE_SET CXX_MODULES FILES
        src/vulkan-renderer.cppm
        src/vulkan-renderer2.cppm
        src/vulkan-util.cppm
)

target_compile_options(vulkan-prog PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

# ---- Shaders ----
# todo: also support glslangValidator
# Find the shader compiler
find_program(SHADER_COMPILER glslc REQUIRED)

# Find all shader source files
file(GLOB_RECURSE SHADER_SOURCES
    CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/shaders/*.frag"
    "${PROJECT_SOURCE_DIR}/shaders/*.vert"
    "${PROJECT_SOURCE_DIR}/shaders/*.comp"
)

# Compile each shader to SPIR-V
set(SHADER_BINARIES)
foreach(SHADER_SOURCE ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
    set(SHADER_OUTPUT "${PROJECT_SOURCE_DIR}/shaders/compiled/${SHADER_NAME}.spv")

    add_custom_command(
        OUTPUT ${SHADER_OUTPUT}
        COMMAND ${SHADER_COMPILER} ${SHADER_SOURCE} -o ${SHADER_OUTPUT}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling ${SHADER_NAME}"
        VERBATIM
    )

    list(APPEND SHADER_BINARIES ${SHADER_OUTPUT})
endforeach()

# Create target that builds all shaders
add_custom_target(Shaders ALL DEPENDS ${SHADER_BINARIES})
