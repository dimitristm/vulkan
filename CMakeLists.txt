cmake_minimum_required(VERSION 3.31.6)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_IMPORT_STD "Enable import std. CMake only has experimental support, \
                       and you need to input a special UUID for your specific CMake version. \
                       Also, for whatever reason, this breaks if you initially have it off and \
                       then set it to on (the other way around works fine). \
                       Make a new build directory to enable it."
                       OFF
)
if(USE_IMPORT_STD)
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
    set(CMAKE_CXX_MODULE_STD ON)
endif()

project(vulkan-project
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(WARNING
        "WARNING: This project is only tested with Clang.\n"
        "Compiler in use: ${CMAKE_CXX_COMPILER_ID}\n"
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
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fetch_vk_bootstrap)

FetchContent_Declare(
    VMA
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    GIT_TAG        v3.3.0
    GIT_SHALLOW TRUE
)
set(VMA_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(VMA_BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(VMA)

FetchContent_Declare(
    fastgltf
    GIT_REPOSITORY https://github.com/spnda/fastgltf.git
    GIT_TAG        v0.9.0
    GIT_SHALLOW TRUE
)

set(FASTGLTF_USE_STD_MODULE ${USE_IMPORT_STD} CACHE BOOL "" FORCE)
set(FASTGLTF_COMPILE_AS_CPP20 ON CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_CPP_MODULES ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(fastgltf)

find_package(Vulkan REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# ---- Executable ----
add_executable(vulkan-prog)

target_include_directories(vulkan-prog PRIVATE ${CMAKE_SOURCE_DIR}/vendored)

target_link_libraries(vulkan-prog
    PRIVATE
        Vulkan::Vulkan
        SDL3::SDL3
        vk-bootstrap::vk-bootstrap
        GPUOpen::VulkanMemoryAllocator
        glm::glm
        fastgltf::module
)

target_sources(vulkan-prog
    PRIVATE
        src/main.cpp
        src/models.cpp
        src/vulkan-engine.cpp
        src/vulkan-util.cpp
        vendored/vma_compilation_unit.cpp
        vendored/imgui/imgui.cpp
        vendored/imgui/imgui_impl_sdl3.cpp
        vendored/imgui/imgui_impl_vulkan.cpp
        vendored/imgui/imgui_demo.cpp
        vendored/imgui/imgui_draw.cpp
        vendored/imgui/imgui_tables.cpp
        vendored/imgui/imgui_widgets.cpp

    PRIVATE FILE_SET CXX_MODULES FILES
        src/models.cppm
        src/user-input.cppm
        src/vertex-buffer-attribute-types.cppm
        src/vulkan-engine.cppm
        src/vulkan-renderer.cppm
        src/vulkan-util.cppm
)

target_compile_options(vulkan-prog PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

target_compile_definitions(vulkan-prog PRIVATE
    USE_IMPORT_STD=$<BOOL:${USE_IMPORT_STD}>
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
        COMMAND ${SHADER_COMPILER} --target-env=vulkan1.3 ${SHADER_SOURCE} -o ${SHADER_OUTPUT}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling ${SHADER_NAME}"
        VERBATIM
    )

    list(APPEND SHADER_BINARIES ${SHADER_OUTPUT})
endforeach()

# Create target that builds all shaders
add_custom_target(Shaders ALL DEPENDS ${SHADER_BINARIES})
