cmake_minimum_required(VERSION 3.31.6)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS OFF)

option(VK_PROJ_USE_IMPORT_STD "Enable import std. CMake only has experimental support, \
                       and you need to input a special UUID for your specific CMake version. \
                       Also, for whatever reason, this breaks if you initially have it off and \
                       then set it to on (the other way around works fine). \
                       Make a new build directory to enable it."
                       OFF
)
if(VK_PROJ_USE_IMPORT_STD)
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
    set(CMAKE_CXX_MODULE_STD ON)
endif()

project(vulkan-project
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(TRACY_ENABLE "" OFF)
option(TRACY_ON_DEMAND "" OFF)
option(TRACY_LTO "" ON)
option(TRACY_NO_EXIT "" OFF)
option(TRACY_NO_SAMPLING "" OFF)

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
set(FASTGLTF_USE_STD_MODULE ${VK_PROJ_USE_IMPORT_STD} CACHE BOOL "" FORCE)
set(FASTGLTF_COMPILE_AS_CPP20 ON CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_CPP_MODULES ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(fastgltf)

FetchContent_Declare(
    TBB
    GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
    GIT_TAG        v2023.0.0
    GIT_SHALLOW TRUE
)
set(TBB_TEST OFF CACHE BOOL "" FORCE)
set(TBB_INSTALL OFF CACHE BOOL "" FORCE)
set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TBB4PY_BUILD OFF CACHE BOOL "Disable TBB Python bindings" FORCE)
# set(TBB_STRICT OFF CACHE BOOL "Do not treat compiler warnings as errors" FORCE)
FetchContent_MakeAvailable(TBB)

find_package(Vulkan REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# Vendored libraries that don't have their own CMakeLists.txt
add_library(bc7enc STATIC)
target_sources(bc7enc
    PRIVATE
        vendored/bc7enc/bc7enc.cpp
        vendored/bc7enc/rgbcx.cpp

    PUBLIC FILE_SET CXX_MODULES FILES
        vendored/bc7enc/bc7enc.cppm
)
target_include_directories(bc7enc PUBLIC ${CMAKE_SOURCE_DIR}/vendored/bc7enc)
target_compile_options(bc7enc PRIVATE -w)

add_library(imgui STATIC)
target_sources(imgui
    PRIVATE
        vendored/imgui/imgui.cpp
        vendored/imgui/imgui_impl_sdl3.cpp
        vendored/imgui/imgui_impl_vulkan.cpp
        vendored/imgui/imgui_demo.cpp
        vendored/imgui/imgui_draw.cpp
        vendored/imgui/imgui_tables.cpp
        vendored/imgui/imgui_widgets.cpp

    PUBLIC FILE_SET CXX_MODULES FILES
        vendored/imgui/imgui.cppm
        vendored/imgui/imgui_impl_sdl3.cppm
        vendored/imgui/imgui_impl_vulkan.cppm
)
target_include_directories(imgui PUBLIC ${CMAKE_SOURCE_DIR}/vendored/imgui)

# ---- Executable ----
add_executable(vulkan-prog)

target_include_directories(vulkan-prog
    PRIVATE
        ${CMAKE_SOURCE_DIR}/vendored
        ${CMAKE_SOURCE_DIR}/vendored/tracy/public/tracy
)

target_link_libraries(vulkan-prog
    PRIVATE
        Vulkan::Vulkan
        SDL3::SDL3
        vk-bootstrap::vk-bootstrap
        GPUOpen::VulkanMemoryAllocator
        glm::glm
        fastgltf::module
        TBB::tbb
        bc7enc
        imgui
)

target_sources(vulkan-prog
    PRIVATE
        src/main.cpp
        src/vulkan-engine.cpp
        src/vulkan-util.cpp
        vendored/vma_compilation_unit.cpp
        vendored/stb/stb_image_compilation_unit.cpp
        vendored/xxhash/xxhash.c
        vendored/tracy/public/TracyClient.cpp

    PRIVATE FILE_SET CXX_MODULES FILES
        src/user-input.cppm
        src/types.cppm
        src/vulkan-engine.cppm
        src/vulkan-renderer.cppm
        src/vulkan-util.cppm
        src/util.cppm
        src/assets.cppm
)

target_compile_options(vulkan-prog PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

target_compile_definitions(vulkan-prog PRIVATE
    VK_PROJ_USE_IMPORT_STD=$<BOOL:${VK_PROJ_USE_IMPORT_STD}>
    $<$<BOOL:${TRACY_ENABLE}>:TRACY_ENABLE>
    $<$<BOOL:${TRACY_ON_DEMAND}>:TRACY_ON_DEMAND>
    $<$<BOOL:${TRACY_LTO}>:TRACY_LTO>
    $<$<BOOL:${TRACY_NO_EXIT}>:TRACY_NO_EXIT>
    $<$<BOOL:${TRACY_NO_SAMPLING}>:TRACY_NO_SAMPLING>

    KHRONOS_STATIC
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
