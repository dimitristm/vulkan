cmake_minimum_required(VERSION 3.31.6)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(vulkan-prog
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

include_directories(${CMAKE_SOURCE_DIR}/include)

# Find dependencies
find_package(Vulkan REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# ---- Executable ----
add_executable(vulkan-prog
    main.cpp
)

target_link_libraries(vulkan-prog
    PRIVATE
        Vulkan::Vulkan
        SDL3::SDL3
)

target_compile_options(vulkan-prog PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
