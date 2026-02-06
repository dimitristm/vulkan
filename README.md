# vulkan
Learning vulkan

Depends on the vulkan loader, the vulkan validation layers, 

# Compiling
Needs a generator that supports modules, like Ninja. Only been tested on clang.

Dependencies:
Vulkan loader
Vulkan validation layers
SDL3
glm
Assimp

Get them on Ubuntu:
```
sudo apt install libvulkan-dev vulkan-validationlayers spirv-tools libsdl3-dev libglm-dev libassimp-dev
```
On fedora:
```
sudo dnf install vulkan-loader-devel vulkan-validation-layers-devel SDL3-devel glm-devel assimp-devel
```

Then build with:
```
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake
cmake --build build-release
```
