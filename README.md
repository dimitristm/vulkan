# Vulkan-based renderer and a custom asset file format to go with it
No graphics libraries, just the Vulkan API.
Creates an abstraction over the Vulkan API, which is then used by the Renderer.
The custom asset format optimizes for rendering quality, loading speed and file size.

## Features:
Renderer features:
* Physically based renderer with Burley Diffuse, Trowbridge-Reitz(GGX) NDF, Smith geometric shadowing
* Forward rendering
* MSAA antialiasing
* Mip mapping, trilinear sampling
* Bindless rendering
* Directional lighting

Vulkan abstraction features:
* Automatic cleaning of vulkan resources
* Swapchains and presentation
* Push constants
* Specialization constants
* Compute and Graphics pipelines
* Basic resources like GPU-side synchronization primitives, gpu buffers, images, descriptor sets, shaders, etc
* Basic operations like barriers, blits, draw calls, uploading memory to gpu
* DearImgui integration

There is also a converter from glb and gltf files to my custom assetpack format, which optimizes textures and meshes in such a way that they can be rendered at a higher quality for less overall memory. It also optimizes loading speed.
More specifically:
*Removes duplicate assets within and across gltf/glb files.
*Generates mips. Uses BC7 on color data and BC5 on normals and roughness/metallic maps.
*Uses meshoptimizer to increase cache locality, reduce overdraw, and discard duplicate vertices
*Uses LZ4 compression on the mips and meshes. The compression is done blocks that fit in L1 cache to increase decompression speed and allow for parallel decompression on the cpu. The blocks are fetched and decompressed in parallel using SDL async IO (wrapper over io_uring) and oneTBB.

# Compiling
Needs a generator that supports modules, like Ninja. Only been tested on clang with mold.

Right now I've only compiled this on Fedora 44, but it should be similar on other platforms.
Dependencies:  
Since it compiles SDL3 itself, it depends on SDL3's dependencies.
They can be found here:  
https://wiki.libsdl.org/SDL3/README-linux  
https://wiki.libsdl.org/SDL3/README-windows

On fedora, also install libstdc++-static. Other distributions include this with the compiler.
You'll also need the vulkan loader, glslc (shader compiler), and optionally (for debugging) the vulkan validation layers:
Ubuntu:
```
sudo apt install libvulkan-dev glslc vulkan-validationlayers clang mold ninja-build
```
On Fedora:
```
sudo dnf install vulkan-loader-devel glslc vulkan-validation-layers-devel libstdc++-static clang mold ninja-build
```

Then build with:
```
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake
cmake --build build-release
```
