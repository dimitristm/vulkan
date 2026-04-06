module;
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <VkBootstrap.h>
#include <vulkan/vk_enum_string_helper.h>

#include <glm/vec2.hpp>

#include <vector>
#include <span>
#include <array>
#include <print>
#include <fstream>
#include <optional>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl3.h"
#include "imgui/imgui_impl_vulkan.h"

// Disable harmless VMA warnings
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Weverything"
#elif defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wall"
#  pragma GCC diagnostic ignored "-Wextra"
#endif

#include <vk_mem_alloc.h>

#if defined(__clang__)
#  pragma clang diagnostic pop
#elif defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif
module vulkanEngine;
//import <vector>;


#ifndef NDEBUG
    const bool use_validation_layers = true;
#else
    const bool use_validation_layers = false;
#endif

const uint64_t timeout_length = 3000000000;

static void warnmeaboutthisifyoucompile(int sdf){
    sdf++;
    std::println("{}", sdf);
}

namespace struct_makers {
static VkCommandBufferSubmitInfo command_buffer_submit_info(VkCommandBuffer cmd){
    return VkCommandBufferSubmitInfo{
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext         = nullptr,
        .commandBuffer = cmd,
        .deviceMask    = 0,
    };
}

static VkSemaphoreSubmitInfo semaphore_submit_info(
    VkPipelineStageFlags2 stageMask,
    VkSemaphore semaphore)
{
    return VkSemaphoreSubmitInfo{
        .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext     = nullptr,
        .semaphore = semaphore,
        .value     = 1,
        .stageMask = stageMask,
        .deviceIndex = 0,
    };
}

static VkSubmitInfo2 submit_info(
    VkCommandBufferSubmitInfo* cmd,
    VkSemaphoreSubmitInfo* signalSemaphoreInfo, // Can be null, in which case we do not signal a semaphore
    VkSemaphoreSubmitInfo* waitSemaphoreInfo)   // Can be null, in which case we do not wait on a semaphore
{
    return VkSubmitInfo2{
        .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext                    = nullptr,
        .flags{},
        .waitSemaphoreInfoCount   = waitSemaphoreInfo == nullptr ? 0u : 1u,
        .pWaitSemaphoreInfos      = waitSemaphoreInfo,
        .commandBufferInfoCount   = 1,
        .pCommandBufferInfos      = cmd,
        .signalSemaphoreInfoCount = signalSemaphoreInfo == nullptr ? 0u : 1u,
        .pSignalSemaphoreInfos    = signalSemaphoreInfo,
    };
}

static VkImageCreateInfo image_create_info(
    VkFormat format,
    VkImageUsageFlags usageFlags,
    VkExtent3D extent,
    VkImageLayout initial_layout)
{
    return VkImageCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext         = nullptr,
        .flags{},
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = format,
        .extent        = extent,
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = usageFlags,
        .sharingMode{},
        .queueFamilyIndexCount{},
        .pQueueFamilyIndices{},
        .initialLayout = initial_layout,
    };
}

static VkRenderingAttachmentInfo attachment_info(
    VkImageView view,
    VkClearValue* clear,
    VkImageLayout layout)
{
    return VkRenderingAttachmentInfo{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext       = nullptr,
        .imageView   = view,
        .imageLayout = layout,
        .resolveMode{},
        .resolveImageView{},
        .resolveImageLayout{},
        .loadOp      = (clear != nullptr) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = (clear != nullptr) ? *clear : VkClearValue{},
    };
}

static VkRenderingInfo rendering_info(
    VkExtent2D renderExtent,
    VkRenderingAttachmentInfo* colorAttachment,
    VkRenderingAttachmentInfo* depthAttachment)
{
    return VkRenderingInfo{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext                = nullptr,
        .flags{},
        .renderArea           = {{0, 0}, renderExtent},
        .layerCount           = 1,
        .viewMask{},
        .colorAttachmentCount = 1,
        .pColorAttachments    = colorAttachment,
        .pDepthAttachment     = depthAttachment,
        .pStencilAttachment   = nullptr,
    };
}

} // End of namespace struct_makers. TODO: remove this namespace and add 'make' to the name of each function.


static VkDescriptorPool create_descriptor_pool(VkDevice device, uint32_t pool_size, uint32_t max_sets){
    const int poolsize_count = 11;
    VkDescriptorPoolSize sizes[poolsize_count] = {
        { .type=VK_DESCRIPTOR_TYPE_SAMPLER,                .descriptorCount=pool_size },
        { .type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount=pool_size },
        { .type=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          .descriptorCount=pool_size },
        { .type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          .descriptorCount=pool_size },
        { .type=VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,   .descriptorCount=pool_size },
        { .type=VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,   .descriptorCount=pool_size },
        { .type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         .descriptorCount=pool_size },
        { .type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         .descriptorCount=pool_size },
        { .type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount=pool_size },
        { .type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, .descriptorCount=pool_size },
        { .type=VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,       .descriptorCount=pool_size },
    };

    VkDescriptorPoolCreateInfo descriptor_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .maxSets = max_sets,
        .poolSizeCount = poolsize_count,
        .pPoolSizes = sizes,
    };

    VkDescriptorPool pool{};
    VK_CHECK(vkCreateDescriptorPool(device, &descriptor_pool_info, nullptr, &pool));
    return pool;
}

static VmaAllocator init_vma_allocator(
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkInstance instance,
    APIVersionVulkan version,
    VmaAllocatorCreateFlagBits flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
    unsigned long preferred_large_heap_block_size = 0) // Default 0 usually means 256 MiB
{
    VmaAllocator allocator{};
    VmaAllocatorCreateInfo allocatorInfo{
        .flags = flags,
        .physicalDevice = physical_device,
        .device = device,
        .preferredLargeHeapBlockSize = preferred_large_heap_block_size,
        .pAllocationCallbacks = nullptr,
        .pDeviceMemoryCallbacks = nullptr,
        .pHeapSizeLimit = nullptr,
        .pVulkanFunctions = nullptr,
        .instance = instance,
        .vulkanApiVersion = version.to_vk_enum(),
        .pTypeExternalMemoryHandleTypes = nullptr,
    };
    VK_CHECK(vmaCreateAllocator(&allocatorInfo, &allocator));
    return allocator;
}

static void destroy_swapchain(VkSwapchainKHR swapchain, VkDevice device, std::vector<VkImageView> &swapchain_image_views){
    for(auto &swapchain_image_view : swapchain_image_views){
        vkDestroyImageView(device, swapchain_image_view, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);
}

VulkanEngine::VulkanEngine(SDL_Window *window){
    vkb::InstanceBuilder builder;
    vkb::Instance vkb_inst = builder.set_app_name("Vulkan App")
        .request_validation_layers(use_validation_layers)
        .add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT)
        .add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT)
        //.add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT)
        //.add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT)
        //.add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT)
        .use_default_debug_messenger()
        .require_api_version(api_version.major, api_version.minor, api_version.patch)
        .build()
        .value();
    this->vk_instance = vkb_inst.instance;
    this->debug_messenger = vkb_inst.debug_messenger;

    SDL_Vulkan_CreateSurface(window, vk_instance, nullptr, &surface);

    // Init physical device
    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.synchronization2 = VK_TRUE;
    features13.dynamicRendering = VK_TRUE;
    features13.maintenance4 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.descriptorIndexing = VK_TRUE;
    features12.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceFeatures features{};
    features.multiDrawIndirect = VK_TRUE;

    vkb::PhysicalDeviceSelector selector{vkb_inst};
    vkb::PhysicalDevice vkb_physical_device = selector
        .set_minimum_version(api_version.major, api_version.minor)
        .set_required_features_13(features13)
        .set_required_features_12(features12)
        .set_required_features(features)
        .set_surface(this->surface)
        .select()
        .value();

    vkb::DeviceBuilder device_builder{vkb_physical_device};
    vkb::Device vkb_device = device_builder.build().value();
    this->physical_device = vkb_physical_device.physical_device;
    this->device = vkb_device.device;
    this->graphics_queue = vkb_device.get_queue(vkb::QueueType::graphics).value();
    this->graphics_queue_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

    if (vkb_device.get_queue(vkb::QueueType::present).value() != graphics_queue){
        std::println("Error: device does not support shared graphics and present queues.");
    }

    allocator = init_vma_allocator(physical_device, device, vk_instance, api_version);

    this->descriptor_pool = create_descriptor_pool(device, 1000, 200);
}

VulkanEngine::~VulkanEngine(){
    VK_CHECK(vkDeviceWaitIdle(device));

    if (imgui_is_initialized){
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
    }

    for(auto &pipeline : created_pipelines) vkDestroyPipeline(device, pipeline, nullptr);
    for(auto &pipeline_layout : created_pipeline_layouts) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    for(auto &shader_module : created_shader_modules) vkDestroyShaderModule(device, shader_module, nullptr);
    for(auto &created_buffer : created_buffers) vmaDestroyBuffer(allocator, created_buffer.buffer, created_buffer.allocation);

    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    for(auto &layout : descriptor_set_layouts_to_delete) vkDestroyDescriptorSetLayout(device, layout, nullptr);
    for(auto &command_pool : created_command_pools) vkDestroyCommandPool(device, command_pool, nullptr);
    for(auto &fence : created_fences) vkDestroyFence(device, fence, nullptr);
    for(auto &sema : created_semaphores) vkDestroySemaphore(device, sema, nullptr);
    for(const auto &img_view : created_image_views) vkDestroyImageView(device, img_view, nullptr);
    for(auto &image : created_images) vmaDestroyImage(allocator, image.image, image.allocation);
    for(auto &swapchain : created_swapchains) destroy_swapchain(swapchain.swapchain, device, swapchain.image_views);

    vmaDestroyAllocator(allocator);
    vkDestroySurfaceKHR(vk_instance, surface, nullptr); // can i destroy surface after device?
    vkDestroyDevice(device, nullptr);
    vkb::destroy_debug_utils_messenger(vk_instance, debug_messenger);
    vkDestroyInstance(vk_instance, nullptr);
}

void VulkanEngine::init_imgui(SDL_Window *window, VkFormat image_format, MSAALevel msaa_level){
    ImGui::CreateContext();
    ImGui_ImplSDL3_InitForVulkan(window);

    // this initializes imgui for Vulkan
    ImGui_ImplVulkan_InitInfo init_info = {
        .ApiVersion = api_version.to_vk_enum(),
        .Instance = vk_instance,
        .PhysicalDevice = physical_device,
        .Device = device,
        .QueueFamily = graphics_queue_family,
        .Queue = graphics_queue,
        .DescriptorPool = nullptr,
        .DescriptorPoolSize = 1000, // Probably overkill
        .MinImageCount = 3,
        .ImageCount = 3,
        .PipelineCache{},
        .PipelineInfoMain{
            .RenderPass{},
            .Subpass{},
            .MSAASamples = static_cast<VkSampleCountFlagBits>(msaa_level),
            .PipelineRenderingCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
                .pNext = nullptr,
                .viewMask{},
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &image_format,
                .depthAttachmentFormat{},
                .stencilAttachmentFormat{},
            },
        },
        .UseDynamicRendering = true,
        .Allocator{},
        .CheckVkResultFn{},
        .MinAllocationSize = 1024L * 1024L, // todo: might be a waste of memory according to imgui devs, but validation layers are unreadable without this here
        .CustomShaderVertCreateInfo{},
        .CustomShaderFragCreateInfo{},
    };
    ImGui_ImplVulkan_Init(&init_info);

    ImGui_ImplSDL3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    this->imgui_is_initialized = true;
}

Image::Image(
    VulkanEngine &vk,
    VkExtent2D extent,
    VkFormat format,
    VkImageUsageFlags image_usage_flags,
    VkMemoryPropertyFlagBits memory_property_flags)
    :
    extent(extent),
    format(format),
    layout(VK_IMAGE_LAYOUT_UNDEFINED)
{
    VkImageCreateInfo img_create_info = struct_makers::image_create_info(format, image_usage_flags, {extent.width, extent.height, 1}, VK_IMAGE_LAYOUT_UNDEFINED);
    VmaAllocationCreateInfo img_alloc_info = {};
    img_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    img_alloc_info.requiredFlags = VkMemoryPropertyFlags(memory_property_flags);
    VkImage image{};
    VmaAllocation allocation{};
    vmaCreateImage(vk.allocator, &img_create_info, &img_alloc_info, &image, &allocation, nullptr);

    this->vk_image = image;
    this->allocation = allocation;

    vk.created_images.push_back({.image=vk_image, .allocation=allocation});
}

Image::Image(VkImage img, VkExtent2D extent, VkFormat format)
:vk_image(img), allocation(nullptr), extent(extent), format(format), layout(VK_IMAGE_LAYOUT_UNDEFINED)
{}

ImageView::ImageView(
    VulkanEngine &vk,
    const Image &img,
    ImageAspects aspects,
    uint32_t base_mip_level,
    uint32_t mip_level_count)
{
    // todo add asserts that make sure the aspect you gave makes sense
        VkImageViewCreateInfo view_info{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext            = nullptr,
        .flags{},
        .image            = img.vk_image,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = img.format,
        .components{},
        .subresourceRange = {
            .aspectMask     = static_cast<VkImageAspectFlags>(aspects),
            .baseMipLevel   = base_mip_level,
            .levelCount     = mip_level_count,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        },
    };
    VK_CHECK(vkCreateImageView(vk.device, &view_info, nullptr, &view));

    vk.created_image_views.insert(view);
}

ImageView::ImageView(VkImageView vk_view):view(vk_view){}


GpuFence::GpuFence(VulkanEngine &vk, bool signaled){
    VkFenceCreateInfo fence_create_info {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0u,
    };
    VK_CHECK(vkCreateFence(vk.device, &fence_create_info, nullptr, &fence));
    vk.created_fences.push_back(fence);
}

void GpuFence::wait(VulkanEngine &vk){
    VK_CHECK(vkWaitForFences(vk.device, 1, &fence, VK_TRUE, timeout_length));
    VK_CHECK(vkResetFences(vk.device, 1, &fence));
}

GpuSemaphore::GpuSemaphore(VulkanEngine &vk){
    VkSemaphoreCreateInfo semaphore_create_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    VK_CHECK(vkCreateSemaphore(vk.device, &semaphore_create_info, nullptr, &semaphore));
    vk.created_semaphores.push_back(semaphore);
}

static void destroy_swapchain(VkSwapchainKHR swapchain, VkDevice device, std::vector<ImageView> &swapchain_image_views){
    for(auto &swapchain_image_view : swapchain_image_views){
        vkDestroyImageView(device, swapchain_image_view.view, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);
}

Swapchain::Swapchain(VulkanEngine &vk, SDL_Window *window, VkPresentModeKHR present_mode) {
    build_swapchain(vk, window, present_mode);
}

[[nodiscard]] uint32_t Swapchain::acquire_next_image(VulkanEngine &vk, GpuSemaphore signal_sema) const{
    uint32_t swapchain_image_index;
    VK_CHECK(vkAcquireNextImageKHR(vk.device, swapchain, timeout_length, signal_sema.semaphore, nullptr, &swapchain_image_index));
    return swapchain_image_index;
}

void Swapchain::present(VulkanEngine &vk, GpuSemaphore wait_sema, uint32_t swapchain_image_index){
    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait_sema.semaphore,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &swapchain_image_index,
        .pResults{},
    };
    VK_CHECK(vkQueuePresentKHR(vk.graphics_queue, &present_info));
}


void Swapchain::build_swapchain(
    VulkanEngine &vk,
    SDL_Window *window,
    VkPresentModeKHR present_mode)
{
    assert((vk.created_swapchains.size() == 0) && "We don't currently support multiple swapchains.");
    glm::ivec2 size;
    SDL_GetWindowSizeInPixels(window, &size.x, &size.y);

    vkb::SwapchainBuilder swapchain_builder{vk.physical_device, vk.device, vk.surface};
    vkb::Swapchain vkb_swapchain = swapchain_builder
        // The combination of VK_FORMAT_B8G8R8A8_UNORM and VK_COLOR_SPACE_SRGB_NONLINEAR_KHR assume that you
        // will write in linear space and then manually encode the image to sRGB (aka do gamma correction)
        // as the last thing before blitting to swapchain and presenting.
        .set_desired_format(VkSurfaceFormatKHR{ .format = VK_FORMAT_B8G8R8A8_UNORM, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        .set_desired_present_mode(present_mode)
        .set_desired_extent(size.x, size.y)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .set_desired_min_image_count(vkb::SwapchainBuilder::TRIPLE_BUFFERING)
        .build()
        .value();

    this->extent = vkb_swapchain.extent;
    this->swapchain = vkb_swapchain.swapchain;

    std::vector<VkImage> vk_images = vkb_swapchain.get_images().value();
    for (const auto &vk_img : vk_images){
        this->images.emplace_back(vk_img, vkb_swapchain.extent, vkb_swapchain.image_format);
    }
    this->image_format = vkb_swapchain.image_format;

    std::vector<VkImageView> vk_image_views = vkb_swapchain.get_image_views().value();
    for(VkImageView vk_view : vk_image_views){
        this->image_views.emplace_back(vk_view);
    }
    vk.created_swapchains.push_back({.swapchain=swapchain, .image_views=vk_image_views});
}

void Swapchain::destroy_this_swapchain(VkDevice device){
    //TODO: we MUST remove the swapchain from the VulkanEngine too...
    destroy_swapchain(swapchain, device, image_views);
}

void Swapchain::rebuild_swapchain(
    VulkanEngine &vk,
    SDL_Window *window,
    VkPresentModeKHR present_mode)
{
    destroy_this_swapchain(vk.device);
    build_swapchain(vk, window, present_mode);
}

CommandPool::CommandPool(VulkanEngine &vk){
    VkCommandPoolCreateInfo command_pool_info{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = vk.graphics_queue_family,
    };
    VK_CHECK(vkCreateCommandPool(vk.device, &command_pool_info, nullptr, &pool));
    vk.created_command_pools.push_back(pool);
}

DescriptorSet:: DescriptorSet(VulkanEngine &vk, VkDescriptorSetLayout layout)
:layout(layout)
{
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.pNext = nullptr;
    alloc_info.descriptorPool = vk.descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout;
    VK_CHECK(vkAllocateDescriptorSets(vk.device, &alloc_info, &set));
}


void DescriptorSet::update(VulkanEngine &vk, uint32_t bind, std::span<ImageView> views){
    std::vector<VkDescriptorImageInfo> img_infos;
    for (auto image_view : views){
        img_infos.push_back({
            .sampler{},
            .imageView = image_view.view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        });
    }

    VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = set,
        .dstBinding = bind,
        .dstArrayElement{},
        .descriptorCount = static_cast<uint32_t>(img_infos.size()),
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = img_infos.data(),
        .pBufferInfo{},
        .pTexelBufferView{},
    };
    vkUpdateDescriptorSets(vk.device, 1, &write, 0, nullptr);
}

void DescriptorSet::update(VulkanEngine &vk, uint32_t bind, ImageView &view){
    update(vk, bind, std::span<ImageView>(&view, 1));
}

Shader::Shader(VulkanEngine &vk, const std::string_view filepath){
    assert(filepath.find(".spv") != std::string_view::npos && "Compiled SPIR-V shaders should have .spv in their name. You either passed an uncompiled shader or your shader does not follow naming convention");
    assert(filepath.data()[filepath.size()] == '\0' && "Error: filepath was not null-terminated string");
    // open the file. With cursor at the end
    std::ifstream file(filepath.data(), std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::println("Error: could not open file {}. Did you misspell the filepath? Do you have the permissions for it?", filepath); // todo: have the program find if it's a name or permission issue
        abort();
    }

    // find what the size of the file is by looking up the location of the cursor
    // because the cursor is at the end, it gives the size directly in bytes
    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read((char*)buffer.data(), fileSize);
    file.close();

    // create a new shader module, using the buffer we loaded
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.codeSize = buffer.size() * sizeof(uint32_t);
    createInfo.pCode = buffer.data();

    if (vkCreateShaderModule(vk.device, &createInfo, nullptr, &module) != VK_SUCCESS) {
        std::println("Error: could not create shader module for shader {}", filepath);
        abort();
    }
    vk.created_shader_modules.push_back(module);
}

static bool push_constants_valid(const std::optional<std::vector<VkPushConstantRange>> &push_constants){
    const int max_push_constant_size_bytes = 128;
    if (push_constants.has_value()){
        for (const auto &pconstant : push_constants.value()){
            if (pconstant.offset + pconstant.size > max_push_constant_size_bytes){
                std::println("Assert failed: push constants go beyond the {} byte limit, making them invalid on some hardware.", max_push_constant_size_bytes);
                std::println("The push constant range in question had an offset of {} and a size of {}, thus writing up to byte {}", pconstant.offset, pconstant.size, pconstant.size + pconstant.offset);
                return false;
            }
            if (pconstant.offset % 4 != 0){
                std::println("Assert failed: a push constant had an offset of {}, which isn't a multiple of 4.", pconstant.offset);
                return false;
            }
            if (pconstant.size % 4 != 0){
                std::println("Assert failed: a push constant had a size of {}, which isn't a multiple of 4.", pconstant.size);
                return false;
            }
            if (pconstant.offset < 0){
                std::println("Assert failed: a push constant had an offset of {}", pconstant.offset);
                return false;
            }
            if (pconstant.size < 1){
                std::println("Assert failed: a push constant had a size of {}", pconstant.size);
                return false;
            }
        }
    }
    return true;
}

PipelineLayout::PipelineLayout(
    VulkanEngine &vk,
    std::initializer_list<DescriptorSet> descriptor_sets,
    const std::optional<std::vector<VkPushConstantRange>> &push_constants)
{
    // These are the minimums required by vulkan 1.3 for all devices, exceeding them would mean not supporting some devices.
    const int max_descriptor_sets_in_shader = 4;
    assert(descriptor_sets.size() <= max_descriptor_sets_in_shader && "Error: over 4 descriptor sets bound to one shader. This would make the shader not run on all hardware.");
    assert(push_constants_valid(push_constants));

    std::array<VkDescriptorSetLayout, max_descriptor_sets_in_shader> desc_set_layouts;
    int i = 0;
    for (const auto &set : descriptor_sets) {
        desc_set_layouts.at(i++) = set.layout;
    }

    VkPipelineLayoutCreateInfo layout_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags{},
        .setLayoutCount = static_cast<uint32_t>(descriptor_sets.size()),
        .pSetLayouts = desc_set_layouts.data(),
        .pushConstantRangeCount = push_constants.has_value() ? static_cast<uint32_t>(push_constants->size()) : 0,
        .pPushConstantRanges = push_constants.has_value() ? push_constants->data() : nullptr,
    };
    VK_CHECK(vkCreatePipelineLayout(vk.device, &layout_create_info, nullptr, &layout));
    vk.created_pipeline_layouts.push_back(layout);
}

PipelineLayout::PipelineLayout(VulkanEngine &vk, DescriptorSet descriptor_set, const std::optional<std::vector<VkPushConstantRange>> &push_constants)
:PipelineLayout(vk, {descriptor_set}, push_constants){}


SpecializationInfo::SpecializationInfo(size_t total_data_size)
:
data_buffer_max_capacity_in_bytes(total_data_size),
data(malloc(data_buffer_max_capacity_in_bytes))
{
    assert(total_data_size >= 4 && "Specialization info data must be at least 4 bytes large");
    assert(total_data_size % 4 == 0 && "It only ever makes sense for the size of a specialization data buffer to be a multiple of 4");
    entries.reserve(data_buffer_max_capacity_in_bytes/sizeof(int32_t));
    if(data == nullptr) { abort(); }
}

SpecializationInfo &SpecializationInfo::reset(){
    finalized = false;
    entries.clear();
    data_size_in_bytes = 0;
    return *this;
}

// DO NOT cache the output of this funciton. If you ever reset this instance of SpecializationInfo or let it be destroyed,
// the returned value of this function becomes invalid. Just pass it to a pipeline creation function and let it go.
const VkSpecializationInfo *SpecializationInfo::get_vk_specialization_info(){
    if (!finalized){
        info =  VkSpecializationInfo{
            .mapEntryCount = static_cast<uint32_t>(entries.size()),
            .pMapEntries = entries.data(),
            .dataSize = data_size_in_bytes,
            .pData = data,
        };
        finalized = true;
    }
    return &info;
}

SpecializationInfo::~SpecializationInfo(){
    free(data);
}

static VkPipelineShaderStageCreateInfo make_pipeline_shader_stage_info(
    Shader shader_module,
    VkShaderStageFlagBits stage,
    SpecializationInfo */* _Nullable */specialization_info)
{
    return VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags{},
        .stage = stage,
        .module = shader_module.module,
        .pName = "main",
        .pSpecializationInfo = specialization_info != nullptr ? specialization_info->get_vk_specialization_info() : nullptr,
    };
}

ComputePipeline::ComputePipeline(
    VulkanEngine &vk,
    ComputeShader shader_module,
    PipelineLayout pipeline_layout,
    SpecializationInfo */* _Nullable */specialization_info)
:layout(pipeline_layout)
{
    auto stageinfo = make_pipeline_shader_stage_info(shader_module, VK_SHADER_STAGE_COMPUTE_BIT, specialization_info);

    VkComputePipelineCreateInfo computePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags{},
        .stage = stageinfo,
        .layout = this->layout.layout,
        .basePipelineHandle{},
        .basePipelineIndex{},
    };
    VK_CHECK(vkCreateComputePipelines(vk.device,VK_NULL_HANDLE,1,&computePipelineCreateInfo, nullptr, &pipeline));
    vk.created_pipelines.push_back(pipeline);
}

VulkanBuffer::VulkanBuffer(
    VulkanEngine &vk,
    uint32_t capacity_in_bytes,
    VkBufferUsageFlags usage_flags,
    VmaAllocationCreateFlags vma_flags,
    VkMemoryPropertyFlags memory_property_flags_required,
    VkMemoryPropertyFlags memory_property_flags_preferred)
    :
    capacity_in_bytes(capacity_in_bytes)
{
    VkBufferCreateInfo buf_create_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags{},
        .size = capacity_in_bytes,
        .usage = usage_flags | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount{},
        .pQueueFamilyIndices{},
    };

    VmaAllocationCreateInfo alloc_create_info{};
    alloc_create_info.flags = vma_flags;
    alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_create_info.requiredFlags = memory_property_flags_required;
    alloc_create_info.preferredFlags = memory_property_flags_preferred;

    VmaAllocationInfo alloc_info;
    vmaCreateBuffer(vk.allocator, &buf_create_info, &alloc_create_info, &buffer, &allocation, &alloc_info);
    vk.created_buffers.push_back({.buffer = buffer, .allocation = allocation});

    VkBufferDeviceAddressInfo buffer_device_adress_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = buffer,
    };
    device_address = vkGetBufferDeviceAddress(vk.device, &buffer_device_adress_info);
}

StorageBuffer::StorageBuffer(VulkanEngine &vk, uint32_t size_in_bytes, bool is_transfer_source, bool is_transfer_dest)
:VulkanBuffer(vk,
                size_in_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | (is_transfer_source ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : 0)
                | (is_transfer_dest ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0),
                0,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
{}

StagingBuffer::StagingBuffer(VulkanEngine &vk, uint64_t size_in_bytes)
:VulkanBuffer(vk,
                size_in_bytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
{
    VmaAllocationInfo alloc_info;
    vmaGetAllocationInfo(vk.allocator, allocation, &alloc_info);
    mapped_data = alloc_info.pMappedData;
}


ReadbackBuffer::ReadbackBuffer(VulkanEngine &vk, uint32_t size_in_bytes)
:VulkanBuffer(vk,
                size_in_bytes,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
{
    VmaAllocationInfo alloc_info;
    vmaGetAllocationInfo(vk.allocator, allocation, &alloc_info);
    mapped_data = alloc_info.pMappedData;
}

IndexBuffer::IndexBuffer(VulkanEngine &vk, uint32_t total_indexes)
:VulkanBuffer(vk,
                total_indexes * sizeof(uint32_t),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                0, 0)
{}


CommandBuffer::CommandBuffer(const VulkanEngine &vk, const CommandPool &pool){
    VkCommandBufferAllocateInfo alloc_info{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = pool.pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VK_CHECK(vkAllocateCommandBuffers(vk.device, &alloc_info, &buffer));
}

void CommandBuffer::draw_imgui(ImageView target_image_view, VkExtent2D draw_extent) const{
    ImGui::Render(); // todo performance: maybe we should call this on another thread while other things are going on

    VkRenderingAttachmentInfo colorAttachment = struct_makers::attachment_info(target_image_view.view, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = struct_makers::rendering_info(draw_extent, &colorAttachment, nullptr);

    vkCmdBeginRendering(this->buffer, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), this->buffer);

    vkCmdEndRendering(this->buffer);

    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

void CommandBuffer::restart(bool one_time_submit) const{
    VK_CHECK(vkResetCommandBuffer(buffer, 0));
    VkCommandBufferBeginInfo begin_info{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = static_cast<VkCommandBufferUsageFlags>(one_time_submit ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : 0),
        .pInheritanceInfo = nullptr,
    };
    VK_CHECK(vkBeginCommandBuffer(buffer, &begin_info));
}

void CommandBuffer::submit(
    VulkanEngine &vk,
    std::optional<GpuSemaphore> wait_sema,
    std::optional<VkPipelineStageFlagBits2> wait_stage_mask, // Commands in the buffer that use these stages will not run until wait_sema is signaled
    std::optional<GpuSemaphore> signal_sema, // Will be signaled when every command in the buffer is complete
    std::optional<VkPipelineStageFlagBits2> signal_stage_mask, // Stages that wait on the signal_sema will have access to writes done by commands in the buffer (other stages will have outdated cached data, causing errors)
    GpuFence signal_fence) const
{
    assert(wait_sema.has_value() == wait_stage_mask.has_value() && "You can't have a wait sema without a wait stage mask or the other way around");
    assert(signal_sema.has_value() == signal_stage_mask.has_value() && "You can't have a signal sema without a signal stage mask or the other way around");

    VkCommandBufferSubmitInfo cmd_info = struct_makers::command_buffer_submit_info(buffer);
    VkSemaphoreSubmitInfo wait_info = wait_sema.has_value() ? struct_makers::semaphore_submit_info(*wait_stage_mask, wait_sema->semaphore) : VkSemaphoreSubmitInfo{};
    VkSemaphoreSubmitInfo signal_info = signal_sema.has_value() ? struct_makers::semaphore_submit_info(*signal_stage_mask, signal_sema->semaphore) : VkSemaphoreSubmitInfo{};
    VkSubmitInfo2 submit_info = struct_makers::submit_info(&cmd_info,
                                                            signal_sema.has_value() ? &signal_info : nullptr,
                                                            wait_sema.has_value() ? &wait_info : nullptr);

    VK_CHECK(vkQueueSubmit2(vk.graphics_queue, 1, &submit_info, signal_fence.fence));
}

void CommandBuffer::submit(VulkanEngine &vk, GpuFence signal_fence) const{
    submit(vk, std::nullopt, std::nullopt, std::nullopt, std::nullopt, signal_fence);
}

void CommandBuffer::copy_buffer(const VulkanBuffer &src, const VulkanBuffer &dst, const std::span<VkBufferCopy> ranges) const{
    vkCmdCopyBuffer(buffer, src.buffer, dst.buffer, ranges.size(), ranges.data());
}

void CommandBuffer::copy_buffer(const VulkanBuffer &src, const VulkanBuffer &dst, VkBufferCopy range) const{
    vkCmdCopyBuffer(buffer, src.buffer, dst.buffer, 1, &range);
}

void CommandBuffer::copy_entire_buffer(const VulkanBuffer &src, const VulkanBuffer &dst) const{
    assert(dst.capacity_in_bytes >= src.capacity_in_bytes);
    VkBufferCopy range{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = src.capacity_in_bytes,
    };
    vkCmdCopyBuffer(buffer, src.buffer, dst.buffer, 1, &range);
}

void CommandBuffer::blit(
    const Image &source,
    const Image &destination,
    glm::ivec2 src_top_left,
    glm::ivec2 src_bottom_right,
    glm::ivec2 dst_top_left,
    glm::ivec2 dst_bottom_right,
    ImageAspects aspects) const // todo add VkCmdCopyImage function for when we don't need to blit
{
    assert(source.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && "Layout must be VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL for performance reasons.");
    assert(destination.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && "Layout must be VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL for performance reasons.");

    VkImageBlit2 blit_region{
        .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
        .pNext = nullptr,

        .srcSubresource{
            .aspectMask = static_cast<VkImageAspectFlags>(aspects),
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .srcOffsets{
            {src_top_left.x, src_top_left.y, 0},
            {src_bottom_right.x, src_bottom_right.y, 1},
        },

        .dstSubresource{
            .aspectMask = static_cast<VkImageAspectFlags>(aspects),
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,

        },
        .dstOffsets{
            {dst_top_left.x, dst_top_left.y, 0},
            {dst_bottom_right.x, dst_bottom_right.y, 1},
        },
    };
    VkBlitImageInfo2 blit_info{
        .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
        .pNext = nullptr,
        .srcImage = source.vk_image,
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstImage = destination.vk_image,
        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .regionCount = 1,
        .pRegions = &blit_region,
        .filter = VK_FILTER_LINEAR,
    };
    vkCmdBlitImage2(this->buffer, &blit_info);
}

void CommandBuffer::blit_entire_images(const Image &source, const Image &destination, ImageAspects aspects) const    {
    blit(source, destination, {0,0}, {source.extent.width, source.extent.height}, {0,0}, {destination.extent.width, destination.extent.height}, aspects);
}

void CommandBuffer::bind_pipeline(const ComputePipeline &pipeline) const{
    vkCmdBindPipeline(this->buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
}

void CommandBuffer::bind_descriptor_sets(const ComputePipeline &pipeline, std::initializer_list<DescriptorSet> sets) const{
    const int max_sets = 4;
    if (sets.size() > max_sets){
        std::println("Error: cannot have more than {} descriptor sets in bind_descriptor_sets.", max_sets);
        abort();
    };
    VkDescriptorSet vk_sets[max_sets];
    int i = 0;
    for(const auto &set : sets){
        vk_sets[i++] = set.set;
    }

    vkCmdBindDescriptorSets(this->buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout.layout, 0, sets.size(), vk_sets, 0, nullptr);
}
void CommandBuffer::bind_descriptor_sets(ComputePipeline pipeline, DescriptorSet set) const{
    bind_descriptor_sets(pipeline, {set});
}

void CommandBuffer::dispatch(uint32_t x, uint32_t y, uint32_t z) const{
    vkCmdDispatch(this->buffer, x, y, z);
}

void CommandBuffer::end() const{
    VK_CHECK(vkEndCommandBuffer(this->buffer));
}

DescriptorSetBuilder &DescriptorSetBuilder::bind(uint32_t binding, VkDescriptorType type, VkShaderStageFlagBits accessible_stages_flags){
    finalized = false;
    VkDescriptorSetLayoutBinding new_bind{
        .binding = binding,
        .descriptorType = type,
        .descriptorCount = 1,
        .stageFlags = accessible_stages_flags,
        .pImmutableSamplers = nullptr,
    };
    bindings.push_back(new_bind);
    return *this;
}

DescriptorSet DescriptorSetBuilder::build(VulkanEngine &vk){
    if (!finalized){
        VkDescriptorSetLayoutCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        };
        VK_CHECK(vkCreateDescriptorSetLayout(vk.device, &info, nullptr, &layout));
        vk.descriptor_set_layouts_to_delete.push_back(layout);
        finalized = true;
    }

    return DescriptorSet{vk, layout};
}

DescriptorSetBuilder &DescriptorSetBuilder::reset(){
    finalized = false;
    layout = {},
    bindings.clear();
    return *this;
}
