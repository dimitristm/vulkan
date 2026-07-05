module;
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_vulkan.h>

#include <VkBootstrap.h>
#include <vulkan/vk_enum_string_helper.h>

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


#if !VK_PROJ_USE_IMPORT_STD
#include <vector>
#include <span>
#include <array>
#include <print>
#include <fstream>
#include <optional>
#include <utility>
#endif

module vulkanEngine;
#if VK_PROJ_USE_IMPORT_STD
import std;
#endif
import types;

import imgui_impl_sdl3;
import imgui_impl_vulkan;
import glm;

void VK_CHECK(VkResult result){
    if(result != VK_SUCCESS){
        std::println("Vulkan error: {}", string_VkResult(result));
        abort();
    }
}

#ifndef NDEBUG
    const bool use_validation_layers = true;
#else
    const bool use_validation_layers = false;
#endif

const uint64_t timeout_length = 3000000000;

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

static VkSubmitInfo2 submit_info2(
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

static VkRenderingAttachmentInfo rendering_attachment_info(
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
        .renderArea           = {.offset={.x=0, .y=0}, .extent=renderExtent},
        .layerCount           = 1,
        .viewMask{},
        .colorAttachmentCount = 1,
        .pColorAttachments    = colorAttachment,
        .pDepthAttachment     = depthAttachment,
        .pStencilAttachment   = nullptr,
    };
}

static VkBufferImageCopy2 buffer_image_copy2(
    const Image &image,
    uint64_t buffer_offset,
    u32 mip_level,
    u32 base_layer,
    u32 layer_count,
    ImageAspects aspects,
    VkOffset3D img_offset)
{
    assert(base_layer + layer_count <= image.layer_count);
    const auto &base_extent = image.extent;
    return VkBufferImageCopy2{
        .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        .pNext = nullptr,
        .bufferOffset = buffer_offset,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource{
            .aspectMask = static_cast<VkImageAspectFlags>(aspects),
            .mipLevel = mip_level,
            .baseArrayLayer = base_layer,
            .layerCount = layer_count,
        },
        .imageOffset = img_offset,
        .imageExtent = {
            .width  = std::max(1u, base_extent.width >> mip_level),
            .height = std::max(1u, base_extent.height >> mip_level),
            .depth  = 1,
        },
    };
}

static VkCopyBufferToImageInfo2 copy_buffer_to_image_info2(
    const VulkanBuffer &buffer,
    const Image &image,
    const VkBufferImageCopy2 *regions,
    u32 region_count,
    VkImageLayout layout)
{
    return VkCopyBufferToImageInfo2{
        .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
        .pNext = nullptr,
        .srcBuffer = buffer.buffer,
        .dstImage = image.vk_image,
        .dstImageLayout = layout,
        .regionCount = region_count,
        .pRegions = regions,
    };
}

static VkCopyImageToBufferInfo2 copy_image_to_buffer_info2(
    const Image &image,
    const VulkanBuffer &buffer,
    const VkBufferImageCopy2 *regions,
    u32 region_count,
    VkImageLayout layout)
{
    return VkCopyImageToBufferInfo2{
        .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
        .pNext = nullptr,
        .srcImage = image.vk_image,
        .srcImageLayout = layout,
        .dstBuffer = buffer.buffer,
        .regionCount = region_count,
        .pRegions = regions,
    };
}

} // End of namespace struct_makers.


bool PushConstantsBuilder::range_stages_do_not_overlap(const std::vector<VkPushConstantRange>& old, VkShaderStageFlags new_flags){
    for (const auto &range : old){
        if ((bool)(range.stageFlags & new_flags)){
            // todo print which stage we're talking about
            std::println("Assert failed: specified multiple push constant ranges for the same shader stage. Are you trying to make push constants for multiple pipelines? You'll have to use a PushConstantsBuilder for each pipeline that has unique push constant ranges.");
            return false;
        }
    }
    return true;
}

PushConstantsBuilder &PushConstantsBuilder::reset(){
    current_last_byte_used = 0;
    ranges.clear();
    return *this;
}

static VkDescriptorPool create_descriptor_pool(VkDevice device, u32 pool_size, u32 max_sets){
    const i32 poolsize_count = 11;
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
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
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

// The present fences are waited on so it's safe to delete the swapchain, but the fences themselves
// are not deleted. That should happen independently.
static void destroy_swapchain(const VulkanEngine &vk,
                              VkSwapchainKHR swapchain,
                              const std::vector<VkImageView> &swapchain_image_views,
                              std::vector<VkFence> &present_fences){
    GpuFence::wait(vk, std::span(present_fences), true);
    for (const auto &swapchain_image_view : swapchain_image_views){
        vkDestroyImageView(vk.device, swapchain_image_view, nullptr);
    }
    vkDestroySwapchainKHR(vk.device, swapchain, nullptr);
}

VulkanEngine::VulkanEngine(SDL_Window *window){
    vkb::InstanceBuilder builder;
    std::println("validation layers = {}", use_validation_layers);
    vkb::Instance vkb_inst = builder.set_app_name("Vulkan App")
        .request_validation_layers(use_validation_layers)
        .enable_extension(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME)
        .enable_extension(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME)
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
    VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT EXT_swapchain_maintenance1{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT,
        .pNext = nullptr,
        .swapchainMaintenance1 = VK_TRUE,
    };

    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.synchronization2 = VK_TRUE;
    features13.dynamicRendering = VK_TRUE;
    features13.maintenance4 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.descriptorIndexing = VK_TRUE;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    features12.descriptorBindingPartiallyBound = VK_TRUE;
    features12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;

    VkPhysicalDeviceFeatures features{};
    features.multiDrawIndirect = VK_TRUE;

    vkb::PhysicalDeviceSelector selector{vkb_inst};
    vkb::PhysicalDevice vkb_physical_device = selector
        .set_minimum_version(api_version.major, api_version.minor)
        .set_required_features_13(features13)
        .set_required_features_12(features12)
        .set_required_features(features)
        .add_required_extension(VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME)
        .set_surface(this->surface)
        .select()
        .value();

    vkb::DeviceBuilder device_builder{vkb_physical_device};
    vkb::Device vkb_device = device_builder.add_pNext(&EXT_swapchain_maintenance1).build().value();
    this->physical_device = vkb_physical_device.physical_device;
    this->device = vkb_device.device;
    this->graphics_queue = vkb_device.get_queue(vkb::QueueType::graphics).value();
    this->graphics_queue_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

    if (vkb_device.get_queue(vkb::QueueType::present).value() != graphics_queue){
        std::println("Error: device does not support shared graphics and present queues.");
    }

    allocator = init_vma_allocator(physical_device, device, vk_instance, api_version);

    this->descriptor_pool = create_descriptor_pool(device, 2000, 200);
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
    for(auto &sema : created_semaphores) vkDestroySemaphore(device, sema, nullptr);
    for(const auto &img_view : created_image_views) vkDestroyImageView(device, img_view, nullptr);
    for(const auto &image : created_images) vmaDestroyImage(allocator, image.image, image.allocation);
    for(auto &sampler : created_samplers) vkDestroySampler(device, sampler, nullptr);
    for(auto &swapchain : created_swapchains) destroy_swapchain(*this, swapchain.swapchain, swapchain.image_views, swapchain.present_fences);
    for(auto &fence : created_fences) vkDestroyFence(device, fence, nullptr);

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
            .ExtraDynamicStates{},
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

    ImVec4* colors = ImGui::GetStyle().Colors;
    colors[ImGuiCol_WindowBg]               = ImVec4(0.00f, 0.00f, 0.00f, 0.94f);
    colors[ImGuiCol_TitleBg]                = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_TitleBgActive]          = ImVec4(0.03f, 0.05f, 0.08f, 1.00f);

    ImGui::NewFrame();

    this->imgui_is_initialized = true;
}

VkSurfaceCapabilitiesKHR VulkanEngine::get_surface_capabilities() const{
    VkSurfaceCapabilitiesKHR capabilities;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities));
    return capabilities;
}

Sampler::Sampler(VulkanEngine &vk,
                 VkFilter mag_filter,
                 VkFilter min_filter,
                 VkSamplerMipmapMode mipmap_mode,
                 VkBool32 anisotropy_enable,
                 f32 max_anisotropy,
                 VkSamplerAddressMode address_modeU,
                 VkSamplerAddressMode address_modeV,
                 VkBool32 compare_enable,
                 VkCompareOp compare_op,
                 f32 mip_lod_bias,
                 f32 min_lod,
                 f32 max_lod,
                 VkBorderColor border_color,
                 VkBool32 unnormalized_coordinates,
                 VkSamplerAddressMode address_modeW)
{
    VkSamplerCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags{},
        .magFilter = mag_filter,
        .minFilter = min_filter,
        .mipmapMode = mipmap_mode,
        .addressModeU = address_modeU,
        .addressModeV = address_modeV,
        .addressModeW = address_modeW,
        .mipLodBias = mip_lod_bias,
        .anisotropyEnable = anisotropy_enable,
        .maxAnisotropy = max_anisotropy,
        .compareEnable = compare_enable,
        .compareOp = compare_op,
        .minLod = min_lod,
        .maxLod = max_lod,
        .borderColor = border_color,
        .unnormalizedCoordinates = unnormalized_coordinates,
    };
    VK_CHECK(vkCreateSampler(vk.device, &info, nullptr, &this->sampler));
    vk.created_samplers.push_back(sampler);
}

Image::Image(
    VulkanEngine &vk,
    VkExtent2D extent,
    VkFormat format,
    VkImageUsageFlags image_usage_flags,
    VkMemoryPropertyFlagBits memory_property_flags,
    MSAALevel msaa_level,
    u32 mip_level_count,
    u32 layer_count)
    :
    extent(extent),
    format(format),
    layer_count(layer_count)
{
    assert(mip_level_count > 0 && "Minimum mip level is 1, not 0.");
    assert(extent.width > 0 && extent.height > 0);
    VkImageCreateInfo img_create_info{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext         = nullptr,
        .flags{},
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = format,
        .extent        = {.width=extent.width, .height=extent.height, .depth=1},
        .mipLevels     = mip_level_count,
        .arrayLayers   = layer_count,
        .samples       = static_cast<VkSampleCountFlagBits>(msaa_level),
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = image_usage_flags,
        .sharingMode{},
        .queueFamilyIndexCount{},
        .pQueueFamilyIndices{},
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VmaAllocationCreateInfo img_alloc_info = {};
    img_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    img_alloc_info.requiredFlags = VkMemoryPropertyFlags(memory_property_flags);
    VkImage image{};
    VmaAllocation allocation{};
    vmaCreateImage(vk.allocator, &img_create_info, &img_alloc_info, &image, &allocation, nullptr);

    this->vk_image = image;
    this->allocation = allocation;

    vk.created_images.insert({.image=vk_image, .allocation=allocation});
}

Image::Image(VkImage img, VkExtent2D extent, VkFormat format, u32 layer_count)
:vk_image(img), allocation(nullptr), extent(extent), format(format), mip_level_count(1), layer_count(layer_count)
{}

void Image::erase_self(VulkanEngine &vk){
    vmaDestroyImage(vk.allocator, vk_image, allocation);
    vk.created_images.erase({.image=vk_image, .allocation=allocation});
}

ImageView::ImageView(
    VulkanEngine &vk,
    const Image &img,
    ImageAspects aspects,
    u32 base_mip_level,
    u32 mip_level_count)
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
            .layerCount     = img.layer_count,
        },
    };
    VK_CHECK(vkCreateImageView(vk.device, &view_info, nullptr, &view));

    vk.created_image_views.insert(view);
}

ImageView::ImageView(VkImageView vk_view):view(vk_view){}

void ImageView::erase_self(VulkanEngine &vk) const{
    vkDestroyImageView(vk.device, view, nullptr);
    vk.created_image_views.erase(view);
}

GpuFence::GpuFence(VulkanEngine &vk, bool signaled){
    VkFenceCreateInfo fence_create_info {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0u,
    };
    VK_CHECK(vkCreateFence(vk.device, &fence_create_info, nullptr, &fence));
    vk.created_fences.push_back(fence);
}

void GpuFence::wait(const VulkanEngine &vk) const{
    VK_CHECK(vkWaitForFences(vk.device, 1, &fence, VK_TRUE, timeout_length));
}

void GpuFence::reset(const VulkanEngine &vk) const{
    VK_CHECK(vkResetFences(vk.device, 1, &fence));
}

void GpuFence::wait_and_reset(const VulkanEngine &vk) const {
    wait(vk);
    reset(vk);
}

bool GpuFence::is_signaled(const VulkanEngine &vk) const {
    const auto result = vkGetFenceStatus(vk.device, this->fence);
    if (result == VK_SUCCESS) return true;
    else if (result == VK_NOT_READY) return false;
    else VK_CHECK(result);
    std::unreachable();
}

void GpuFence::wait(const VulkanEngine &vk, std::span<VkFence> fences, bool wait_all){
    assert(fences.size() > 0 && "The spec forbids calling vkWaitForFences with a fence count of 0");
    VK_CHECK(vkWaitForFences(vk.device, fences.size(), fences.data(), static_cast<VkBool32>(wait_all), timeout_length));
}

void GpuFence::reset(const VulkanEngine &vk, std::span<VkFence> fences){
    assert(fences.size() > 0 && "The spec forbids calling vkResetFences with a fence count of 0");
    VK_CHECK(vkResetFences(vk.device, fences.size(), fences.data()));
}

void GpuFence::wait_and_reset(const VulkanEngine &vk, std::span<VkFence> fences, bool wait_all){
    assert(fences.size() > 0 && "The spec forbids calling vkWaitForFences and vkResetFences with a fence count of 0");
    wait(vk, fences, wait_all);
    reset(vk, fences);
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

Swapchain::Swapchain(VulkanEngine &vk, SDL_Window *window, VkPresentModeKHR present_mode) {
    auto f = GpuFence(vk, true);
    present_fences.reserve(16);
    present_fences.push_back(f.fence);
    initialize_swapchain(vk, window, present_mode);
}

[[nodiscard]] std::tuple<u32, VkResult> Swapchain::acquire_next_image(VulkanEngine &vk, GpuSemaphore signal_sema) const{
    u32 swapchain_image_index;
    VkResult result = vkAcquireNextImageKHR(vk.device, swapchain, timeout_length, signal_sema.semaphore, nullptr, &swapchain_image_index);
    if (result != VK_ERROR_OUT_OF_DATE_KHR && result != VK_SUBOPTIMAL_KHR) VK_CHECK(result);
    return {swapchain_image_index, result};
}

[[nodiscard]] VkResult Swapchain::present(VulkanEngine &vk, GpuSemaphore wait_sema, u32 swapchain_image_index){
    VkFence free_present_fence = VK_NULL_HANDLE;
    for(const auto &fence : present_fences){
        auto f = GpuFence(fence);
        if (f.is_signaled(vk)){
            f.reset(vk);
            free_present_fence = f.fence;
            break;
        }
    }
    if (free_present_fence == VK_NULL_HANDLE){
        auto f = GpuFence(vk, false);
        free_present_fence = f.fence;
        present_fences.push_back(free_present_fence);
        for(auto swap_track : vk.created_swapchains){
            if (swap_track.swapchain == this->swapchain) swap_track.present_fences.emplace_back(free_present_fence);
        }
        assert(present_fences.size() < 8 && "Oddly large amount of present fences, should never be this large as far as I know");
    }
    VkSwapchainPresentFenceInfoKHR swapchain_present_fence_info{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
        .pNext = nullptr,
        .swapchainCount = 1,
        .pFences = &free_present_fence,
    };

    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = &swapchain_present_fence_info,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait_sema.semaphore,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &swapchain_image_index,
        .pResults{},
    };
    VkResult result = vkQueuePresentKHR(vk.graphics_queue, &present_info);
    if (result != VK_ERROR_OUT_OF_DATE_KHR && result != VK_SUBOPTIMAL_KHR) VK_CHECK(result);
    return result;
}

[[nodiscard]] static bool presentable_swapchain_exists_inner(const VkExtent2D &surface_extent, const ivec2 &window_size){
    return surface_extent != VkExtent2D{0, 0} && window_size != ivec2{0,0};
}

[[nodiscard]] bool Swapchain::presentable_swapchain_exists(VulkanEngine &vk, SDL_Window *window){
    VkSurfaceCapabilitiesKHR capabilities = vk.get_surface_capabilities();
    ivec2 window_size = util::get_window_size_in_pixels(window);
    return presentable_swapchain_exists_inner(capabilities.currentExtent, window_size);
}

[[nodiscard]] std::optional<VkExtent2D> Swapchain::decide_extent(VulkanEngine &vk, SDL_Window *window){
    VkSurfaceCapabilitiesKHR capabilities = vk.get_surface_capabilities();
    const ivec2 window_size = util::get_window_size_in_pixels(window);
    const VkExtent2D &surface_extent = capabilities.currentExtent;

    const auto decide_based_on_window_size = [&](){
        const VkExtent2D &max_extent = capabilities.maxImageExtent;
        const VkExtent2D &min_extent = capabilities.minImageExtent;
        return VkExtent2D{
            .width = std::clamp(static_cast<u32>(window_size.x), min_extent.width, max_extent.width),
            .height = std::clamp(static_cast<u32>(window_size.y), min_extent.height, max_extent.height),
        };
    };

    if (!presentable_swapchain_exists_inner(surface_extent, window_size)) return std::nullopt;

    bool program_decides_extent = (surface_extent == VkExtent2D{UINT32_MAX, UINT32_MAX});
    return program_decides_extent ? decide_based_on_window_size() : surface_extent;
}


void Swapchain::initialize_swapchain(
    VulkanEngine &vk,
    SDL_Window *window,
    VkPresentModeKHR present_mode,
    const VkSwapchainKHR *old_swapchain)
{
    std::optional<VkExtent2D> maybe_size = decide_extent(vk, window);
    if (!maybe_size.has_value()){
        auto window_size = util::get_window_size_in_pixels(window);
        auto surface_extent = vk.get_surface_capabilities().currentExtent;
        std::println("Tried to initialize swapchain while either window or surface capabilities has size of {{0,0}}.");
        std::println("window size = {}, {}", window_size.x, window_size.y);
        std::println("surface_extent= {}, {}", surface_extent.width, surface_extent.height);
        abort();
    }
    auto &size = *maybe_size;

    std::array present_modes{
        VK_PRESENT_MODE_FIFO_KHR,
        VK_PRESENT_MODE_MAILBOX_KHR,
    };

    VkSwapchainPresentModesCreateInfoEXT present_modes_create_info{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT,
        .pNext = nullptr,
        .presentModeCount = static_cast<u32>(present_modes.size()),
        .pPresentModes = present_modes.data(),
    };

    VkSwapchainPresentScalingCreateInfoEXT present_scaling_create_info{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_EXT,
        .pNext = nullptr,
        .scalingBehavior = 0,
        .presentGravityX{},
        .presentGravityY{},
    };

    vkb::SwapchainBuilder swapchain_builder{vk.physical_device, vk.device, vk.surface};
    swapchain_builder.set_desired_format(VkSurfaceFormatKHR{ .format = VK_FORMAT_B8G8R8A8_SRGB, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
                     .add_pNext(&present_modes_create_info)
                     .add_pNext(&present_scaling_create_info)
                     .set_desired_present_mode(present_mode)
                     .set_desired_extent(size.width, size.height)
                     .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
                     .set_create_flags(VK_SWAPCHAIN_CREATE_DEFERRED_MEMORY_ALLOCATION_BIT_EXT)
                     .set_desired_min_image_count(vkb::SwapchainBuilder::TRIPLE_BUFFERING);
    if (old_swapchain != nullptr) swapchain_builder.set_old_swapchain(*old_swapchain);
    vkb::Swapchain vkb_swapchain = swapchain_builder.build().value();

    image_views.clear();
    images.clear();

    this->extent = vkb_swapchain.extent;
    this->swapchain = vkb_swapchain.swapchain;

    std::vector<VkImage> vk_images = vkb_swapchain.get_images().value();
    this->images.reserve(vk_images.size());
    for (const auto &vk_img : vk_images){
        this->images.emplace_back(vk_img, vkb_swapchain.extent, vkb_swapchain.image_format, 1);
    }
    this->image_format = vkb_swapchain.image_format;

    std::vector<VkImageView> vk_image_views = vkb_swapchain.get_image_views().value();
    this->image_views.reserve(vk_image_views.size());
    for(VkImageView vk_view : vk_image_views){
        this->image_views.emplace_back(vk_view);
    }
    vk.created_swapchains.push_back({.swapchain=swapchain, .image_views=vk_image_views, .present_fences=present_fences});
}

void Swapchain::rebuild_swapchain(
    VulkanEngine &vk,
    SDL_Window *window,
    VkPresentModeKHR present_mode)
{
    VkSwapchainKHR old_swapchain = this->swapchain;

    initialize_swapchain(vk, window, present_mode, &old_swapchain);
    // Get rid of the old swapchain in the VulkanEngine and destroy its views
    for (int i = 0; const auto &swapchain : vk.created_swapchains){
        if (swapchain.swapchain == old_swapchain) {
            destroy_swapchain(vk, swapchain.swapchain, swapchain.image_views, present_fences);
            vk.created_swapchains.erase(vk.created_swapchains.begin() + i);
            break;
        }
        ++i;
    }
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

DescriptorSet::DescriptorSet(VulkanEngine &vk, VkDescriptorSetLayout layout)
:layout(layout)
{
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.pNext = nullptr;
    alloc_info.descriptorPool = vk.descriptor_pool; // todo dynamically make more descriptor pools if vkAllocateDescriptorSets returns OUT_OF_POOL_MEMORY
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout;
    VK_CHECK(vkAllocateDescriptorSets(vk.device, &alloc_info, &set));
}


void DescriptorSet::update_storage_images(VulkanEngine &vk, u32 bind, std::span<ImageView> views){
    std::vector<VkDescriptorImageInfo> img_infos;
    img_infos.reserve(views.size());
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
        .descriptorCount = static_cast<u32>(img_infos.size()),
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = img_infos.data(),
        .pBufferInfo{},
        .pTexelBufferView{},
    };
    vkUpdateDescriptorSets(vk.device, 1, &write, 0, nullptr);
}

void DescriptorSet::update_storage_image(VulkanEngine &vk, u32 bind, ImageView &view){
    update_storage_images(vk, bind, std::span<ImageView>(&view, 1));
}

void DescriptorSet::update_sampled_images(VulkanEngine &vk, u32 bind, std::span<ImageView> views){
    std::vector<VkDescriptorImageInfo> img_infos;
    img_infos.reserve(views.size());
    for (auto &image_view : views){
        img_infos.push_back({
            .sampler{},
            .imageView = image_view.view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        });
    }

    VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = set,
        .dstBinding = bind,
        .dstArrayElement{},
        .descriptorCount = static_cast<u32>(img_infos.size()),
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .pImageInfo = img_infos.data(),
        .pBufferInfo{},
        .pTexelBufferView{},
    };
    vkUpdateDescriptorSets(vk.device, 1, &write, 0, nullptr);
}

void DescriptorSet::update_sampled_image(VulkanEngine &vk, u32 bind, ImageView &view){
    update_sampled_images(vk, bind, std::span<ImageView>(&view, 1));
}

void DescriptorSet::update_samplers(VulkanEngine &vk, u32 bind, std::span<Sampler> samplers) {
    std::vector<VkDescriptorImageInfo> img_infos;
    img_infos.reserve(samplers.size());

    for (auto sampler : samplers) {
        img_infos.push_back({
            .sampler = sampler.sampler,
            .imageView = VK_NULL_HANDLE,
            .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
    }

    VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = set,
        .dstBinding = bind,
        .dstArrayElement{},
        .descriptorCount = static_cast<u32>(img_infos.size()),
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .pImageInfo = img_infos.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    vkUpdateDescriptorSets(vk.device, 1, &write, 0, nullptr);
}

void DescriptorSet::update_sampler(VulkanEngine &vk, u32 bind, Sampler &sampler){
    update_samplers(vk, bind, std::span<Sampler>(&sampler, 1));
}

void DescriptorSet::update_storage_buffers(VulkanEngine &vk, u32 bind, std::span<StorageBuffer> buffers) {
    std::vector<VkDescriptorBufferInfo> buf_infos;
    buf_infos.reserve(buffers.size());
    for (auto &buffer : buffers) {
        buf_infos.push_back({
            .buffer = buffer.get_vk_buffer(),
            .offset = 0,
            .range = buffer.capacity_in_bytes,
        });
    }
    VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = set,
        .dstBinding = bind,
        .dstArrayElement = 0,
        .descriptorCount = static_cast<u32>(buf_infos.size()),
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = nullptr,
        .pBufferInfo = buf_infos.data(),
        .pTexelBufferView = nullptr,
    };
    vkUpdateDescriptorSets(vk.device, 1, &write, 0, nullptr);
}

void DescriptorSet::update_storage_buffer(VulkanEngine &vk, u32 bind, StorageBuffer &buffer) {
    update_storage_buffers(vk, bind, std::span<StorageBuffer>(&buffer, 1));
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
    std::vector<u32> buffer(fileSize / sizeof(u32));
    file.seekg(0);
    file.read((char*)buffer.data(), fileSize);
    file.close();

    // create a new shader module, using the buffer we loaded
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.codeSize = buffer.size() * sizeof(u32);
    createInfo.pCode = buffer.data();

    if (vkCreateShaderModule(vk.device, &createInfo, nullptr, &module) != VK_SUCCESS) {
        std::println("Error: could not create shader module for shader {}", filepath);
        abort();
    }
    vk.created_shader_modules.push_back(module);
}

static bool push_constants_valid(const std::optional<std::vector<VkPushConstantRange>> &push_constants){
    const i32 max_push_constant_size_bytes = 128;
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
    const i32 max_descriptor_sets_in_shader = 4;
    assert(descriptor_sets.size() <= max_descriptor_sets_in_shader && "Error: over 4 descriptor sets bound to one shader. This would make the shader not run on all hardware.");
    assert(push_constants_valid(push_constants));

    std::array<VkDescriptorSetLayout, max_descriptor_sets_in_shader> desc_set_layouts;
    i32 i = 0;
    for (const auto &set : descriptor_sets) {
        desc_set_layouts.at(i++) = set.layout;
    }

    VkPipelineLayoutCreateInfo layout_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags{},
        .setLayoutCount = static_cast<u32>(descriptor_sets.size()),
        .pSetLayouts = desc_set_layouts.data(),
        .pushConstantRangeCount = push_constants.has_value() ? static_cast<u32>(push_constants->size()) : 0,
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
    entries.reserve(data_buffer_max_capacity_in_bytes/sizeof(i32));
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
            .mapEntryCount = static_cast<u32>(entries.size()),
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
    uint64_t capacity_in_bytes,
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

StorageBuffer::StorageBuffer(VulkanEngine &vk, uint64_t size_in_bytes, bool is_transfer_source, bool is_transfer_dest)
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


ReadbackBuffer::ReadbackBuffer(VulkanEngine &vk, uint64_t size_in_bytes)
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

IndexBuffer::IndexBuffer(VulkanEngine &vk, uint64_t total_indexes)
:VulkanBuffer(vk,
                total_indexes * sizeof(u32),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                0, 0)
{}

static VkCommandBufferAllocateInfo make_VkCommandBufferAllocateInfo(const CommandPool &pool, u32 command_buffer_count){
    return VkCommandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = pool.pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = command_buffer_count,
    };
}

CommandBuffer::CommandBuffer(const VulkanEngine &vk, const CommandPool &pool){
    auto alloc_info = make_VkCommandBufferAllocateInfo(pool, 1);
    VK_CHECK(vkAllocateCommandBuffers(vk.device, &alloc_info, &buffer));
}

void CommandBuffer::make_command_buffers(const VulkanEngine &vk, std::vector<CommandBuffer> &buffers, const CommandPool &pool, i32 how_many_buffers_to_append){
    auto alloc_info = make_VkCommandBufferAllocateInfo(pool, how_many_buffers_to_append);
    thread_local std::vector<VkCommandBuffer> vk_buffers;
    vk_buffers.resize(how_many_buffers_to_append);
    buffers.reserve(buffers.size() + how_many_buffers_to_append);

    VK_CHECK(vkAllocateCommandBuffers(vk.device, &alloc_info, vk_buffers.data()));
    for (int i = 0; i < how_many_buffers_to_append; ++i){
        buffers.emplace_back(vk_buffers[i]);
    }
}

void CommandBuffer::draw_imgui(ImageView target_image_view, VkExtent2D draw_extent) const{
    ImGui::Render(); // todo performance: maybe we should call this on another thread while other things are going on

    VkRenderingAttachmentInfo colorAttachment = struct_makers::rendering_attachment_info(target_image_view.view, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
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
    VkSubmitInfo2 submit_info = struct_makers::submit_info2(&cmd_info,
                                                            signal_sema.has_value() ? &signal_info : nullptr,
                                                            wait_sema.has_value() ? &wait_info : nullptr);

    VK_CHECK(vkQueueSubmit2(vk.graphics_queue, 1, &submit_info, signal_fence.fence));
}

void CommandBuffer::submit(VulkanEngine &vk, GpuFence signal_fence) const{
    submit(vk, std::nullopt, std::nullopt, std::nullopt, std::nullopt, signal_fence);
}

void CommandBuffer::copy_buffer(const VulkanBuffer &src, const VulkanBuffer &dst, const std::span<VkBufferCopy2> ranges) const{
    VkCopyBufferInfo2 copy_info{
        .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
        .pNext = nullptr,
        .srcBuffer = src.buffer,
        .dstBuffer = dst.buffer,
        .regionCount = static_cast<u32>(ranges.size()),
        .pRegions = ranges.data()
    };
    vkCmdCopyBuffer2(buffer, &copy_info);
}

void CommandBuffer::copy_buffer(const VulkanBuffer &src, const VulkanBuffer &dst, VkBufferCopy2 &range) const{
    copy_buffer(src, dst, std::span<VkBufferCopy2>(&range, 1));
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

//The image layout MUST be VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL.
void CommandBuffer::copy_buffer_to_image(
    const VulkanBuffer &buffer,
    const Image &image,
    uint64_t buffer_offset,
    u32 mip_level,
    u32 base_layer,
    u32 layer_count,
    ImageAspects aspects,
    const VkOffset3D &img_offset = VkOffset3D{}) const
{
    VkBufferImageCopy2 region = struct_makers::buffer_image_copy2(image, buffer_offset, mip_level, base_layer, layer_count, aspects, img_offset);
    VkCopyBufferToImageInfo2 buffer_image_copy = struct_makers::copy_buffer_to_image_info2(buffer, image, &region, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdCopyBufferToImage2(this->buffer, &buffer_image_copy);
}

//The image layout MUST be VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL.
void CommandBuffer::copy_buffer_to_image(const VulkanBuffer &buffer, const Image &image, const VkBufferImageCopy2 &region) const{
    VkCopyBufferToImageInfo2 buffer_image_copy = struct_makers::copy_buffer_to_image_info2(buffer, image, &region, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdCopyBufferToImage2(this->buffer, &buffer_image_copy);
}

//The image layout MUST be VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL.
void CommandBuffer::copy_buffer_to_image(const VulkanBuffer &buffer, const Image &image, std::span<VkBufferImageCopy2> regions) const{
    VkCopyBufferToImageInfo2 buffer_image_copy = struct_makers::copy_buffer_to_image_info2(buffer, image, regions.data(), regions.size(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdCopyBufferToImage2(this->buffer, &buffer_image_copy);
}

//The image layout MUST be VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.
void CommandBuffer::copy_image_to_buffer(
    const Image &image,
    const VulkanBuffer &buffer,
    uint64_t buffer_offset,
    u32 mip_level,
    u32 base_layer,
    u32 layer_count,
    ImageAspects aspects,
    const VkOffset3D &img_offset = VkOffset3D{}) const
{
    VkBufferImageCopy2 region = struct_makers::buffer_image_copy2(image, buffer_offset, mip_level, base_layer, layer_count, aspects, img_offset);
    VkCopyImageToBufferInfo2 image_buffer_copy = struct_makers::copy_image_to_buffer_info2(image, buffer, &region, 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdCopyImageToBuffer2(this->buffer, &image_buffer_copy);
}

//The image layout MUST be VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.
void CommandBuffer::copy_image_to_buffer( const Image &image, const VulkanBuffer &buffer, const VkBufferImageCopy2 &region) const
{
    VkCopyImageToBufferInfo2 image_buffer_copy = struct_makers::copy_image_to_buffer_info2(image, buffer, &region, 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdCopyImageToBuffer2(this->buffer, &image_buffer_copy);
}

//The image layout MUST be VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.
void CommandBuffer::copy_image_to_buffer( const Image &image, const VulkanBuffer &buffer, std::span<VkBufferImageCopy2> regions) const
{
    VkCopyImageToBufferInfo2 image_buffer_copy = struct_makers::copy_image_to_buffer_info2(image, buffer, regions.data(), regions.size(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdCopyImageToBuffer2(this->buffer, &image_buffer_copy);
}

//The image layout MUST be VK_IMAGE_LAYOUT_TRANSFER_SRC/DST_OPTIMAL.
void CommandBuffer::copy_image(const Image &src, const Image &dst, ImageAspects src_aspects, ImageAspects dst_aspects, u32 src_mip_level, u32 dst_mip_level) const {
    assert((src.extent.height == dst.extent.height) && "copy_image is designed for images of the same extent.");// todo make copy_image_regions
    assert((src.extent.width == dst.extent.width) && "copy_image is designed for images of the same extent.");
    assert(src.layer_count == dst.layer_count && "cop_image is designed for images with the same amount of layers.");

    VkImageCopy2 region{
        .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
        .pNext = nullptr,
        .srcSubresource{
            .aspectMask = static_cast<VkImageAspectFlags>(src_aspects),
            .mipLevel = src_mip_level,
            .baseArrayLayer = 0,
            .layerCount = src.layer_count,
        },
        .srcOffset{},
        .dstSubresource{
            .aspectMask = static_cast<VkImageAspectFlags>(dst_aspects),
            .mipLevel = dst_mip_level,
            .baseArrayLayer = 0,
            .layerCount = dst.layer_count,
        },
        .dstOffset{},
        .extent = {.width=src.extent.width, .height=src.extent.height, .depth=1},
    };
    VkCopyImageInfo2 image_copy{
        .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
        .pNext = nullptr,
        .srcImage = src.vk_image,
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstImage = dst.vk_image,
        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .regionCount = 1,
        .pRegions = &region,
    };
    vkCmdCopyImage2(this->buffer, &image_copy);
}

// The layouts of the images MUST be VK_IMAGE_LAYOUT_TRANSFER_SRC/DST_OPTIMAL
void CommandBuffer::blit(
    const Image &src,
    const Image &dst,
    ivec2 src_top_left,
    ivec2 src_bottom_right,
    ivec2 dst_top_left,
    ivec2 dst_bottom_right,
    ImageAspects aspects) const // todo add VkCmdCopyImage function for when we don't need to blit
{
    assert(src.layer_count == dst.layer_count);

    VkImageBlit2 blit_region{
        .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
        .pNext = nullptr,

        .srcSubresource{
            .aspectMask = static_cast<VkImageAspectFlags>(aspects),
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = src.layer_count,
        },
        .srcOffsets{
            {.x=src_top_left.x, .y=src_top_left.y, .z=0},
            {.x=src_bottom_right.x, .y=src_bottom_right.y, .z=1},
        },

        .dstSubresource{
            .aspectMask = static_cast<VkImageAspectFlags>(aspects),
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = dst.layer_count,

        },
        .dstOffsets{
            {.x=dst_top_left.x, .y=dst_top_left.y, .z=0},
            {.x=dst_bottom_right.x, .y=dst_bottom_right.y, .z=1},
        },
    };
    VkBlitImageInfo2 blit_info{
        .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
        .pNext = nullptr,
        .srcImage = src.vk_image,
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstImage = dst.vk_image,
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
    bind_descriptor_sets_inner(VK_PIPELINE_BIND_POINT_COMPUTE, sets, pipeline.layout.layout);
}
void CommandBuffer::bind_descriptor_set(const ComputePipeline &pipeline, DescriptorSet set) const{
    bind_descriptor_sets(pipeline, {set});
}

void CommandBuffer::dispatch(u32 x, u32 y, u32 z) const{
    vkCmdDispatch(this->buffer, x, y, z);
}

void CommandBuffer::end() const{
    VK_CHECK(vkEndCommandBuffer(this->buffer));
}

DescriptorSetBuilder &DescriptorSetBuilder::bind(u32 binding, VkDescriptorType type, u32 count, VkShaderStageFlagBits accessible_stages_flags){
    finalized = false;
    VkDescriptorSetLayoutBinding new_bind{
        .binding = binding,
        .descriptorType = type,
        .descriptorCount = count,
        .stageFlags = accessible_stages_flags,
        .pImmutableSamplers = nullptr,
    };
    bindings.push_back(new_bind);
    return *this;
}

DescriptorSet DescriptorSetBuilder::build(VulkanEngine &vk){
    if (!finalized){
        std::vector<VkDescriptorBindingFlags> binding_flags(bindings.size());
        for (size_t i = 0; i < bindings.size(); ++i) {
            binding_flags[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
        }
        VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_ci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .pNext = nullptr,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindingFlags = binding_flags.data(),
        };

        VkDescriptorSetLayoutCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = &binding_flags_ci,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            .bindingCount = static_cast<u32>(bindings.size()),
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
