module;

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <VkBootstrap.h>
#include <vulkan/vk_enum_string_helper.h>

#include <glm/vec2.hpp>

#include <array>
#include <thread>
#include <chrono>
#include <cmath>
#include <print>
#include <fstream>
#include <functional>
#include <tuple>


#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"

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

//TODO: enable VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT

export module vulkanUtil;

static inline void VK_CHECK(VkResult result){
    if(result != VK_SUCCESS){
        std::print("Vulkan error: {}", string_VkResult(result));
        abort();
    }
}

#ifndef NDEBUG
    const bool use_validation_layers = true;
#else
    const bool use_validation_layers = true;
#endif

struct APIVersionVulkan{
    u_int32_t major;
    u_int32_t minor;
    u_int32_t patch;

    [[nodiscard]] uint32_t to_vk_enum()const{
        return VK_MAKE_VERSION(major, minor, patch);
    }
};


namespace struct_makers {

static VkCommandPoolCreateInfo command_pool_create_info(
    uint32_t queueFamilyIndex,
    VkCommandPoolCreateFlags flags)
{
    return VkCommandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = flags,
        .queueFamilyIndex = queueFamilyIndex,
    };
}

static VkCommandBufferAllocateInfo command_buffer_allocate_info( VkCommandPool pool, uint32_t count){
    return VkCommandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = count,
    };
}

static VkFenceCreateInfo fence_create_info(VkFenceCreateFlags flags){
    return VkFenceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
    };
}

static VkSemaphoreCreateInfo semaphore_create_info(VkSemaphoreCreateFlags flags){
    return VkSemaphoreCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
    };
}

static VkCommandBufferBeginInfo command_buffer_begin_info(VkCommandBufferUsageFlags flags){
    return VkCommandBufferBeginInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = flags,
        .pInheritanceInfo = nullptr,
    };
}

static VkImageSubresourceRange image_subresource_range(VkImageAspectFlags aspectMask){
    return VkImageSubresourceRange{
        .aspectMask     = aspectMask,
        .baseMipLevel   = 0,
        .levelCount     = VK_REMAINING_MIP_LEVELS,
        .baseArrayLayer = 0,
        .layerCount     = VK_REMAINING_ARRAY_LAYERS,
    };
}

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
    VkExtent3D extent)
{
    return VkImageCreateInfo{
        .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext       = nullptr,
        .flags{},
        .imageType   = VK_IMAGE_TYPE_2D,
        .format      = format,
        .extent      = extent,
        .mipLevels   = 1,
        .arrayLayers = 1,
        .samples     = VK_SAMPLE_COUNT_1_BIT,
        .tiling      = VK_IMAGE_TILING_OPTIMAL,
        .usage       = usageFlags,
        .sharingMode{},
        .queueFamilyIndexCount{},
        .pQueueFamilyIndices{},
        .initialLayout{},
    };
}

static VkImageViewCreateInfo imageview_create_info(
    VkFormat format,
    VkImage image,
    VkImageAspectFlags aspectFlags)
{
    return VkImageViewCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext            = nullptr,
        .flags{},
        .image            = image,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = format,
        .components{},
        .subresourceRange = {
            .aspectMask     = aspectFlags,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        },
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
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    for(auto &swapchain_image_view : swapchain_image_views){
        vkDestroyImageView(device, swapchain_image_view, nullptr);
    }
}

struct Swapchain{
    VkSwapchainKHR swapchain{};
    VkFormat image_format = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent{};
    std::vector<VkImage> images;
    std::vector<VkImageView> image_views;

    Swapchain(
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkSurfaceKHR surface,
        glm::uvec2 size,
        VkPresentModeKHR present_mode)
    {
        build_swapchain(physical_device, device, surface, size, present_mode);
    }
    Swapchain() = default;

 private:
    void build_swapchain(
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkSurfaceKHR surface,
        glm::uvec2 size,
        VkPresentModeKHR present_mode)
    {
        vkb::SwapchainBuilder swapchain_builder{physical_device, device, surface};
        vkb::Swapchain vkb_swapchain = swapchain_builder
            // The combination of VK_FORMAT_B8G8R8A8_UNORM and VK_COLOR_SPACE_SRGB_NONLINEAR_KHR assume that you
            // will write in linear space and then manually encode the image to sRGB (aka do gamma correction)
            // as the last thing before blitting to swapchain and presenting.
            .set_desired_format(VkSurfaceFormatKHR{ .format = image_format, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
            .set_desired_present_mode(present_mode)
            .set_desired_extent(size.x, size.y)
            .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            .build()
            .value();

        this->extent = vkb_swapchain.extent;
        this->swapchain = vkb_swapchain.swapchain;
        this->images = vkb_swapchain.get_images().value();
        this->image_views = vkb_swapchain.get_image_views().value();
    }

 public:
    // Almost certainly not to be used outside of the vulkan-util.cppm file. The swapchain is destroyed by the
    // VulkanInstanceInfo that made it.
    void destroy_this_swapchain(VkDevice device){
        destroy_swapchain(swapchain, device, image_views);
    }

    void rebuild_swapchain(
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkSurfaceKHR surface,
        glm::uvec2 size,
        VkPresentModeKHR present_mode)
    {
        destroy_this_swapchain(device);
        build_swapchain(physical_device, device, surface, size, present_mode);
    }
};

struct VulkanImage{
    VkImage vk_image;
    VkImageView image_view;
    VmaAllocation allocation;
    VkExtent3D extent;
    VkFormat format;
    VkImageLayout layout;

    VulkanImage(
        VkExtent3D extent,
        VkFormat format,
        VkImageUsageFlags image_usage_flags,
        VkMemoryPropertyFlagBits memory_property_flags)
    {
    }
};

export struct VulkanEngine{
    VkInstance vk_instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkSurfaceKHR surface{};
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue graphics_queue;
    u_int32_t graphics_queue_family;
    VmaAllocator allocator;
    APIVersionVulkan api_version{.major=1, .minor=3, .patch=0};

 private:
    // These are used in the destructor to delete everything this engine made.
    // Might also be used in asserts in debug mode to ensure this is the engine that made them.
    // Maybe I'll get around to deleting individual elements
    struct SwapchainTrackingInfo{VkSwapchainKHR swapchain; std::vector<VkImageView> image_views;};
    std::vector<SwapchainTrackingInfo> created_swapchains;
    struct ImageTrackingInfo{VkImage image; VkImageView view; VmaAllocation allocation;};
    std::vector<ImageTrackingInfo> created_images;
    std::vector<VkCommandPool> created_command_pools;
    std::vector<VkFence> created_fences;
    std::vector<VkSemaphore> created_semaphores;

 public:
    VulkanEngine(SDL_Window *window) {
        vkb::InstanceBuilder builder;
        vkb::Instance vkb_inst = builder.set_app_name("Vulkan App")
            .request_validation_layers(use_validation_layers)
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
        features13.synchronization2 = (uint)true;
        features13.dynamicRendering = (uint)true;

        VkPhysicalDeviceVulkan12Features features12{};
        features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        features12.descriptorIndexing = (uint)true;
        features12.bufferDeviceAddress = (uint)true;

        vkb::PhysicalDeviceSelector selector{vkb_inst};
        vkb::PhysicalDevice vkb_physical_device = selector
            .set_minimum_version(api_version.major, api_version.minor)
            .set_required_features_13(features13)
            .set_required_features_12(features12)
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
    };

    Swapchain create_swapchain(SDL_Window *window, VkPresentModeKHR present_mode){
        assert((created_swapchains.size() == 0) && "We don't currently support multiple swapchains.");

        int pixel_width{}, pixel_height{};
        SDL_GetWindowSizeInPixels(window, &pixel_width, &pixel_height);
        Swapchain swapchain = {physical_device, device, surface, {pixel_width, pixel_height}, present_mode};
        created_swapchains.push_back({swapchain.swapchain, swapchain.image_views});
        return swapchain;
    }

    VulkanImage create_image(VkExtent3D extent, VkFormat format, VkImageUsageFlags image_usage_flags, VkMemoryPropertyFlagBits memory_property_flags){
        VkImageCreateInfo img_create_info = struct_makers::image_create_info(format, image_usage_flags, extent);
        VmaAllocationCreateInfo img_alloc_info = {};
        img_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        img_alloc_info.requiredFlags = VkMemoryPropertyFlags(memory_property_flags);
        VkImage image;
        VmaAllocation allocation;
        vmaCreateImage(allocator, &img_create_info, &img_alloc_info, &image, &allocation, nullptr);

        VulkanImage ;

    }

    ~VulkanEngine(){
        vkDeviceWaitIdle(device);

        for(auto &command_pool : created_command_pools){
            vkDestroyCommandPool(device, command_pool, nullptr);
        }
        for(auto &fence : created_fences){
            vkDestroyFence(device, fence, nullptr);
        }
        for(auto &sema : created_semaphores){
            vkDestroySemaphore(device, sema, nullptr);
        }
        for(auto &image : created_images){
            vkDestroyImageView(device, image.view, nullptr);
            vmaDestroyImage(allocator, image.image, image.allocation);
        }
        for(auto &swapchain : created_swapchains){
            destroy_swapchain(swapchain.swapchain, device, swapchain.image_views);
        }
        vmaDestroyAllocator(allocator);
        vkDestroySurfaceKHR(vk_instance, surface, nullptr); // can i destroy surface after device?
        vkDestroyDevice(device, nullptr);
        vkb::destroy_debug_utils_messenger(vk_instance, debug_messenger);
        vkDestroyInstance(vk_instance, nullptr);
    }
};

struct FrameData {
    VkSemaphore swapchain_semaphore;
    VkSemaphore render_semaphore;
    VkFence render_fence;
    VkCommandPool command_pool;
    VkCommandBuffer main_command_buffer;
};



























































