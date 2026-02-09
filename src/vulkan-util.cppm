module;

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <VkBootstrap.h>
#include <vulkan/vk_enum_string_helper.h>

#include <glm/vec2.hpp>

#include <vector>
#include <span>
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



static VkDescriptorPool create_descriptor_pool(VkDevice device, uint32_t pool_size, uint32_t max_sets){
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER,                pool_size },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, pool_size },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          pool_size },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          pool_size },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,   pool_size },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,   pool_size },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         pool_size },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         pool_size },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, pool_size },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, pool_size },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,     pool_size },
    };

    VkDescriptorPoolCreateInfo descriptor_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .maxSets = max_sets,
        .poolSizeCount = 1,
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
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    for(auto &swapchain_image_view : swapchain_image_views){
        vkDestroyImageView(device, swapchain_image_view, nullptr);
    }
}

export struct Swapchain{
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
        glm::uvec2 &size,
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

export struct VulkanImage{
    VkImage vk_image;
    VkImageView view;
    VmaAllocation allocation;
    VkExtent3D extent;
    VkFormat format;
    VkImageLayout layout;

    VulkanImage(
        VkDevice device,
        VkExtent3D extent,
        VkFormat format,
        VkImageUsageFlags image_usage_flags,
        VkMemoryPropertyFlagBits memory_property_flags,
        VmaAllocator allocator,
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED)
    :extent(extent),
     format(format),
     layout(layout)
    {
        VkImageCreateInfo img_create_info = struct_makers::image_create_info(format, image_usage_flags, extent, layout);
        VmaAllocationCreateInfo img_alloc_info = {};
        img_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        img_alloc_info.requiredFlags = VkMemoryPropertyFlags(memory_property_flags);
        VkImage image{};
        VmaAllocation allocation{};
        vmaCreateImage(allocator, &img_create_info, &img_alloc_info, &image, &allocation, nullptr);

        VkImageViewCreateInfo view_info = struct_makers::imageview_create_info(format, image, VK_IMAGE_ASPECT_COLOR_BIT);
        VkImageView view{};
        VK_CHECK(vkCreateImageView(device, &view_info, nullptr, &view));

        this->vk_image = image;
        this->view = view;
        this->allocation = allocation;
    }
};

export struct GpuFence{
    VkFence fence;
    GpuFence(VkDevice device, bool signaled){
        VkFenceCreateInfo fence_create_info {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0u,
        };
        VK_CHECK(vkCreateFence(device, &fence_create_info, nullptr, &fence));
    }
};

export struct GpuSemaphore{
    VkSemaphore semaphore;
    GpuSemaphore(VkDevice device){
        VkSemaphoreCreateInfo semaphore_create_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
        };
        VK_CHECK(vkCreateSemaphore(device, &semaphore_create_info, nullptr, &semaphore));
    }
};

export struct CommandPool{
    VkCommandPool pool;
    CommandPool(VkDevice device, uint32_t graphics_queue_family){
        VkCommandPoolCreateInfo command_pool_info{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext            = nullptr,
            .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = graphics_queue_family,
        };
        VK_CHECK(vkCreateCommandPool(device, &command_pool_info, nullptr, &pool));
    }
};

export struct CommandBuffer{
    VkCommandBuffer buffer;
    CommandBuffer(VkDevice device, VkCommandPool pool){
        VkCommandBufferAllocateInfo alloc_info{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext              = nullptr,
            .commandPool        = pool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VK_CHECK(vkAllocateCommandBuffers(device, &alloc_info, &buffer));
    }
};

export struct DescriptorSet{
    VkDescriptorSet set;
    DescriptorSet(VkDevice device, VkDescriptorSetLayout layout, VkDescriptorPool pool){
        VkDescriptorSetAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.pNext = nullptr;
        alloc_info.descriptorPool = pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &layout;
        VK_CHECK(vkAllocateDescriptorSets(device, &alloc_info, &set));
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
    VkDescriptorPool descriptor_pool;
    APIVersionVulkan api_version{.major=1, .minor=3, .patch=0};
    const uint64_t timeout_length = 3000000000;

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
    std::vector<VkDescriptorSetLayout> layouts_to_delete;

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
 
        this->descriptor_pool = create_descriptor_pool(device, 1000, 200);
    };

    ~VulkanEngine(){
        vkDeviceWaitIdle(device);

        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        for(auto &layout : layouts_to_delete){
            vkDestroyDescriptorSetLayout(device, layout, nullptr);
        }
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

    void register_for_deletion(VkDescriptorSetLayout layout){
        layouts_to_delete.push_back(layout);
    }

    Swapchain create_swapchain(SDL_Window *window, VkPresentModeKHR present_mode){
        assert((created_swapchains.size() == 0) && "We don't currently support multiple swapchains.");

        glm::ivec2 size;
        SDL_GetWindowSizeInPixels(window, &size.x, &size.y);
        Swapchain swapchain{
            physical_device, device, surface, size, present_mode
        };
        created_swapchains.push_back({swapchain.swapchain, swapchain.image_views});
        return swapchain;
    }

    VulkanImage create_image(
        VkExtent3D extent,
        VkFormat format,
        VkImageUsageFlags image_usage_flags,
        VkMemoryPropertyFlagBits memory_property_flags,
        VmaAllocator allocator,
        VkImageLayout layout)
    {
        VulkanImage image{
            device, extent, format, image_usage_flags, memory_property_flags, allocator, layout
        };
        created_images.push_back(ImageTrackingInfo{.image=image.vk_image, .view=image.view, .allocation=image.allocation});
        return image;
    }

    DescriptorSet allocate_descriptor_set_from_layout(VkDescriptorSetLayout layout) const{
        return {device, layout, descriptor_pool};
    }

    void update_storage_image_descriptor(DescriptorSet set, std::span<VulkanImage> images, uint32_t bind) const{
        std::vector<VkDescriptorImageInfo> img_infos;
        for (auto image : images){
            img_infos.push_back({
                .sampler{},
                .imageView = image.view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            });
        }

        VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set.set,
            .dstBinding = bind,
            .dstArrayElement{},
            .descriptorCount = static_cast<uint32_t>(img_infos.size()),
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = img_infos.data(),
            .pBufferInfo{},
            .pTexelBufferView{},
        };

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    void update_storage_image_descriptor(DescriptorSet &set, VulkanImage &image, uint32_t bind) const{
        update_storage_image_descriptor(set, std::span<VulkanImage>(&image, 1), bind);
    }

    void wait(GpuFence &fence) const{
        VK_CHECK(vkWaitForFences(device, 1, &fence.fence, (VkBool32)true, timeout_length));
        VK_CHECK(vkResetFences(device, 1, &fence.fence));
    }

    // Returns the index of the next image in the swapchain
    [[nodiscard]] uint32_t acquire_next_image(const Swapchain &swapchain, GpuSemaphore wait_sema) const{
        uint32_t swapchain_image_index;
        VK_CHECK(vkAcquireNextImageKHR(device, swapchain.swapchain, timeout_length, wait_sema.semaphore, nullptr, &swapchain_image_index));
        return swapchain_image_index;
    }

    void restart_buffer(CommandBuffer cmd_buffer, bool one_time_submit){
        VK_CHECK(vkResetCommandBuffer(cmd_buffer.buffer, 0));
        VkCommandBufferBeginInfo begin_info = struct_makers::command_buffer_begin_info(
            one_time_submit ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : 0
        );
	    VK_CHECK(vkBeginCommandBuffer(cmd_buffer.buffer, &begin_info));
    }

    void transition(
        const CommandBuffer &cmd_buffer,
        const VulkanImage &img,
        VkImageLayout new_layout,
        VkPipelineStageFlags2 src_stage_mask,
        VkAccessFlags2 src_access_mask,
        VkPipelineStageFlags2 dst_stage_mask,
        VkAccessFlags2 dst_access_mask)
    {
        VkImageMemoryBarrier2 imageBarrier {};
        imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        imageBarrier.pNext = nullptr;
        imageBarrier.srcStageMask = src_stage_mask;
        imageBarrier.srcAccessMask = src_access_mask;
        imageBarrier.dstStageMask = dst_stage_mask;
        imageBarrier.dstAccessMask = dst_access_mask;
        imageBarrier.oldLayout = img.layout;
        imageBarrier.newLayout = new_layout;
        VkImageAspectFlags aspectMask = (new_layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;//todo think more about this
        imageBarrier.subresourceRange = struct_makers::image_subresource_range(aspectMask);
        imageBarrier.image = img.vk_image;

        VkDependencyInfo depInfo {};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.pNext = nullptr;
        depInfo.imageMemoryBarrierCount = 1; //todo performance: function that lets you do multiple transitions in one vkCmdPipelineBarrier2 call
        depInfo.pImageMemoryBarriers = &imageBarrier;

        vkCmdPipelineBarrier2(cmd_buffer.buffer, &depInfo);
    }

    void submit_commands(
        CommandBuffer cmd_buffer,
        GpuSemaphore wait_sema,//todo: use std::optional for cases where we don't signal/wait
        VkPipelineStageFlagBits2 wait_stage_mask,
        GpuSemaphore signal_sema,
        VkPipelineStageFlagBits2 signal_stage_mask,
        GpuFence signal_fence) const
    {
        VkCommandBufferSubmitInfo cmd_info = struct_makers::command_buffer_submit_info(cmd_buffer.buffer);
        VkSemaphoreSubmitInfo wait_info = struct_makers::semaphore_submit_info(wait_stage_mask, wait_sema.semaphore);
        VkSemaphoreSubmitInfo signal_info = struct_makers::semaphore_submit_info(signal_stage_mask, signal_sema.semaphore);
	    VkSubmitInfo2 submit_info = struct_makers::submit_info(&cmd_info,&signal_info,&wait_info);
        VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit_info, signal_fence.fence));
    }

    void present(const Swapchain& swapchain, GpuSemaphore wait_sema, uint32_t swapchain_image_index) const{
        VkPresentInfoKHR present_info = {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.pNext = nullptr;
        present_info.pSwapchains = &swapchain.swapchain;
        present_info.swapchainCount = 1;
        present_info.pWaitSemaphores = &wait_sema.semaphore;
        present_info.waitSemaphoreCount = 1;
        present_info.pImageIndices = &swapchain_image_index;
        VK_CHECK(vkQueuePresentKHR(graphics_queue, &present_info));
    }

};

// This is what you're meant to use instead of Layouts and Layout Bindings. You can
// keep using build as many times as you want just like you would with a Layout.
export struct DescriptorSetBuilder{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    VkDescriptorSetLayout layout;
    bool finalized = false;

    consteval DescriptorSetBuilder& bind(uint32_t binding, VkDescriptorType type, VkShaderStageFlagBits accessible_stages_flags = VK_SHADER_STAGE_ALL){
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

    DescriptorSet build(VulkanEngine &engine){
        if (!finalized){
            VkDescriptorSetLayoutCreateInfo info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .bindingCount = static_cast<uint32_t>(bindings.size()),
                .pBindings = bindings.data(),
            };
            VK_CHECK(vkCreateDescriptorSetLayout(engine.device, &info, nullptr, &layout));
            engine.register_for_deletion(layout);
            finalized = true;
        }

        return engine.allocate_descriptor_set_from_layout(layout);
    }
};

struct FrameData {
    VkSemaphore swapchain_semaphore;
    VkSemaphore render_semaphore;
    VkFence render_fence;
    VkCommandPool command_pool;
    VkCommandBuffer main_command_buffer;
};












































