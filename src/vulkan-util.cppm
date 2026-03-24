module;

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <VkBootstrap.h>
#include <vulkan/vk_enum_string_helper.h>

#include <glm/vec2.hpp>

#include <boost/pfr.hpp>

#include <vector>
#include <span>
#include <array>
#include <print>
#include <fstream>
#include <unordered_set>
#include <optional>


#include <glm/vec3.hpp>
#include <glm/vec4.hpp>


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


export module vulkanUtil;
import vertexBufferAttributeTypes;

static inline void VK_CHECK(VkResult result){
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

struct APIVersionVulkan{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;

    [[nodiscard]] uint32_t to_vk_enum()const{
        return VK_MAKE_VERSION(major, minor, patch);
    }
};


const uint64_t timeout_length = 3000000000;

export enum class MSAALevel : std::uint8_t{
    //todo: check physical device limits, they might not support one of these
    OFF = VK_SAMPLE_COUNT_1_BIT,
    X2 = VK_SAMPLE_COUNT_2_BIT,
    X4 = VK_SAMPLE_COUNT_4_BIT,
    X8 = VK_SAMPLE_COUNT_8_BIT,
};

namespace struct_makers {

static VkCommandBufferBeginInfo command_buffer_begin_info(VkCommandBufferUsageFlags flags){
    return VkCommandBufferBeginInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = flags,
        .pInheritanceInfo = nullptr,
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

export template <typename T>
struct PushConstant {
    static_assert(std::is_trivially_copyable_v<T>, "Push constant type must be trivially copyable");

    const uint32_t offset;
    const uint32_t size;
    const VkShaderStageFlags shader_stages;
    T data;

    PushConstant(uint32_t offset, VkShaderStageFlags shader_stages, const T& data = {})
    :
    offset(offset),
    size(sizeof(T)),
    shader_stages(shader_stages),
    data(data)
    {}
};

export class PushConstantsBuilder{
    uint32_t current_last_byte_used = 0;
    std::vector<VkPushConstantRange> ranges;

    static bool range_stages_do_not_overlap(const std::vector<VkPushConstantRange>& old, VkShaderStageFlags new_flags){
        for (const auto &range : old){
            if ((bool)(range.stageFlags & new_flags)){
                // todo print which stage we're talking about
                std::println("Assert failed: specified multiple push constant ranges for the same shader stage. Are you trying to make push constants for multiple pipelines? You'll have to use a PushConstantsBuilder for each pipeline that has unique push constant ranges.");
                return false;
            }
        }
        return true;
    }

public:
    PushConstantsBuilder() = default;

    template <typename T>
    PushConstant<T> add(VkShaderStageFlags shader_stages){
        static_assert(sizeof(T) % 4 == 0, "Size of push constant must be multiple of 4");
        assert(range_stages_do_not_overlap(ranges, shader_stages));

        VkPushConstantRange range{
            .stageFlags = shader_stages,
            .offset = current_last_byte_used,
            .size = sizeof(T),
        };

        ranges.push_back(range);

        current_last_byte_used += sizeof(T);

        assert(current_last_byte_used <= 128 && "Assert failed: used more push constant space than is guaranteed to exist");

        return PushConstant<T>{range.offset, range.stageFlags};
    }

    [[nodiscard]] const std::vector<VkPushConstantRange>& get_ranges() const{
        return ranges;
    }
};



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

export struct VulkanEngine{
    VkInstance vk_instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkSurfaceKHR surface{};
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue graphics_queue;
    uint32_t graphics_queue_family;
    VmaAllocator allocator;
    VkDescriptorPool descriptor_pool;
    APIVersionVulkan api_version{.major=1, .minor=3, .patch=0};

    // These are used in the destructor to delete everything this engine made.

    // The reason why we need to do this instead of just adding destructors is that this way we
    // can ensure that they are destroyed in the correct order, thus the user of the objects does
    // not need to be careful about the order in which they declare the objects. While this usually isn't
    // a concern when they are just declaring them on the stack one by one as the fact that default constructors do not exist for these objects prevents them from initializing
    // an object before they initialize the objects they depend on, they could still call the destructors in the wrong order if they use something like a vector, as those
    // don't need to contain an initialized member.

    // Might also use these in asserts in debug mode to ensure this is the engine that made them.
    // Maybe I'll get around to deleting individual elements
    struct SwapchainTrackingInfo{VkSwapchainKHR swapchain; std::vector<VkImageView> image_views;};
    std::vector<SwapchainTrackingInfo> created_swapchains;
    struct ImageTrackingInfo{VkImage image; VmaAllocation allocation;};
    std::vector<ImageTrackingInfo> created_images;
    std::unordered_set<VkImageView> created_image_views;
    std::vector<VkCommandPool> created_command_pools;
    std::vector<VkFence> created_fences;
    std::vector<VkSemaphore> created_semaphores;
    std::vector<VkDescriptorSetLayout> descriptor_set_layouts_to_delete;
    std::vector<VkShaderModule> created_shader_modules;
    std::vector<VkPipeline> created_pipelines;
    std::vector<VkPipelineLayout> created_pipeline_layouts;
    struct BufferTrackingInfo{VkBuffer buffer; VmaAllocation allocation;};
    std::vector<BufferTrackingInfo> created_buffers;

    bool imgui_is_initialized = false;

    VulkanEngine(const VulkanEngine &) = delete;
    VulkanEngine(VulkanEngine &&) = delete;
    VulkanEngine &operator=(const VulkanEngine &) = delete;
    VulkanEngine &operator=(VulkanEngine &&) = delete;

    VulkanEngine(SDL_Window *window) {
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
        VK_CHECK(vkDeviceWaitIdle(device));

        if (imgui_is_initialized){
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplSDL3_Shutdown();
            ImGui::DestroyContext();
        }

        for(auto &pipeline : created_pipelines){
            vkDestroyPipeline(device, pipeline, nullptr);
        }
        for(auto &pipeline_layout : created_pipeline_layouts){
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        }

        for(auto &shader_module : created_shader_modules){
            vkDestroyShaderModule(device, shader_module, nullptr);
        }

        for(auto &created_buffer : created_buffers){
            vmaDestroyBuffer(allocator, created_buffer.buffer, created_buffer.allocation);
        }

        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        for(auto &layout : descriptor_set_layouts_to_delete){
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
        for(const auto &img_view : created_image_views){
            vkDestroyImageView(device, img_view, nullptr);
        }
        for(auto &image : created_images){
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


    void init_imgui(SDL_Window *window, VkFormat image_format, MSAALevel msaa_level = MSAALevel::OFF) {
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
};

export enum class ImageAspects:VkImageAspectFlags{
    COLOR = VK_IMAGE_ASPECT_COLOR_BIT,
    DEPTH = VK_IMAGE_ASPECT_DEPTH_BIT,
    STENCIL = VK_IMAGE_ASPECT_STENCIL_BIT,
    DEPTH_STENCIL = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
};

export struct Image{
    VkImage vk_image;
    VmaAllocation allocation;
    VkExtent2D extent;
    VkFormat format;
    // Important to remember that this isn't the layout the image is currently in, but is instead
    // the layout that the latest recorded image barrier command (transition command) has transitioned it to
    VkImageLayout layout;

    [[nodiscard]] VkFormat get_format() const { return format; }

    Image(
        VulkanEngine &vk,
        VkExtent2D extent,// was a glm uvec2
        VkFormat format,
        VkImageUsageFlags image_usage_flags,
        VkMemoryPropertyFlagBits memory_property_flags)
    :extent(extent),
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

    Image(VkImage img, VkExtent2D extent, VkFormat format)
    :vk_image(img), allocation(nullptr), extent(extent), format(format), layout(VK_IMAGE_LAYOUT_UNDEFINED)
    {}
};

export struct ImageView{
    VkImageView view;

    ImageView(
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

    ImageView(VkImageView vk_view):view(vk_view){}
};

export struct GpuFence{
    VkFence fence;
    GpuFence(VulkanEngine &vk, bool signaled){
        VkFenceCreateInfo fence_create_info {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0u,
        };
        VK_CHECK(vkCreateFence(vk.device, &fence_create_info, nullptr, &fence));
        vk.created_fences.push_back(fence);
    }

    void wait(VulkanEngine &vk){
        VK_CHECK(vkWaitForFences(vk.device, 1, &fence, VK_TRUE, timeout_length));
        VK_CHECK(vkResetFences(vk.device, 1, &fence));
    }
};

export struct GpuSemaphore{
    VkSemaphore semaphore;
    GpuSemaphore(VulkanEngine &vk){
        VkSemaphoreCreateInfo semaphore_create_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
        };
        VK_CHECK(vkCreateSemaphore(vk.device, &semaphore_create_info, nullptr, &semaphore));
        vk.created_semaphores.push_back(semaphore);
    }
};


static void destroy_swapchain(VkSwapchainKHR swapchain, VkDevice device, std::vector<ImageView> &swapchain_image_views){
    for(auto &swapchain_image_view : swapchain_image_views){
        vkDestroyImageView(device, swapchain_image_view.view, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);
}

export struct Swapchain{
    VkSwapchainKHR swapchain{};
    VkFormat image_format{};
    VkExtent2D extent{};
    std::vector<Image> images;
    std::vector<ImageView> image_views;

    [[nodiscard]] std::vector<Image> &get_images() { return images; }
    [[nodiscard]] std::vector<ImageView> &get_image_views() { return image_views; }
    [[nodiscard]] const VkExtent2D &get_extent() const { return extent; }
    [[nodiscard]] VkFormat get_format() const { return image_format; }

    Swapchain(VulkanEngine &vk, SDL_Window *window, VkPresentModeKHR present_mode)
    {
        build_swapchain(vk, window, present_mode);
    }

    // Returns the index of the next image in the swapchain
    [[nodiscard]] uint32_t acquire_next_image(VulkanEngine &vk, GpuSemaphore signal_sema) const{
        uint32_t swapchain_image_index;
        VK_CHECK(vkAcquireNextImageKHR(vk.device, swapchain, timeout_length, signal_sema.semaphore, nullptr, &swapchain_image_index));
        return swapchain_image_index;
    }

    void present(VulkanEngine &vk, GpuSemaphore wait_sema, uint32_t swapchain_image_index){
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

 private:
    void build_swapchain(
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

 public:
    // Almost certainly not to be used outside of the vulkan-util.cppm file. The swapchain is destroyed by the
    // VulkanInstanceInfo that made it.
    void destroy_this_swapchain(VkDevice device){
        //TODO: we MUST remove the swapchain from the VulkanEngine too...
        destroy_swapchain(swapchain, device, image_views);
    }

    void rebuild_swapchain(
        VulkanEngine &vk,
        SDL_Window *window,
        VkPresentModeKHR present_mode)
    {
        destroy_this_swapchain(vk.device);
        build_swapchain(vk, window, present_mode);
    }
};

export struct CommandPool{
    VkCommandPool pool;
    CommandPool(VulkanEngine &vk){
        VkCommandPoolCreateInfo command_pool_info{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext            = nullptr,
            .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = vk.graphics_queue_family,
        };
        VK_CHECK(vkCreateCommandPool(vk.device, &command_pool_info, nullptr, &pool));
        vk.created_command_pools.push_back(pool);
    }
};

export struct DescriptorSet{
    VkDescriptorSet set;
    VkDescriptorSetLayout layout;

    DescriptorSet(VulkanEngine &vk, VkDescriptorSetLayout layout)
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

    void update(VulkanEngine &vk, uint32_t bind, std::span<ImageView> views){
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

    void update(VulkanEngine &vk, uint32_t bind, ImageView &view){
        update(vk, bind, std::span<ImageView>(&view, 1));
    }
};

struct Shader{
    VkShaderModule module;

    Shader(VulkanEngine &vk, const std::string_view filepath){
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
};

export struct ComputeShader : public Shader{
    ComputeShader(VulkanEngine &vk, const std::string_view filepath)
    :Shader(vk, filepath) {
        assert(filepath.find(".comp") != std::string_view::npos && "Compute shaders should have .comp in their name. You either passed the wrong shader or your shader does not follow naming convention");
    }
};
export struct VertexShader : public Shader {
    VertexShader(VulkanEngine &vk, const std::string_view filepath)
    :Shader(vk, filepath) {
        assert(filepath.find(".vert") != std::string_view::npos && "Vertex shaders should have .vert in their name. You either passed the wrong shader or your shader does not follow naming convention");
    }
};
export struct FragmentShader : public Shader {
    FragmentShader(VulkanEngine &vk, const std::string_view filepath)
    :Shader(vk, filepath) {
        assert(filepath.find(".frag") != std::string_view::npos && "Fragment shaders should have .frag in their name. You either passed the wrong shader or your shader does not follow naming convention");
    }
};

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

export struct PipelineLayout{
    VkPipelineLayout layout;
    PipelineLayout(
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

    PipelineLayout(VulkanEngine &vk, DescriptorSet descriptor_set, const std::optional<std::vector<VkPushConstantRange>> &push_constants)
    :PipelineLayout(vk, {descriptor_set}, push_constants){}
};

// This struct does not need to be kept alive after being used to create a pipeline, and thus can be reset for use in another pipeline's initialization.
export struct SpecializationInfo{
private:
    VkSpecializationInfo info{};
public:
    std::vector<VkSpecializationMapEntry> entries;
    size_t data_buffer_max_capacity_in_bytes{};
    void *data{};
    bool finalized{};
    // Differs from data_buffer_in_bytes in that this is the actually used memory, while data_buffer_in_bytes includes potentially unused memory
    // and always reflects the actual amount of bytes available in the buffer.
    size_t data_size_in_bytes{};

    SpecializationInfo(size_t total_data_size = 128)
    :data_buffer_max_capacity_in_bytes(total_data_size),
    data(malloc(data_buffer_max_capacity_in_bytes))
    {
        assert(total_data_size >= 4 && "Specialization info data must be at least 4 bytes large");
        assert(total_data_size % 4 == 0 && "It only ever makes sense for the size of a specialization data buffer to be a multiple of 4");
        entries.reserve(data_buffer_max_capacity_in_bytes/sizeof(int32_t));
        if(data == nullptr) { abort(); }
    }

    template<typename T>
    SpecializationInfo &add_entry(uint32_t constant_ID, T constant_value){
        static_assert(
            std::is_same_v<T, int32_t> ||
            std::is_same_v<T, uint32_t> ||
            std::is_same_v<T, float> ||
            std::is_same_v<T, double> ||
            std::is_same_v<T, VkBool32>,
            "Specialization constants can only be of the following types: int32_t, uint32_t, float, double, VkBool32"
        );
        static_assert(sizeof(T) == 4 || sizeof(T) == 8);

        if(data_size_in_bytes + sizeof(T) > data_buffer_max_capacity_in_bytes){
            assert(false && "Exceeded limit of data buffer while adding entry to specialization info");
            std::println("Exceeded limit of data buffer while adding entry to specialization info");
            abort();
        }
        assert(std::none_of(entries.begin(), entries.end(), [&](const VkSpecializationMapEntry& e){ return e.constantID == constant_ID; }) && "Duplicate specialization constant ID: can't set the same constant twice");

        finalized = false;

        uint32_t offset = data_size_in_bytes;
        entries.emplace_back(constant_ID, offset, sizeof(T));
        memcpy((std::byte*)data + offset, &constant_value, sizeof(T));
        data_size_in_bytes += sizeof(T);
        return *this;
    }

    SpecializationInfo &reset(){
        finalized = false;
        entries.clear();
        data_size_in_bytes = 0;
        return *this;
    }

    // DO NOT cache the output of this funciton. If you ever reset this instance of SpecializationInfo or let it be destroyed,
    // the returned value of this function becomes invalid. Just pass it to a pipeline creation function and let it go.
    const VkSpecializationInfo *get_vk_specialization_info(){
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

    ~SpecializationInfo(){
        free(data);
    }
    SpecializationInfo(const SpecializationInfo&) = delete;
    SpecializationInfo(SpecializationInfo&&) = delete;
    SpecializationInfo& operator=(const SpecializationInfo&) = delete;
    SpecializationInfo& operator=(SpecializationInfo&&) = delete;
};

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

export struct ComputePipeline{
    VkPipeline pipeline;
    PipelineLayout layout;

    ComputePipeline(
        VulkanEngine &vk,
        ComputeShader shader_module,
        PipelineLayout pipeline_layout,
        SpecializationInfo */* _Nullable */specialization_info = nullptr)
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
};

struct VulkanBuffer{
    VkBuffer buffer;
    uint32_t size_in_bytes;
    VmaAllocation allocation;
private:
    VkDeviceAddress device_address;
public:
    [[nodiscard]] const VkDeviceAddress &get_device_address() const { return device_address; }

    VulkanBuffer(
        VulkanEngine &vk,
        uint32_t size_in_bytes,
        VkBufferUsageFlags usage_flags,
        VmaAllocationCreateFlags vma_flags,
        VkMemoryPropertyFlags memory_property_flags_required,
        VkMemoryPropertyFlags memory_property_flags_preferred = 0)
        :size_in_bytes(size_in_bytes)
    {
        VkBufferCreateInfo buf_create_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags{},
            .size = size_in_bytes,
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
};

// A GPU-side buffer for the gpu to read from and write to.
export struct StorageBuffer : public VulkanBuffer{
    StorageBuffer(VulkanEngine &vk, uint32_t size_in_bytes, bool is_transfer_source, bool is_transfer_dest)
    :VulkanBuffer(vk,
                  size_in_bytes,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                  | (is_transfer_source ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : 0)
                  | (is_transfer_dest ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0),
                  0,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    {}
};

// Used to get data from Host to Device. For performance reasons, host should write into it in sequentially, not at random.
// Copy the contents into a buffer that resides in the Device instead of accessing it from the device directly.
export struct StagingBuffer : public VulkanBuffer{
private:
    void *mapped_data;
public:
    void *get_mapped_data() { return mapped_data; } // remember that if you ever have to add defragmentation or begin unmapping/remapping it you'll have to instead fetch this with vmaGetAllocationInfo every time because it might change

    StagingBuffer(VulkanEngine &vk, uint64_t size_in_bytes)
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
};

// The opposite of a staging buffer, can be used to copy gpu resources to the cpu or the gpu might store its result in it directly
// It's fine for the host to access it in whatever way it pleases, but access will be slow for the device (unless we're on a UMA machine)
export struct ReadbackBuffer : public VulkanBuffer{
private:
    void *mapped_data;
public:
    void *get_mapped_data() { return mapped_data; } // remember that if you ever have to add defragmentation or begin unmapping/remapping it you'll have to instead fetch this with vmaGetAllocationInfo every time because it might change

    ReadbackBuffer(VulkanEngine &vk, uint32_t size_in_bytes)
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
};

export struct IndexBuffer : public VulkanBuffer{
    static const VkIndexType index_type = VK_INDEX_TYPE_UINT32;
    IndexBuffer(VulkanEngine &vk, uint32_t total_indexes)
    :VulkanBuffer(vk,
                  total_indexes * sizeof(uint32_t),
                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  0, 0)
    {}
};

export template <typename VertexStruct>
struct VertexBuffer : public VulkanBuffer{
    static const std::size_t vertex_attribute_count = boost::pfr::tuple_size_v<VertexStruct>;
    static const uint32_t stride = sizeof(VertexStruct);

    VertexBuffer(VulkanEngine &vk, int element_count)
    :VulkanBuffer(vk,
                 sizeof(VertexStruct) * element_count,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 0, 0)
    {
        static_assert(vertex_attribute_count <= 16, "Use only up to 16 vertex attributes as some GPUs don't support more than that.");
    }

    template <typename T>
    static constexpr VkFormat get_vertex_attribute_format(const std::string_view field_name, const T& field_value){
        using U = std::remove_cvref_t<T>;

        // This list contains only widely supported formats.
        if constexpr      (std::is_same_v<U, int8_norm_t>)   return VK_FORMAT_R8_SNORM;
        else if constexpr (std::is_same_v<U, int16_norm_t>)  return VK_FORMAT_R16_SNORM;
        else if constexpr (std::is_same_v<U, int8_t>)        return VK_FORMAT_R8_SINT;
        else if constexpr (std::is_same_v<U, int16_t>)       return VK_FORMAT_R16_SINT;
        else if constexpr (std::is_same_v<U, int32_t>)       return VK_FORMAT_R32_SINT;
        else if constexpr (std::is_same_v<U, uint8_norm_t>)  return VK_FORMAT_R8_UNORM;
        else if constexpr (std::is_same_v<U, uint16_norm_t>) return VK_FORMAT_R16_UNORM;
        else if constexpr (std::is_same_v<U, uint8_t>)       return VK_FORMAT_R8_UINT;
        else if constexpr (std::is_same_v<U, uint16_t>)      return VK_FORMAT_R16_UINT;
        else if constexpr (std::is_same_v<U, uint32_t>)      return VK_FORMAT_R32_UINT;
        else if constexpr (std::is_same_v<U, float>)         return VK_FORMAT_R32_SFLOAT;
        //else if constexpr (std::is_same_v<U, double>)        return VK_FORMAT_R64_SFLOAT; unsupported by most hardware (~27% support on gpuinfo.org)

        else if constexpr (std::is_same_v<U, ivec2_norm8>)   return VK_FORMAT_R8G8_SNORM;
        else if constexpr (std::is_same_v<U, ivec2_norm16>)  return VK_FORMAT_R16G16_SNORM;
        else if constexpr (std::is_same_v<U, ivec3_norm8>)   return VK_FORMAT_R8G8B8_SNORM; // ~82% support on gpuinfo.org
        else if constexpr (std::is_same_v<U, ivec3_norm16>)  return VK_FORMAT_R16G16B16_SNORM; // ~82.5%
        else if constexpr (std::is_same_v<U, ivec4_norm8>)   return VK_FORMAT_R8G8B8A8_SNORM;
        else if constexpr (std::is_same_v<U, ivec4_norm16>)  return VK_FORMAT_R16G16B16A16_SNORM;
        else if constexpr (std::is_same_v<U, uvec2_norm8>)   return VK_FORMAT_R8G8_UNORM;
        else if constexpr (std::is_same_v<U, uvec2_norm16>)  return VK_FORMAT_R16G16_UNORM;
        else if constexpr (std::is_same_v<U, uvec3_norm8>)   return VK_FORMAT_R8G8B8_UNORM; // ~82%
        else if constexpr (std::is_same_v<U, uvec3_norm16>)  return VK_FORMAT_R16G16B16_UNORM; // ~82%
        else if constexpr (std::is_same_v<U, uvec4_norm8>)   return VK_FORMAT_R8G8B8A8_UNORM;
        else if constexpr (std::is_same_v<U, uvec4_norm16>)  return VK_FORMAT_R16G16B16A16_UNORM;

        else if constexpr (std::is_same_v<U, glm::i8vec2>)   return VK_FORMAT_R8G8_SINT;
        else if constexpr (std::is_same_v<U, glm::i8vec3>)   return VK_FORMAT_R8G8B8_SINT; // ~82%
        else if constexpr (std::is_same_v<U, glm::i8vec4>)   return VK_FORMAT_R8G8B8A8_SINT;
        else if constexpr (std::is_same_v<U, glm::i16vec2>)  return VK_FORMAT_R16G16_SINT;
        else if constexpr (std::is_same_v<U, glm::i16vec3>)  return VK_FORMAT_R16G16B16_SINT; // ~78%
        else if constexpr (std::is_same_v<U, glm::i16vec4>)  return VK_FORMAT_R16G16B16A16_SINT;
        else if constexpr (std::is_same_v<U, glm::ivec2>)    return VK_FORMAT_R32G32_SINT;
        else if constexpr (std::is_same_v<U, glm::ivec3>)    return VK_FORMAT_R32G32B32_SINT;
        else if constexpr (std::is_same_v<U, glm::ivec4>)    return VK_FORMAT_R32G32B32A32_SINT;
        else if constexpr (std::is_same_v<U, glm::u8vec2>)   return VK_FORMAT_R8G8_UINT;
        else if constexpr (std::is_same_v<U, glm::u8vec3>)   return VK_FORMAT_R8G8B8_UINT; // ~82%
        else if constexpr (std::is_same_v<U, glm::u8vec4>)   return VK_FORMAT_R8G8B8A8_UINT;
        else if constexpr (std::is_same_v<U, glm::u16vec2>)  return VK_FORMAT_R16G16_UINT;
        else if constexpr (std::is_same_v<U, glm::u16vec3>)  return VK_FORMAT_R16G16B16_UINT; // ~78%
        else if constexpr (std::is_same_v<U, glm::u16vec4>)  return VK_FORMAT_R16G16B16A16_UINT;
        else if constexpr (std::is_same_v<U, glm::uvec2>)    return VK_FORMAT_R32G32_UINT;
        else if constexpr (std::is_same_v<U, glm::uvec3>)    return VK_FORMAT_R32G32B32_UINT;
        else if constexpr (std::is_same_v<U, glm::uvec4>)    return VK_FORMAT_R32G32B32A32_UINT;

        else if constexpr (std::is_same_v<U, glm::vec2>)     return VK_FORMAT_R32G32_SFLOAT;
        else if constexpr (std::is_same_v<U, glm::vec3>)     return VK_FORMAT_R32G32B32_SFLOAT;
        else if constexpr (std::is_same_v<U, glm::vec4>)     return VK_FORMAT_R32G32B32A32_SFLOAT;

        else if constexpr (std::is_same_v<U, int32_A2R10G10B10_t>)          return VK_FORMAT_A2R10G10B10_SINT_PACK32;
        else if constexpr (std::is_same_v<U, int32_A2R10G10B10_norm_t>)     return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
        else if constexpr (std::is_same_v<U, uint32_A2R10G10B10_t>)         return VK_FORMAT_A2R10G10B10_UINT_PACK32;
        else if constexpr (std::is_same_v<U, uint32_A2R10G10B10_norm_t>)    return VK_FORMAT_A2R10G10B10_UNORM_PACK32;


        std::println("Your vertex struct conainted a field named '{}', which has a type that isn't supported as a vertex attribute.", field_name);
        assert(false);
        abort();
        return VK_FORMAT_UNDEFINED;
    }

    [[nodiscard]] static constexpr std::array<VkVertexInputAttributeDescription, vertex_attribute_count>
    get_vertex_attribute_descriptions(uint32_t binding = 0, uint32_t first_location = 0)
    {
        std::array<VkVertexInputAttributeDescription, vertex_attribute_count> vertex_attribute_descriptions;
        uint32_t curr_location = first_location;
        uint32_t curr_offset = 0;
        int i = 0;
        boost::pfr::for_each_field_with_name(VertexStruct{}, [&](std::string_view name, const auto &value){
            vertex_attribute_descriptions[i++] = {
                .location = curr_location,
                .binding = binding,
                .format = get_vertex_attribute_format(name, value),
                .offset = curr_offset,
            };
            ++curr_location;
            curr_offset += sizeof(std::remove_cvref_t<decltype(value)>);
        });
        return vertex_attribute_descriptions;
    }
};

export template <typename T>
struct GraphicsPipeline{
    VkPipeline pipeline;
    PipelineLayout layout;
    GraphicsPipeline(
        VulkanEngine &vk,
        VertexShader vert_shader,
        FragmentShader frag_shader,
        PipelineLayout pipeline_layout,
        const VertexBuffer<T> &vertex_buffer,
        VkFormat color_attachment_format,
        VkFormat depth_attachment_format,
        MSAALevel msaa_level,
        SpecializationInfo */* _Nullable */vert_specialization_info = nullptr,
        SpecializationInfo */* _Nullable */frag_specialization_info = nullptr,
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
    :
    layout(pipeline_layout)
    {
        assert(
            topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST ||
            topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
            topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        );
        VkPipelineShaderStageCreateInfo shader_stage_infos[2];
        shader_stage_infos[0] = make_pipeline_shader_stage_info(vert_shader, VK_SHADER_STAGE_VERTEX_BIT, vert_specialization_info);
        shader_stage_infos[1] = make_pipeline_shader_stage_info(frag_shader, VK_SHADER_STAGE_FRAGMENT_BIT, frag_specialization_info);

        VkPipelineRenderingCreateInfo pipeline_rendering_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .pNext = nullptr,
            .viewMask{},
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &color_attachment_format,
            .depthAttachmentFormat = depth_attachment_format,
            .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
        };

        VkVertexInputBindingDescription vertex_binding_description{
            .binding = 0,
            .stride = vertex_buffer.stride,
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        auto vertex_attribute_descriptions = vertex_buffer.get_vertex_attribute_descriptions();

        VkPipelineVertexInputStateCreateInfo vert_input_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags{},
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertex_binding_description,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_attribute_descriptions.size()),
            .pVertexAttributeDescriptions = vertex_attribute_descriptions.data(),
        };

        VkPipelineInputAssemblyStateCreateInfo assembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags{},
            .topology = topology,
            .primitiveRestartEnable = VK_FALSE,
        };

        VkPipelineViewportStateCreateInfo viewport_state = {};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo raster_state{};
        raster_state.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster_state.cullMode = VK_CULL_MODE_BACK_BIT;
        raster_state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster_state.lineWidth = 1.0f;
        raster_state.polygonMode = VK_POLYGON_MODE_FILL;

        VkPipelineMultisampleStateCreateInfo ms_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags{},
            .rasterizationSamples = static_cast<VkSampleCountFlagBits>(msaa_level),
            .sampleShadingEnable = VK_FALSE, //true = ssaa, false = msaa
            .minSampleShading{},
            .pSampleMask = nullptr,
            .alphaToCoverageEnable{},
            .alphaToOneEnable{}
        };

        VkPipelineDepthStencilStateCreateInfo ds_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthTestEnable = static_cast<VkBool32>(depth_attachment_format != VK_FORMAT_UNDEFINED),
            .depthWriteEnable = static_cast<VkBool32>(depth_attachment_format != VK_FORMAT_UNDEFINED),
            .depthCompareOp = VK_COMPARE_OP_GREATER, // reversed Z

            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
            .front = {},
            .back = {},
            .minDepthBounds = 0.0f,
            .maxDepthBounds = 1.0f
        };

        VkPipelineColorBlendAttachmentState color_blend_attachment_alpha{
            .blendEnable = VK_TRUE,

            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = VK_BLEND_OP_ADD,

            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,

            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                              VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT
        };

        VkPipelineColorBlendStateCreateInfo color_blend{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext =nullptr,
            .flags{},
            .logicOpEnable = VK_FALSE,
            .logicOp{},
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment_alpha,
            .blendConstants{},
        };

        std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags{},
            .dynamicStateCount = dynamic_states.size(),
            .pDynamicStates = dynamic_states.data(),
        };

        VkGraphicsPipelineCreateInfo graphics_pipeline_create_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &pipeline_rendering_create_info,
            .flags{},
            .stageCount = 2,
            .pStages = shader_stage_infos,
            .pVertexInputState = &vert_input_state_info,
            .pInputAssemblyState = &assembly,
            .pTessellationState = nullptr,
            .pViewportState = &viewport_state,
            .pRasterizationState = &raster_state,
            .pMultisampleState = &ms_state,
            .pDepthStencilState = &ds_state,
            .pColorBlendState = &color_blend,
            .pDynamicState = &dynamic_state_info,
            .layout = this->layout.layout,
            .renderPass = VK_NULL_HANDLE,
            .subpass{},
            .basePipelineHandle{},
            .basePipelineIndex{},
        };
        VK_CHECK(vkCreateGraphicsPipelines(vk.device, VK_NULL_HANDLE, 1, &graphics_pipeline_create_info, nullptr, &this->pipeline));
        vk.created_pipelines.push_back(pipeline);
    }
};

export struct CommandBuffer{
    VkCommandBuffer buffer;
    CommandBuffer(const VulkanEngine &vk, const CommandPool &pool){
        VkCommandBufferAllocateInfo alloc_info{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext              = nullptr,
            .commandPool        = pool.pool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VK_CHECK(vkAllocateCommandBuffers(vk.device, &alloc_info, &buffer));
    }

    void draw_imgui(ImageView target_image_view, VkExtent2D draw_extent) const{
        ImGui::Render(); // todo performance: maybe we should call this on another thread while other things are going on

        VkRenderingAttachmentInfo colorAttachment = struct_makers::attachment_info(target_image_view.view, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingInfo renderInfo = struct_makers::rendering_info(draw_extent, &colorAttachment, nullptr);

        vkCmdBeginRendering(this->buffer, &renderInfo);

        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), this->buffer);

        vkCmdEndRendering(this->buffer);

        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
    }

    void restart(bool one_time_submit) const{
        VK_CHECK(vkResetCommandBuffer(buffer, 0));
        VkCommandBufferBeginInfo begin_info = struct_makers::command_buffer_begin_info(
            one_time_submit ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : 0
        );
        VK_CHECK(vkBeginCommandBuffer(buffer, &begin_info));
    }

    void submit(
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

    void submit(VulkanEngine &vk, GpuFence signal_fence) const{
        submit(vk, std::nullopt, std::nullopt, std::nullopt, std::nullopt, signal_fence);
    }

    void copy_buffer(const VulkanBuffer &src, const VulkanBuffer &dst, const std::span<VkBufferCopy> ranges) const{
        vkCmdCopyBuffer(buffer, src.buffer, dst.buffer, ranges.size(), ranges.data());
    }

    void copy_buffer(const VulkanBuffer &src, const VulkanBuffer &dst, VkBufferCopy range) const{
        vkCmdCopyBuffer(buffer, src.buffer, dst.buffer, 1, &range);
    }

    void copy_entire_buffer(const VulkanBuffer &src, const VulkanBuffer &dst) const{
        assert(dst.size_in_bytes >= src.size_in_bytes);
        VkBufferCopy range{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = src.size_in_bytes,
        };
        vkCmdCopyBuffer(buffer, src.buffer, dst.buffer, 1, &range);
    }

    template <typename T>
    void draw(ImageView color_attachment, VkExtent2D draw_extent, GraphicsPipeline<T> pipeline, const VertexBuffer<T>& vertex_buffer, uint32_t vertex_count) const{
        VkRenderingAttachmentInfo color_render_attachment_info{
            .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext       = nullptr,
            .imageView   = color_attachment.view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode{},
            .resolveImageView{},
            .resolveImageLayout{},
            .loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue{},
        };

        VkRenderingInfo rendering_info = struct_makers::rendering_info(draw_extent, &color_render_attachment_info, nullptr);

        vkCmdBeginRendering(buffer, &rendering_info);

        VkDeviceSize offsets = 0;
        vkCmdBindVertexBuffers(buffer, 0, 1, &vertex_buffer.buffer, &offsets);
        vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);

        VkViewport viewport = {
            .x = 0,
            .y = 0,
            .width = static_cast<float>(draw_extent.width),
            .height = static_cast<float>(draw_extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        vkCmdSetViewport(buffer, 0, 1, &viewport);

        VkRect2D scissor{
            .offset{.x = 0, .y = 0},
            .extent{draw_extent},
        };
        vkCmdSetScissor(buffer, 0, 1, &scissor);

        vkCmdDraw(buffer, vertex_count, 1, 0, 0);
        vkCmdEndRendering(buffer);
    }

    template <typename T>
    void draw_indexed(ImageView color_attachment, VkExtent2D draw_extent, GraphicsPipeline<T> pipeline, const VertexBuffer<T> &vertex_buffer, const IndexBuffer &index_buffer, uint32_t index_count) const{
        VkRenderingAttachmentInfo color_render_attachment_info{
            .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext       = nullptr,
            .imageView   = color_attachment.view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode{},
            .resolveImageView{},
            .resolveImageLayout{},
            .loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue{},
        };

        VkRenderingInfo rendering_info = struct_makers::rendering_info(draw_extent, &color_render_attachment_info, nullptr);

        vkCmdBeginRendering(buffer, &rendering_info);

        VkDeviceSize offsets = 0;
        vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
        vkCmdBindVertexBuffers(buffer, 0, 1, &vertex_buffer.buffer, &offsets);
        vkCmdBindIndexBuffer(buffer, index_buffer.buffer, 0, IndexBuffer::index_type);

        VkViewport viewport = {
            .x = 0,
            .y = 0,
            .width = static_cast<float>(draw_extent.width),
            .height = static_cast<float>(draw_extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        vkCmdSetViewport(buffer, 0, 1, &viewport);

        VkRect2D scissor{
            .offset{.x = 0, .y = 0},
            .extent{draw_extent},
        };
        vkCmdSetScissor(buffer, 0, 1, &scissor);

        vkCmdDrawIndexed(buffer, index_count, 1, 0, 0, 0);
        vkCmdEndRendering(buffer);

    }

    struct BarrierInfo{
        Image &img;
        bool discard_current_data;
        VkImageLayout new_layout;
        VkPipelineStageFlags2 src_stage_mask;
        VkAccessFlags2 src_access_mask;
        VkPipelineStageFlags2 dst_stage_mask;
        VkAccessFlags2 dst_access_mask;
        ImageAspects aspects;
        uint32_t base_mip_level = 0;
        uint32_t mip_level_count = VK_REMAINING_MIP_LEVELS;
    };
    template<typename... BarrierInfo_T>
    void barrier(const BarrierInfo_T&... barrier_infos) const {
        static_assert((std::is_same_v<std::remove_cvref_t<BarrierInfo_T>, BarrierInfo> && ...), "Arguments to CommandBuffer::barrier must be CommandBuffer::BarrierInfo types.");
        std::array<VkImageMemoryBarrier2, sizeof...(BarrierInfo_T)> image_barriers;

        int i = 0;
        const auto add_image_barrier = [&](const BarrierInfo& barrier_info){
            image_barriers[i++] = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = barrier_info.src_stage_mask,
                .srcAccessMask = barrier_info.src_access_mask,
                .dstStageMask = barrier_info.dst_stage_mask,
                .dstAccessMask = barrier_info.dst_access_mask,
                .oldLayout = barrier_info.discard_current_data ? VK_IMAGE_LAYOUT_UNDEFINED : barrier_info.img.layout,
                .newLayout = barrier_info.new_layout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, // todo was 0 before and worked, figure out more about queue families
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = barrier_info.img.vk_image,
                .subresourceRange = {
                    .aspectMask     = static_cast<VkImageAspectFlags>(barrier_info.aspects),
                    .baseMipLevel   = barrier_info.base_mip_level,
                    .levelCount     = barrier_info.mip_level_count,
                    .baseArrayLayer = 0,
                    .layerCount     = VK_REMAINING_ARRAY_LAYERS,
                },
            };
            barrier_info.img.layout = barrier_info.new_layout;
        };

        (add_image_barrier(barrier_infos), ...);

        VkDependencyInfo dep_info {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags{},
            .memoryBarrierCount{},
            .pMemoryBarriers{},
            .bufferMemoryBarrierCount{},
            .pBufferMemoryBarriers{},
            .imageMemoryBarrierCount = static_cast<uint32_t>(sizeof...(BarrierInfo_T)),
            .pImageMemoryBarriers = image_barriers.data(),
        };

        vkCmdPipelineBarrier2(this->buffer, &dep_info);
    }

    void blit(
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

    void blit_entire_images(const Image &source, const Image &destination, ImageAspects aspects) const    {
        blit(source, destination, {0,0}, {source.extent.width, source.extent.height}, {0,0}, {destination.extent.width, destination.extent.height}, aspects);
    }

    void bind_pipeline(const ComputePipeline &pipeline) const{
        vkCmdBindPipeline(this->buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    }

    void bind_descriptor_sets(const ComputePipeline &pipeline, std::initializer_list<DescriptorSet> sets) const{
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
    void bind_descriptor_sets(ComputePipeline pipeline, DescriptorSet set) const{
        bind_descriptor_sets(pipeline, {set});
    }

    template <typename T>
    void update_push_constants(
        ComputePipeline pipeline,
        const PushConstant<T>& push_constant)
    {
        vkCmdPushConstants(this->buffer,
                           pipeline.layout.layout,
                           push_constant.shader_stages,
                           push_constant.offset,
                           push_constant.size,
                           &push_constant.data);
    }

    void dispatch(uint32_t x, uint32_t y, uint32_t z) const{
	vkCmdDispatch(this->buffer, x, y, z);
    }

    void end() const{
	VK_CHECK(vkEndCommandBuffer(this->buffer));
    }
};

// This is what you're meant to use instead of Descriptor Set Layouts and Descriptor Set Layout Bindings. You can
// keep using build as many times as you want just like you would with a Layout.
export struct DescriptorSetBuilder{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    VkDescriptorSetLayout layout{};
    bool finalized = false;

    DescriptorSetBuilder &bind(uint32_t binding, VkDescriptorType type, VkShaderStageFlagBits accessible_stages_flags = VK_SHADER_STAGE_ALL){
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

    DescriptorSet build(VulkanEngine &vk){
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

    // Don't reset unless you know you won't be making another descriptor set with the same layout again. It's slower otherwise.
    // Objects passed into the creation of other objects must not be destroyed while the new object is in use, so we cannot destroy the layout here.
    // Since we use all descriptor sets until the end of the program, there is no reason to want that, either.
    DescriptorSetBuilder &reset(){
        finalized = false;
        layout = {},
        bindings.clear();
        return *this;
    }
};

