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
#include <unordered_set>


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
    const int poolsize_count = 11;
    VkDescriptorPoolSize sizes[poolsize_count] = {
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
        .poolSizeCount = poolsize_count, //todo turn sizes[] into a std::array
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

    Image(
        VmaAllocator allocator,
        VkExtent2D extent,
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
        vmaCreateImage(allocator, &img_create_info, &img_alloc_info, &image, &allocation, nullptr);

        this->vk_image = image;
        this->allocation = allocation;
    }

    Image(VkImage img, VkExtent2D extent, VkFormat format)
    :vk_image(img), allocation(nullptr), extent(extent), format(format), layout(VK_IMAGE_LAYOUT_UNDEFINED)
    {}
};

export struct ImageView{
    VkImageView view;

    ImageView(
        VkDevice device,
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
        VK_CHECK(vkCreateImageView(device, &view_info, nullptr, &view));
    }

    ImageView(VkImageView vk_view):view(vk_view){}
};


static void destroy_swapchain(VkSwapchainKHR swapchain, VkDevice device, std::vector<ImageView> &swapchain_image_views){
    for(auto &swapchain_image_view : swapchain_image_views){
        vkDestroyImageView(device, swapchain_image_view.view, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);
}


static void destroy_swapchain(VkSwapchainKHR swapchain, VkDevice device, std::vector<VkImageView> &swapchain_image_views){
    for(auto &swapchain_image_view : swapchain_image_views){
        vkDestroyImageView(device, swapchain_image_view, nullptr);
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
            .set_desired_format(VkSurfaceFormatKHR{ .format = VK_FORMAT_B8G8R8A8_UNORM, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
            .set_desired_present_mode(present_mode)
            .set_desired_extent(size.x, size.y)
            .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            .build()
            .value();

        this->extent = vkb_swapchain.extent;
        this->swapchain = vkb_swapchain.swapchain;

        std::vector<VkImage> vk_images = vkb_swapchain.get_images().value();
        for (const auto &vk_img : vk_images){
            this->images.emplace_back(vk_img, vkb_swapchain.extent, vkb_swapchain.image_format);
        }
        this->image_format = vkb_swapchain.image_format;

        for(VkImageView vk_view : vkb_swapchain.get_image_views().value()){
            this->image_views.emplace_back(vk_view);
        }
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

    GpuSemaphore(const GpuSemaphore&) = default;
    GpuSemaphore(GpuSemaphore&&) noexcept = default;
    GpuSemaphore& operator=(const GpuSemaphore&) = default;
    GpuSemaphore& operator=(GpuSemaphore&&) noexcept = default;

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

export struct DescriptorSet{
    VkDescriptorSet set;
    VkDescriptorSetLayout layout;

    DescriptorSet(VkDevice device, VkDescriptorSetLayout layout, VkDescriptorPool pool)
    :layout(layout)
    {
        VkDescriptorSetAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.pNext = nullptr;
        alloc_info.descriptorPool = pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &layout;
        VK_CHECK(vkAllocateDescriptorSets(device, &alloc_info, &set));
    }
};

export struct ShaderModule{
    VkShaderModule module;

 private:
    VkShaderModule load_shader_module(VkDevice device, const std::string_view filepath){
        assert(filepath.data()[filepath.size()] == '\0' && "Error: filepath was not null-terminated string");
        // open the file. With cursor at the end
        std::ifstream file(filepath.data(), std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            std::println("Error: could not open file {}", filepath);
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

        VkShaderModule shader_module;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shader_module) != VK_SUCCESS) {
            std::println("Error: could not create shader module for shader {}", filepath);
            abort();
        }
        return shader_module;
    }
 public:

    ShaderModule(VkDevice device, const std::string_view filepath)
    :module(load_shader_module(device, filepath))
    {}

    ShaderModule(VkShaderModule module)
    :module(module)
    {}
};

export struct ComputePipeline{
    VkPipeline pipeline;
    VkPipelineLayout layout;

    void init_compute_pipeline(VkDevice device, std::span<DescriptorSet> descriptor_sets, ShaderModule shader_module){
        // The max this function supports, not the max the machine supprts. That must be queried independently.
        const int max_descriptor_sets_in_shader = 32;
        if(descriptor_sets.size() > max_descriptor_sets_in_shader){
            std::println("Error: Too many descriptor sets in one shader.");
        }

        std::array<VkDescriptorSetLayout, max_descriptor_sets_in_shader> layouts;
        for (int i = 0; i < descriptor_sets.size(); ++i) {
          layouts.at(i) = descriptor_sets[i].layout;
        }

        VkPipelineLayoutCreateInfo computeLayout{};
        computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        computeLayout.pNext = nullptr;
        computeLayout.pSetLayouts = layouts.data();
        computeLayout.setLayoutCount = descriptor_sets.size();
        VK_CHECK(vkCreatePipelineLayout(device, &computeLayout, nullptr, &layout));

        VkShaderModule compute_shader = shader_module.module;
        VkPipelineShaderStageCreateInfo stageinfo{};
        stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageinfo.pNext = nullptr;
        stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageinfo.module = compute_shader;
        stageinfo.pName = "main";

        VkComputePipelineCreateInfo computePipelineCreateInfo{};
        computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        computePipelineCreateInfo.pNext = nullptr;
        computePipelineCreateInfo.layout = layout;
        computePipelineCreateInfo.stage = stageinfo;
        VK_CHECK(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&computePipelineCreateInfo, nullptr, &pipeline));
    }

    ComputePipeline(VkDevice device, std::span<DescriptorSet> descriptors, ShaderModule shader_module){
        init_compute_pipeline(device, descriptors, shader_module);
    }

    ComputePipeline(VkDevice device, std::span<DescriptorSet> descriptors, std::string_view shader_filepath){
        ShaderModule shader_module = ShaderModule(device, shader_filepath);
        init_compute_pipeline(device, descriptors, shader_module);
        vkDestroyShaderModule(device, shader_module.module, nullptr);
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


    void draw_imgui(ImageView target_image_view, VkExtent2D draw_extent) const{
        VkRenderingAttachmentInfo colorAttachment = struct_makers::attachment_info(target_image_view.view, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingInfo renderInfo = struct_makers::rendering_info(draw_extent, &colorAttachment, nullptr);

        vkCmdBeginRendering(this->buffer, &renderInfo);

        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), this->buffer);

        vkCmdEndRendering(this->buffer);
    }

    void restart(bool one_time_submit) const{
        VK_CHECK(vkResetCommandBuffer(buffer, 0));
        VkCommandBufferBeginInfo begin_info = struct_makers::command_buffer_begin_info(
            one_time_submit ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : 0
        );
        VK_CHECK(vkBeginCommandBuffer(buffer, &begin_info));
    }

    void barrier(
        Image &img,
        bool discard_current_data,
        VkImageLayout new_layout,
        VkPipelineStageFlags2 src_stage_mask,
        VkAccessFlags2 src_access_mask,
        VkPipelineStageFlags2 dst_stage_mask,
        VkAccessFlags2 dst_access_mask,
        ImageAspects aspects,
        uint32_t base_mip_level  = 0,
        uint32_t mip_level_count = VK_REMAINING_MIP_LEVELS) const
    {
        VkImageMemoryBarrier2 image_barrier {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = src_stage_mask,
            .srcAccessMask = src_access_mask,
            .dstStageMask = dst_stage_mask,
            .dstAccessMask = dst_access_mask,
            .oldLayout = discard_current_data ? VK_IMAGE_LAYOUT_UNDEFINED : img.layout,
            .newLayout = new_layout,
            .image = img.vk_image,
            .subresourceRange = {
                .aspectMask     = static_cast<VkImageAspectFlags>(aspects),
                .baseMipLevel   = base_mip_level,
                .levelCount     = mip_level_count,
                .baseArrayLayer = 0,
                .layerCount     = VK_REMAINING_ARRAY_LAYERS,
            },
        };


        VkDependencyInfo dep_info {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .imageMemoryBarrierCount = 1, //todo performance: function that lets you do multiple transitions in one vkCmdPipelineBarrier2 call
            .pImageMemoryBarriers = &image_barrier,
        };

        vkCmdPipelineBarrier2(this->buffer, &dep_info);

        img.layout = new_layout;
    }

    void blit(
        const Image &source,
        const Image &destination,
        glm::ivec2 src_top_left,
        glm::ivec2 src_bottom_right,
        glm::ivec2 dst_top_left,
        glm::ivec2 dst_bottom_right,
        ImageAspects aspects) const // should use VkCmdCopyImage instead? check best practices
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

    void bind_descriptor_sets(const ComputePipeline &pipeline, std::span<DescriptorSet> sets) const{
        const int max_sets = 32;
        if (sets.size() > max_sets){
            std::println("Error: cannot have more than {} descriptor sets in bind_descriptor_sets.", max_sets);
            abort();
        };
        VkDescriptorSet vk_sets[max_sets];
        for(int i = 0; i < sets.size(); ++i){
            vk_sets[i] = sets[i].set;
        }

	vkCmdBindDescriptorSets(this->buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, sets.size(), vk_sets, 0, nullptr);
    }
    void bind_descriptor_sets(ComputePipeline pipeline, DescriptorSet sets) const{
        bind_descriptor_sets(pipeline, std::span<DescriptorSet>(&sets, 1));
    }

    void dispatch(uint32_t x, uint32_t y, uint32_t z) const{
	vkCmdDispatch(this->buffer, x, y, z);
    }

    void end() const{
	VK_CHECK(vkEndCommandBuffer(this->buffer));
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
    struct SwapchainTrackingInfo{VkSwapchainKHR swapchain; std::vector<ImageView> image_views;};
    std::vector<SwapchainTrackingInfo> created_swapchains;
    struct ImageTrackingInfo{VkImage image; VmaAllocation allocation;};
    std::vector<ImageTrackingInfo> created_images;
    std::unordered_set<VkImageView> created_image_views;
    std::vector<VkCommandPool> created_command_pools;
    std::vector<VkFence> created_fences;
    std::vector<VkSemaphore> created_semaphores;
    std::vector<VkDescriptorSetLayout> layouts_to_delete;
    std::vector<ComputePipeline> created_compute_pipelines;

 public:
    VulkanEngine(const VulkanEngine &) = delete;
    VulkanEngine(VulkanEngine &&) = delete;
    VulkanEngine &operator=(const VulkanEngine &) = delete;
    VulkanEngine &operator=(VulkanEngine &&) = delete;

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

        for(auto &compute_pipeline : created_compute_pipelines){
            vkDestroyPipelineLayout(device, compute_pipeline.layout, nullptr);
        }
        for(auto &compute_pipeline : created_compute_pipelines){
            vkDestroyPipeline(device, compute_pipeline.pipeline, nullptr);
        }

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
        for(auto &img_view : created_image_views){
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


    void init_imgui(SDL_Window *window, const Swapchain &swapchain) const{
        ImGui::CreateContext();
        ImGui_ImplSDL3_InitForVulkan(window);

        // this initializes imgui for Vulkan
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.ApiVersion = api_version.to_vk_enum();
        init_info.Instance = vk_instance;
        init_info.PhysicalDevice = physical_device;
        init_info.Device = device;
        init_info.Queue = graphics_queue;
        init_info.QueueFamily = graphics_queue_family;
        init_info.DescriptorPool = nullptr;
        init_info.DescriptorPoolSize = 1000; // Probably overkill
        init_info.MinImageCount = 3;
        init_info.ImageCount = 3;
        init_info.UseDynamicRendering = true;
        //init_info.MinAllocationSize = 1024*1024; would stop best practices validation warning and waste some memory
        init_info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
        init_info.PipelineInfoMain.PipelineRenderingCreateInfo.pNext = nullptr;
        init_info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
        init_info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchain.image_format;
        init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        ImGui_ImplVulkan_Init(&init_info);
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

    Image create_image(
        glm::uvec2 extent,
        VkFormat format,
        VkImageUsageFlags image_usage_flags,
        VkMemoryPropertyFlagBits memory_property_flags)
    {
        Image image(
            allocator, {extent.x, extent.y}, format, image_usage_flags, memory_property_flags
        );
        created_images.push_back(ImageTrackingInfo{.image=image.vk_image, .allocation=image.allocation});
        return image;
    }

    ImageView create_image_view(
        const Image &img,
        ImageAspects aspects,
        uint32_t base_mip_level,
        uint32_t mip_level_count)
    {
        ImageView view(device, img, aspects, base_mip_level, mip_level_count);
        created_image_views.insert(view.view);
        return view;
    }

    CommandPool create_pool(){
        CommandPool pool(device, graphics_queue_family);
        created_command_pools.push_back(pool.pool);
        return pool;
    }

    GpuFence create_fence(bool signaled){
        GpuFence fence(device, signaled);
        created_fences.push_back(fence.fence);
        return fence;
    }

    GpuSemaphore create_semaphore(){
        GpuSemaphore sema(device);
        created_semaphores.push_back(sema.semaphore);
        return sema;
    }

    ComputePipeline create_compute_pipeline(const std::span<DescriptorSet> descriptors, ShaderModule module){
        ComputePipeline pipeline(device, descriptors, module);
        created_compute_pipelines.push_back(pipeline);
        return pipeline;
    }

    ComputePipeline create_compute_pipeline(const std::span<DescriptorSet> descriptors, std::string_view shader_filepath){
        ComputePipeline pipeline(device, descriptors, shader_filepath);
        created_compute_pipelines.push_back(pipeline);
        return pipeline;
    }

    ComputePipeline create_compute_pipeline(DescriptorSet descriptors, ShaderModule module){
        return create_compute_pipeline(std::span<DescriptorSet>(&descriptors, 1), module);
    }

    ComputePipeline create_compute_pipeline(DescriptorSet descriptors, std::string_view shader_filepath){
        return create_compute_pipeline(std::span<DescriptorSet>(&descriptors, 1), shader_filepath);
    }

    [[nodiscard]] CommandBuffer create_command_buffer(CommandPool pool) const{
        return {device, pool.pool};
    }

    DescriptorSet allocate_descriptor_set(VkDescriptorSetLayout layout) const{
        return {device, layout, descriptor_pool};
    }

    void update_storage_image_descriptor(DescriptorSet set, std::span<ImageView> views, uint32_t bind) const{
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

    void update_storage_image_descriptor(DescriptorSet &set, ImageView &view, uint32_t bind) const{
        update_storage_image_descriptor(set, std::span<ImageView>(&view, 1), bind);
    }

    void wait(GpuFence &fence) const{
        VK_CHECK(vkWaitForFences(device, 1, &fence.fence, (VkBool32)true, timeout_length));
        VK_CHECK(vkResetFences(device, 1, &fence.fence));
    }

    // Returns the index of the next image in the swapchain
    [[nodiscard]] uint32_t acquire_next_image(const Swapchain &swapchain, GpuSemaphore signal_sema) const{
        uint32_t swapchain_image_index;
        VK_CHECK(vkAcquireNextImageKHR(device, swapchain.swapchain, timeout_length, signal_sema.semaphore, nullptr, &swapchain_image_index));
        return swapchain_image_index;
    }

    void submit_commands(
        CommandBuffer cmd_buffer,
        GpuSemaphore wait_sema,//todo: use std::optional for cases where we don't signal/wait
        VkPipelineStageFlagBits2 wait_stage_mask, // Commands in cmd_buffer that use these stages will not run until wait_sema is signaled
        GpuSemaphore signal_sema, // Will be signaled when every command in cmd_buffer is complete
        VkPipelineStageFlagBits2 signal_stage_mask, // Stages that wait on the signal_sema will have access to writes done by commands in cmd_buffer (other stages will have outdated cached data, causing errors)
        GpuFence signal_fence) const
    {
        VkCommandBufferSubmitInfo cmd_info = struct_makers::command_buffer_submit_info(cmd_buffer.buffer);
        VkSemaphoreSubmitInfo wait_info = struct_makers::semaphore_submit_info(wait_stage_mask, wait_sema.semaphore);
        VkSemaphoreSubmitInfo signal_info = struct_makers::semaphore_submit_info(signal_stage_mask, signal_sema.semaphore);
        VkSubmitInfo2 submit_info = struct_makers::submit_info(&cmd_info,&signal_info,&wait_info);
        VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit_info, signal_fence.fence));
    }

    void present(const Swapchain& swapchain, GpuSemaphore wait_sema, uint32_t swapchain_image_index) const{
        VkPresentInfoKHR present_info = {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &wait_sema.semaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchain.swapchain,
            .pImageIndices = &swapchain_image_index,
            .pResults{},
        };
        VK_CHECK(vkQueuePresentKHR(graphics_queue, &present_info));
    }

};

// This is what you're meant to use instead of Layouts and Layout Bindings. You can
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

        return engine.allocate_descriptor_set(layout);
    }
};

export struct FrameData {
    GpuSemaphore swapchain_semaphore;
    GpuSemaphore render_semaphore;
    GpuFence render_fence;
    CommandPool command_pool;
    CommandBuffer main_command_buffer;

    FrameData(VulkanEngine &engine)
    :swapchain_semaphore(engine.create_semaphore()),
     render_semaphore(engine.create_semaphore()),
     render_fence(engine.create_fence(true)),
     command_pool(engine.create_pool()),
     main_command_buffer(engine.create_command_buffer(this->command_pool))
    { }
};












































