module;


#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3/SDL_events.h>

#include <VkBootstrap.h>
#include <vulkan/vk_enum_string_helper.h>

#include <array>
#include <thread>
#include <chrono>
#include <cmath>
#include <print>
#include <fstream>
#include <functional>


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

export module vulkanRenderer;

#define VK_CHECK(x)                                                         \
    do{                                                                     \
        VkResult err = x;                                                   \
        if (err) {                                                          \
             std::print("Detected Vulkan error: {}", string_VkResult(err)); \
            abort();                                                        \
        }                                                                   \
    }while(false)                                                           \

#ifndef NDEBUG
    const bool use_validation_layers = true;
#else
    const bool use_validation_layers = true;
#endif

namespace struct_makers{
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
}

// todo: check this
static void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout)
{
    VkImageMemoryBarrier2 imageBarrier {};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    imageBarrier.pNext = nullptr;
    imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    imageBarrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
    imageBarrier.oldLayout = currentLayout;
    imageBarrier.newLayout = newLayout;
    VkImageAspectFlags aspectMask = (newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange = struct_makers::image_subresource_range(aspectMask);
    imageBarrier.image = image;

    VkDependencyInfo depInfo {};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.pNext = nullptr;
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &imageBarrier;

    vkCmdPipelineBarrier2(cmd, &depInfo);
}


static void copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D srcSize, VkExtent2D dstSize) // should use VkCmdCopyImage instead? check best practices
{
    VkImageBlit2 blit_region{};
    blit_region.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
    blit_region.pNext = nullptr;

    blit_region.srcOffsets[1].x = srcSize.width;
    blit_region.srcOffsets[1].y = srcSize.height;
    blit_region.srcOffsets[1].z = 1;

    blit_region.dstOffsets[1].x = dstSize.width;
    blit_region.dstOffsets[1].y = dstSize.height;
    blit_region.dstOffsets[1].z = 1;

    blit_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit_region.srcSubresource.baseArrayLayer = 0;
    blit_region.srcSubresource.layerCount = 1;
    blit_region.srcSubresource.mipLevel = 0;

    blit_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit_region.dstSubresource.baseArrayLayer = 0;
    blit_region.dstSubresource.layerCount = 1;
    blit_region.dstSubresource.mipLevel = 0;

    VkBlitImageInfo2 blit_info{};
    blit_info.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
    blit_info.pNext = nullptr;
    blit_info.dstImage = destination;
    blit_info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; // the format they're currently in. you should pass a struct of the image that contains this info and assert that they really are in this layout
    blit_info.srcImage = source;
    blit_info.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    blit_info.filter = VK_FILTER_LINEAR;
    blit_info.regionCount = 1;
    blit_info.pRegions = &blit_region;

    vkCmdBlitImage2(cmd, &blit_info);
}

struct FrameData {
    VkSemaphore swapchain_semaphore, render_semaphore;
    VkFence render_fence;

    VkCommandPool command_pool;
    VkCommandBuffer main_command_buffer;
};
constexpr unsigned int FRAME_OVERLAP = 2;

const struct APIVersionVulkan{
    u_int32_t major = 0;
    u_int32_t minor = 0;
    u_int32_t patch = 0;
}vkAPI_ver{.major = 1, .minor = 3, .patch = 0};

struct VulkanImageInfo{
    VkImage image;
    VkImageView image_view;
    VmaAllocation allocation;
    VkExtent3D image_extent;
    VkFormat image_format;
};


struct DescriptorLayoutBuilder {
    std::vector<VkDescriptorSetLayoutBinding> bindings; // todo reserve some amount in the constructor

    void add_binding(uint32_t binding, VkDescriptorType type);
    void clear();
    VkDescriptorSetLayout build(VkDevice device, VkShaderStageFlags shaderStages, void* pNext = nullptr, VkDescriptorSetLayoutCreateFlags flags = 0);

    DescriptorLayoutBuilder(){
        bindings.reserve(10); //TODO: is 10 a logical amount here?
    }
};

void DescriptorLayoutBuilder::add_binding(uint32_t binding, VkDescriptorType type) {
    VkDescriptorSetLayoutBinding newbind {};
    newbind.binding = binding;
    newbind.descriptorCount = 1;
    newbind.descriptorType = type;

    bindings.push_back(newbind);
}

void DescriptorLayoutBuilder::clear() {
    bindings.clear();
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device, VkShaderStageFlags shaderStages, void* pNext, VkDescriptorSetLayoutCreateFlags flags)
{
    for (auto& b : bindings){
        b.stageFlags |= shaderStages;
    }

    VkDescriptorSetLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.pNext = pNext;
    info.pBindings = bindings.data();
    info.bindingCount = bindings.size();
    info.flags = flags;

    VkDescriptorSetLayout set = nullptr;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &set));

    return set;
}

struct DescriptorAllocator{
    struct PoolSizePerDescriptorSet{
        VkDescriptorType type;
        float size;
    };

    VkDescriptorPool pool;

    void init_pool(VkDevice device, uint32_t maxSets, std::span<PoolSizePerDescriptorSet> pool_sizes_per_set);
    void clear_descriptors(VkDevice device) const;
    void destroy_pool(VkDevice device) const;

    VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout) const;
};

void DescriptorAllocator::init_pool(VkDevice device, uint32_t maxSets, std::span<PoolSizePerDescriptorSet> pool_sizes_per_set){
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (PoolSizePerDescriptorSet pool_size_mult : pool_sizes_per_set) {
        poolSizes.push_back(VkDescriptorPoolSize {
            .type = pool_size_mult.type,
            .descriptorCount = static_cast<uint32_t>(pool_size_mult.size * maxSets) // this is all really bad redo it completely
        });
    }

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = 0;
    pool_info.maxSets = maxSets;
    pool_info.poolSizeCount = (uint32_t)poolSizes.size();
    pool_info.pPoolSizes = poolSizes.data();

    vkCreateDescriptorPool(device, &pool_info, nullptr, &pool);
}

void DescriptorAllocator::clear_descriptors(VkDevice device) const{
    vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroy_pool(VkDevice device) const{
    vkDestroyDescriptorPool(device,pool,nullptr);
}

VkDescriptorSet DescriptorAllocator::allocate(VkDevice device, VkDescriptorSetLayout layout) const{
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.pNext = nullptr;
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &ds));

    return ds;
}


static bool load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule){
    // open the file. With cursor at the end
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        return false;
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

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return false;
    }
    *outShaderModule = shaderModule;
    return true;
}


export class Renderer{
public:
    VkExtent2D window_extent = {1700, 900};
    SDL_Window* window = nullptr;
    VkInstance vk_instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkSurfaceKHR surface = nullptr;
    VkPhysicalDevice physical_device = nullptr;
    VkDevice device = nullptr;
    VkQueue graphics_queue = nullptr;
    u_int32_t graphics_queue_family = 0;

    VkSwapchainKHR swapchain{};
    VkFormat swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D swapchain_extent {1700, 900};
    std::vector<VkImage> swapchain_images;
    std::vector<VkImageView> swapchain_image_views;

    FrameData frames[FRAME_OVERLAP]{};
    u_int32_t frame_number = 0;

    VmaAllocator allocator = nullptr;

    VulkanImageInfo draw_image;
    VkExtent2D draw_extent;

    DescriptorAllocator globalDescriptorAllocator;
    VkDescriptorSet drawImageDescriptors;
    VkDescriptorSetLayout drawImageDescriptorLayout;

    VkPipeline _gradientPipeline;
    VkPipelineLayout _gradientPipelineLayout;

    // immediate submit structures
    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;

    void create_swapchain(u_int32_t width, u_int32_t height){
        vkb::SwapchainBuilder swapchain_builder{physical_device, device, surface};
        vkb::Swapchain vkb_swapchain = swapchain_builder
            .set_desired_format(VkSurfaceFormatKHR{ .format = swapchain_image_format, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
            .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
            .set_desired_extent(width, height)
            .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            .build()
            .value();

        this->swapchain_extent = vkb_swapchain.extent;
        this->swapchain = vkb_swapchain.swapchain;
        this->swapchain_images = vkb_swapchain.get_images().value();
        this->swapchain_image_views = vkb_swapchain.get_image_views().value();


    };

    void destroy_swapchain(){
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        for(auto &swapchain_image_view : swapchain_image_views){
            vkDestroyImageView(device, swapchain_image_view, nullptr);
        }
    }

    void init_descriptors(){
        //create a descriptor pool that will hold 10 sets with 1 image each
        std::vector<DescriptorAllocator::PoolSizePerDescriptorSet> sizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 }
        };

        globalDescriptorAllocator.init_pool(device, 10, sizes);

        //make the descriptor set layout for our compute draw
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        drawImageDescriptorLayout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        //allocate a descriptor set for our draw image
        drawImageDescriptors = globalDescriptorAllocator.allocate(device,drawImageDescriptorLayout);

        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imgInfo.imageView = draw_image.image_view;

        VkWriteDescriptorSet drawImageWrite = {};
        drawImageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        drawImageWrite.pNext = nullptr;
        drawImageWrite.dstBinding = 0;
        drawImageWrite.dstSet = drawImageDescriptors;
        drawImageWrite.descriptorCount = 1;
        drawImageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        drawImageWrite.pImageInfo = &imgInfo;

        vkUpdateDescriptorSets(device, 1, &drawImageWrite, 0, nullptr);
    }

    void init_background_pipelines(){
        VkPipelineLayoutCreateInfo computeLayout{};
        computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        computeLayout.pNext = nullptr;
        computeLayout.pSetLayouts = &drawImageDescriptorLayout;
        computeLayout.setLayoutCount = 1;

        VK_CHECK(vkCreatePipelineLayout(device, &computeLayout, nullptr, &_gradientPipelineLayout));

        VkShaderModule computeShader;
        if (!load_shader_module("shaders/compiled/gradient.comp.spv", device, &computeShader)){
                std::print("Error when building the compute shader\n");
        }

        VkPipelineShaderStageCreateInfo stageinfo{};
        stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageinfo.pNext = nullptr;
        stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageinfo.module = computeShader;
        stageinfo.pName = "main";

        VkComputePipelineCreateInfo computePipelineCreateInfo{};
        computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        computePipelineCreateInfo.pNext = nullptr;
        computePipelineCreateInfo.layout = _gradientPipelineLayout;
        computePipelineCreateInfo.stage = stageinfo;

        VK_CHECK(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&computePipelineCreateInfo, nullptr, &_gradientPipeline));
        vkDestroyShaderModule(device, computeShader, nullptr);
    }

    void init_pipelines(){
        init_background_pipelines();
    }


    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
    {
        VK_CHECK(vkResetFences(device, 1, &_immFence));
        VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

        VkCommandBuffer cmd = _immCommandBuffer;

        VkCommandBufferBeginInfo cmdBeginInfo = struct_makers::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

        VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

        function(cmd);

        VK_CHECK(vkEndCommandBuffer(cmd));

        VkCommandBufferSubmitInfo cmdinfo = struct_makers::command_buffer_submit_info(cmd);
        VkSubmitInfo2 submit = struct_makers::submit_info(&cmdinfo, nullptr, nullptr);

        // submit command buffer to the queue and execute it.
        //  _renderFence will now block until the graphic commands finish execution
        VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, _immFence));

        VK_CHECK(vkWaitForFences(device, 1, &_immFence, true, 9999999999));
    }


void init_imgui() {
    ImGui::CreateContext();
    ImGui_ImplSDL3_InitForVulkan(window);

    // this initializes imgui for Vulkan
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.ApiVersion = VK_MAKE_VERSION(vkAPI_ver.major, vkAPI_ver.minor, vkAPI_ver.patch);
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
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchain_image_format;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&init_info);
}

    Renderer(){
        SDL_Init(SDL_INIT_VIDEO);
        SDL_WindowFlags window_flags = SDL_WINDOW_VULKAN;

        window = SDL_CreateWindow(
            "Vulkan app",
            window_extent.width,
            window_extent.height,
            window_flags
        );

        // Init instance, messenger, surface
        vkb::InstanceBuilder builder;
        vkb::Instance vkb_inst = builder.set_app_name("Vulkan App")
            .request_validation_layers(use_validation_layers)
            .use_default_debug_messenger()
            .require_api_version(vkAPI_ver.major, vkAPI_ver.minor, vkAPI_ver.patch)
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
            .set_minimum_version(vkAPI_ver.major, vkAPI_ver.minor)
            .set_required_features_13(features13)
            .set_required_features_12(features12)
            .set_surface(surface)
            .select()
            .value();

        vkb::DeviceBuilder device_builder{vkb_physical_device};
        vkb::Device vkb_device = device_builder.build().value();
        this->device = vkb_device.device;
        this->physical_device = vkb_physical_device.physical_device;
        this->graphics_queue = vkb_device.get_queue(vkb::QueueType::graphics).value();
        this->graphics_queue_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

        if (vkb_device.get_queue(vkb::QueueType::present).value() != graphics_queue){
            std::println("Error: device does not support shared graphics and present queues.");
        }

        // Initialize VMA
        VmaAllocatorCreateInfo allocatorInfo{
            .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
            .physicalDevice = physical_device,
            .device = device,
            .preferredLargeHeapBlockSize = 0,
            .pAllocationCallbacks = nullptr,
            .pDeviceMemoryCallbacks = nullptr,
            .pHeapSizeLimit = nullptr,
            .pVulkanFunctions = nullptr,
            .instance = vk_instance,
            .vulkanApiVersion = VK_MAKE_VERSION(vkAPI_ver.major, vkAPI_ver.minor, vkAPI_ver.patch),
            .pTypeExternalMemoryHandleTypes = nullptr,
        };
        vmaCreateAllocator(&allocatorInfo, &allocator);

        create_swapchain(window_extent.width, window_extent.height);

        // Initialize the Draw Image
        draw_image.image_format = VK_FORMAT_R16G16B16A16_SFLOAT;
	draw_image.image_extent = {
		.width = window_extent.width,
		.height = window_extent.height,
		.depth = 1
	}; 

	VkImageUsageFlags draw_image_usages{};
	draw_image_usages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	draw_image_usages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	draw_image_usages |= VK_IMAGE_USAGE_STORAGE_BIT;
	draw_image_usages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	VkImageCreateInfo img_info = struct_makers::image_create_info(draw_image.image_format, draw_image_usages, draw_image.image_extent);

	VmaAllocationCreateInfo img_allocinfo = {};
	img_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	img_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	vmaCreateImage(allocator, &img_info, &img_allocinfo, &draw_image.image, &draw_image.allocation, nullptr);

	VkImageViewCreateInfo rview_info = struct_makers::imageview_create_info(draw_image.image_format, draw_image.image, VK_IMAGE_ASPECT_COLOR_BIT);
	VK_CHECK(vkCreateImageView(device, &rview_info, nullptr, &draw_image.image_view));

        // Initialize command buffers
        VkCommandPoolCreateInfo commandPoolInfo = struct_makers::command_pool_create_info(graphics_queue_family, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

        for (auto & frame : frames) {
            VK_CHECK(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &frame.command_pool));
            VkCommandBufferAllocateInfo cmdAllocInfo = struct_makers::command_buffer_allocate_info(frame.command_pool, 1);
            VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &frame.main_command_buffer));
        }

        // Initialize fence and semaphores
        VkFenceCreateInfo fence_create_info = struct_makers::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
        VkSemaphoreCreateInfo semaphore_create_info = struct_makers::semaphore_create_info(0);
	for (auto &frame : frames) {
            VK_CHECK(vkCreateFence(device, &fence_create_info, nullptr, &frame.render_fence));
            VK_CHECK(vkCreateSemaphore(device, &semaphore_create_info, nullptr, &frame.swapchain_semaphore));
            VK_CHECK(vkCreateSemaphore(device, &semaphore_create_info, nullptr, &frame.render_semaphore));
	}

        // immediate submit initialization
        VK_CHECK(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &_immCommandPool));
        // allocate the command buffer for immediate submits
        VkCommandBufferAllocateInfo cmdAllocInfo = struct_makers::command_buffer_allocate_info(_immCommandPool, 1);
        VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &_immCommandBuffer));

        init_imgui();

        init_descriptors();
        init_pipelines();
    }

    ~Renderer(){
        vkDeviceWaitIdle(device);

        vkDestroyPipelineLayout(device, _gradientPipelineLayout, nullptr);
        vkDestroyPipeline(device, _gradientPipeline, nullptr);

        globalDescriptorAllocator.destroy_pool(device);
        vkDestroyDescriptorSetLayout(device, drawImageDescriptorLayout, nullptr);

        ImGui_ImplVulkan_Shutdown();
        // vkDestroyCommandPool(device, _immCommandPool, nullptr);

        for (auto & frame : frames) {
            vkDestroyCommandPool(device, frame.command_pool, nullptr);
            vkDestroyFence(device, frame.render_fence, nullptr);
            vkDestroySemaphore(device, frame.render_semaphore, nullptr);
            vkDestroySemaphore(device ,frame.swapchain_semaphore, nullptr);
        }
        vkDestroyImageView(device, draw_image.image_view, nullptr);
        vmaDestroyImage(allocator, draw_image.image, draw_image.allocation);
        destroy_swapchain();
        vmaDestroyAllocator(allocator);
        vkDestroySurfaceKHR(vk_instance, surface, nullptr);
        vkDestroyDevice(device, nullptr);
        vkb::destroy_debug_utils_messenger(vk_instance, debug_messenger);
        vkDestroyInstance(vk_instance, nullptr);
        SDL_DestroyWindow(window);
    }

    FrameData& get_current_frame() { return frames[frame_number % FRAME_OVERLAP]; }


    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView) const{
        VkRenderingAttachmentInfo colorAttachment = struct_makers::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingInfo renderInfo = struct_makers::rendering_info(swapchain_extent, &colorAttachment, nullptr);

        vkCmdBeginRendering(cmd, &renderInfo);

        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

        vkCmdEndRendering(cmd);
    }

    void draw_background(VkCommandBuffer cmd) const
    {
        ////make a clear-color from frame number. This will flash with a 120 frame period.
        //VkClearColorValue clearValue;
        //float flash = std::abs(std::sin(frame_number / 120.f));
        //clearValue = { { 0.0f, 0.0f, flash, 1.0f } };
        //
        //VkImageSubresourceRange clearRange = struct_makers::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
        //
        ////clear image
        //vkCmdClearColorImage(cmd, draw_image.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);

	// bind the gradient drawing compute pipeline
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _gradientPipeline);

	// bind the descriptor set containing the draw image for the compute pipeline
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _gradientPipelineLayout, 0, 1, &drawImageDescriptors, 0, nullptr);

	// execute the compute pipeline dispatch. We are using 16x16 workgroup size so we need to divide by it
	vkCmdDispatch(cmd, std::ceil(draw_extent.width / 16.0), std::ceil(draw_extent.height / 16.0), 1);
    }

    void draw(){
        VkFence render_fence = get_current_frame().render_fence;
        VK_CHECK(vkWaitForFences(device, 1, &render_fence, true, 1000000000));
        VK_CHECK(vkResetFences(device, 1, &render_fence));

        uint32_t swapchain_image_index{};
        VK_CHECK(vkAcquireNextImageKHR(device, swapchain, 1000000000, get_current_frame().swapchain_semaphore, nullptr, &swapchain_image_index));

        VkCommandBuffer cmd_buffer = get_current_frame().main_command_buffer;
        VK_CHECK(vkResetCommandBuffer(cmd_buffer, 0));
        VkCommandBufferBeginInfo cmd_buffer_begin_info = struct_makers::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	draw_extent.width = draw_image.image_extent.width;
	draw_extent.height = draw_image.image_extent.height;

	VK_CHECK(vkBeginCommandBuffer(cmd_buffer, &cmd_buffer_begin_info));

	transition_image(cmd_buffer, draw_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

	draw_background(cmd_buffer);

	transition_image(cmd_buffer, draw_image.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	transition_image(cmd_buffer, swapchain_images[swapchain_image_index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

	copy_image_to_image(cmd_buffer, draw_image.image, swapchain_images[swapchain_image_index], draw_extent, swapchain_extent);

        transition_image(cmd_buffer, swapchain_images[swapchain_image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        draw_imgui(cmd_buffer, swapchain_image_views[swapchain_image_index]);

	transition_image(cmd_buffer, swapchain_images[swapchain_image_index], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

	VK_CHECK(vkEndCommandBuffer(cmd_buffer));


        VkCommandBufferSubmitInfo cmd_info = struct_makers::command_buffer_submit_info(cmd_buffer);
        VkSemaphoreSubmitInfo wait_info = struct_makers::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,get_current_frame().swapchain_semaphore);
        VkSemaphoreSubmitInfo signal_info = struct_makers::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame().render_semaphore);
	VkSubmitInfo2 submit_info = struct_makers::submit_info(&cmd_info,&signal_info,&wait_info);
        VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit_info, get_current_frame().render_fence));

        VkPresentInfoKHR present_info = {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.pNext = nullptr;
        present_info.pSwapchains = &swapchain;
        present_info.swapchainCount = 1;
        present_info.pWaitSemaphores = &get_current_frame().render_semaphore;
        present_info.waitSemaphoreCount = 1;
        present_info.pImageIndices = &swapchain_image_index;
        VK_CHECK(vkQueuePresentKHR(graphics_queue, &present_info));

        ++frame_number;
    }

    bool stop_rendering = false;
    void run(){
        SDL_Event event;
        bool quit = false;
        while (!quit){
            while(SDL_PollEvent(&event)){
                if (event.type == SDL_EVENT_QUIT) { quit = true; }
                if (event.type == SDL_EVENT_WINDOW_MINIMIZED) { stop_rendering = true; }
                if (event.type == SDL_EVENT_WINDOW_RESTORED) { stop_rendering = false; }
                ImGui_ImplSDL3_ProcessEvent(&event);
            }

            if (stop_rendering){
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            else{
                ImGui_ImplVulkan_NewFrame();
                ImGui_ImplSDL3_NewFrame();
                ImGui::NewFrame();

                ImGui::ShowDemoWindow();

                ImGui::Render();

                draw();
            }
        }
    }

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;
};
