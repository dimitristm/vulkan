module;

#include <SDL3/SDL_video.h>
#include <boost/pfr.hpp>
#include <cassert>

// Disable harmless VMA warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#elifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#endif

#include <vk_mem_alloc.h>

#ifdef __clang__
#pragma clang diagnostic pop
#elifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <cstdio>

#if !VK_PROJ_USE_IMPORT_STD
#include <algorithm>
#include <array>
#include <deque>
#include <filesystem>
#include <functional>
#include <optional>
#include <print>
#include <span>
#include <utility>
#include <vector>
#endif

export module vulkanEngine;
import types;
import util;

import glm;
#if VK_PROJ_USE_IMPORT_STD
import std;
#endif

using util::PassKey;

class StructMaker;
export class Swapchain;
export class CommandBuffer;
export class DescriptorSet;
export class ImageView;
export class ComputePipeline;
export class PipelineLayout;
export template <typename T>
class GraphicsPipeline;

struct APIVersionVulkan {
    u32 major;
    u32 minor;
    u32 patch;

    [[nodiscard]] u32 to_vk_enum() const { return VK_MAKE_VERSION(major, minor, patch); }
};

export bool operator==(const VkExtent2D& lhs, const VkExtent2D& rhs) noexcept {
    return lhs.width == rhs.width && lhs.height == rhs.height;
}
export bool operator!=(const VkExtent2D& lhs, const VkExtent2D& rhs) noexcept { return !(lhs == rhs); }

export enum class MSAALevel : u8 {
    // todo: check physical device limits, they might not support one of these
    // Can be cast into a VkSampleCountFlagBits
    OFF = VK_SAMPLE_COUNT_1_BIT,
    X2 = VK_SAMPLE_COUNT_2_BIT,
    X4 = VK_SAMPLE_COUNT_4_BIT,
    X8 = VK_SAMPLE_COUNT_8_BIT,
};

export template <typename T>
struct PushConstant {
    static_assert(std::is_trivially_copyable_v<T>, "Push constant type must be trivially copyable");

    const u32 offset;
    const u32 size;
    const VkShaderStageFlags shader_stages;
    T data;

    PushConstant(u32 offset, VkShaderStageFlags shader_stages, const T& data = {})
        : offset(offset),
          size(sizeof(T)),
          shader_stages(shader_stages),
          data(data) {}
};

// Assumes that your push constants have scalar alignment rules on the shader side (use GL_EXT_scalar_block_layout on glsl)
// Otherwise you'll get errors about VkPushConstantRange being set incorrectly.
export class PushConstantsBuilder {
    u32 current_last_byte_used = 0;
    std::vector<VkPushConstantRange> ranges;

    static bool range_stages_do_not_overlap(const std::vector<VkPushConstantRange>& old, VkShaderStageFlags new_flags);

  public:
    PushConstantsBuilder() = default;

    template <typename T>
    PushConstant<T> add(VkShaderStageFlags shader_stages) {
        static_assert(sizeof(T) % 4 == 0, "Size of push constant must be multiple of 4");
        assert(range_stages_do_not_overlap(ranges, shader_stages));

        VkPushConstantRange range{
            .stageFlags = shader_stages,
            .offset = current_last_byte_used,
            .size = sizeof(T),
        };

        ranges.push_back(range);

        current_last_byte_used += sizeof(T);

        assert(
            current_last_byte_used <= 128 && "Assert failed: used more push constant space than is guaranteed to exist"
        );

        return PushConstant<T>{range.offset, range.stageFlags};
    }

    [[nodiscard]] const std::vector<VkPushConstantRange>& get_ranges() const { return ranges; }

    PushConstantsBuilder& reset();
};

class DeletionBatch {
    std::vector<std::function<void()>> commands;

  public:
    DeletionBatch(std::vector<std::function<void()>> cmds)
        : commands(std::move(cmds)) {}
    DeletionBatch() = default;
    void execute() {
        for (auto& cmd : commands) {
            cmd();
        }
        commands.clear();
    }
    void add(std::function<void()>&& deletion_command) { commands.push_back(std::move(deletion_command)); }
    [[nodiscard]] bool is_empty() const { return commands.empty(); }
    void clear() { commands.clear(); }
};

class DeletionBatchQueue {
    struct FencedDeletionBatch {
        DeletionBatch batch;
        VkFence fence{};
    };
    std::deque<FencedDeletionBatch> deletion_batches;

  public:
    void add(DeletionBatch&& batch, VkFence fence) {
        deletion_batches.push_back({.batch = std::move(batch), .fence = fence});
    }
    void force_flush_all() {
        for (auto& fenced_batch : deletion_batches) {
            fenced_batch.batch.execute();
        }
        deletion_batches.clear();
    }
    void flush(VkDevice device) {
        std::erase_if(deletion_batches, [&device](auto& fenced_batch) {
            const bool is_signaled = vkGetFenceStatus(device, fenced_batch.fence) == VK_SUCCESS;
            if (is_signaled) {
                fenced_batch.batch.execute();
            }
            return is_signaled;
        });
    }
};

export struct VulkanEngine;
extern VulkanEngine* g_vk;
export struct VulkanEngine {
    VkInstance vk_instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkSurfaceKHR surface{};
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue graphics_queue;
    u32 graphics_queue_family;
    VmaAllocator allocator;
    VkDescriptorPool descriptor_pool;
    APIVersionVulkan api_version{.major = 1, .minor = 3, .patch = 0};

    DeletionBatch deletions_in_this_fence;
    DeletionBatchQueue queued_deletion_batches;
    DeletionBatch& get_deletions_in_this_fence() { return deletions_in_this_fence; }
    DeletionBatchQueue& get_queued_deletion_batches() { return queued_deletion_batches; }

    void flush_queued_deletion_batches() { queued_deletion_batches.flush(device); };
    void flush_deletions_in_this_fence_immediately() { deletions_in_this_fence.execute(); };

    struct ImageTrackingInfo {
        VkImage image;
        VmaAllocation allocation;
    };
    struct ImageTrackingInfoHash {
        std::size_t operator()(const ImageTrackingInfo& info) const noexcept {
            return std::hash<VkImage>{}(info.image);
        }
    };
    struct ImageTrackingInfoEqual {
        bool operator()(const ImageTrackingInfo& a, const ImageTrackingInfo& b) const noexcept {
            return a.image == b.image;
        }
    };

    std::vector<VkDescriptorSetLayout> descriptor_set_layouts_to_delete;
    std::vector<VkPipelineLayout> pipeline_layouts_to_delete;

    bool imgui_is_initialized = false;

    VulkanEngine(const VulkanEngine&) = delete;
    VulkanEngine(VulkanEngine&&) = delete;
    VulkanEngine& operator=(const VulkanEngine&) = delete;
    VulkanEngine& operator=(VulkanEngine&&) = delete;

    VulkanEngine(SDL_Window* window);
    ~VulkanEngine();

    void init_imgui(SDL_Window* window, VkFormat image_format, MSAALevel msaa_level = MSAALevel::OFF);
    void wait_idle() const { vkDeviceWaitIdle(device); }
    [[nodiscard]] VkSurfaceCapabilitiesKHR get_surface_capabilities() const;
};

// Move-only RAII wrapper. Ensures unique ownership of a vulkan resource (non-copyable). Destructor
// queues the resource for deletion by VulkanEngine. A resource can be a VkSampler, VkImageView etc.
template <typename VkResource, auto vk_destroy_func>
struct ManagedVkResource {
    VkResource underlying = VK_NULL_HANDLE;

    ManagedVkResource() = default;
    explicit ManagedVkResource(VkResource resource)
        : underlying(resource) {}

    ManagedVkResource(const ManagedVkResource&) = delete;
    ManagedVkResource& operator=(const ManagedVkResource&) = delete;

    ManagedVkResource(ManagedVkResource&& other) noexcept
        : underlying(std::exchange(other.underlying, VK_NULL_HANDLE)) {}

    ManagedVkResource& operator=(ManagedVkResource&& other) noexcept {
        std::swap(underlying, other.underlying);
        return *this;
    }

    ~ManagedVkResource() {
        if (underlying != VK_NULL_HANDLE) {
            g_vk->get_deletions_in_this_fence().add([u = underlying]() { vk_destroy_func(g_vk->device, u, nullptr); });
        }
    }

    [[nodiscard]] VkResource get_underlying() const { return underlying; }
    [[nodiscard]] VkResource* get_underlying_ptr() { return &underlying; } // Only use to pass to a vkCreate* function
    [[nodiscard]] const VkResource* get_underlying_ptr() const {
        return &underlying;
    } // Only use for Vulkan functions expecting an array of handles

    bool operator==(const ManagedVkResource& other) const noexcept { return underlying == other.underlying; }
    bool operator!=(const ManagedVkResource& other) const noexcept { return underlying != other.underlying; }
};

// Same as ManagedVkResource but for resources that track and pass a VmaAllocation to their destroy function
template <typename VkResource, auto vma_destroy_func>
struct ManagedVmaResource {
    VkResource underlying = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;

    ManagedVmaResource() = default;
    ManagedVmaResource(VkResource resource, VmaAllocation alloc)
        : underlying(resource),
          allocation(alloc) {}

    ManagedVmaResource(const ManagedVmaResource&) = delete;
    ManagedVmaResource& operator=(const ManagedVmaResource&) = delete;

    ManagedVmaResource(ManagedVmaResource&& other) noexcept
        : underlying(std::exchange(other.underlying, VK_NULL_HANDLE)),
          allocation(std::exchange(other.allocation, VK_NULL_HANDLE)) {}

    ManagedVmaResource& operator=(ManagedVmaResource&& other) noexcept {
        std::swap(underlying, other.underlying);
        std::swap(allocation, other.allocation);
        return *this;
    }

    ~ManagedVmaResource() {
        if (allocation != VK_NULL_HANDLE) {
            g_vk->get_deletions_in_this_fence().add([u = underlying, a = allocation]() {
                vma_destroy_func(g_vk->allocator, u, a);
            });
        }
    }

    [[nodiscard]] VkResource get_underlying() const { return underlying; }
    [[nodiscard]] VkResource* get_underlying_ptr() { return &underlying; } // Only use to pass to a vkCreate* function
    [[nodiscard]] const VkResource* get_underlying_ptr() const {
        return &underlying;
    } // Only use for Vulkan functions expecting an array of handles
    [[nodiscard]] VmaAllocation get_allocation() const { return allocation; }
    [[nodiscard]] VmaAllocation* get_allocation_ptr() {
        return &allocation;
    } // Only use to pass to a vkCreate* function
    [[nodiscard]] const VmaAllocation* get_allocation_ptr() const {
        return &allocation;
    } // Only use for Vulkan functions expecting an array of handles

    bool operator==(const ManagedVmaResource& other) const noexcept { return underlying == other.underlying; }
    bool operator!=(const ManagedVmaResource& other) const noexcept { return underlying != other.underlying; }
};

export class GpuFence {
    ManagedVkResource<VkFence, vkDestroyFence> fence;

  public:
    [[nodiscard]] VkFence get_underlying(PassKey<Swapchain, CommandBuffer> /* key */) const {
        return fence.get_underlying();
    }
    // by default we wait forever, if we change the timeout we crash with VK_TIMEOUT if we run out of time
    // todo: try_wait for returning instead of crashing
    static constexpr u64 default_timeout = UINT64_MAX;

    GpuFence() = delete;

    GpuFence(bool signaled);
    explicit GpuFence(VkFence fence)
        : fence(fence) {}

    void wait(u64 timeout = default_timeout) const;
    void reset() const;
    void wait_and_reset(u64 timeout = default_timeout) const;
    [[nodiscard]] bool is_signaled() const;

    static void wait(std::span<const VkFence> fences, bool wait_all, u64 timeout = default_timeout);
    static void reset(std::span<const VkFence> fences);
    static void wait_and_reset(std::span<const VkFence> fences, bool wait_all, u64 timeout = default_timeout);
};

export class Sampler {
    ManagedVkResource<VkSampler, vkDestroySampler> sampler;

  public:
    [[nodiscard]] VkSampler get_underlying(PassKey<DescriptorSet> /* key */) const { return sampler.get_underlying(); }
    Sampler() = delete;

    Sampler(
        VkFilter mag_filter, VkFilter min_filter, VkSamplerMipmapMode mipmap_mode, VkBool32 anisotropy_enable,
        f32 max_anisotropy, VkSamplerAddressMode address_modeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VkSamplerAddressMode address_modeV = VK_SAMPLER_ADDRESS_MODE_REPEAT, VkBool32 compare_enable = VK_FALSE,
        VkCompareOp compare_op = VK_COMPARE_OP_ALWAYS, f32 mip_lod_bias = 0, f32 min_lod = 0.0f,
        f32 max_lod = VK_LOD_CLAMP_NONE, VkBorderColor border_color = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
        VkBool32 unnormalized_coordinates = VK_FALSE,
        VkSamplerAddressMode address_modeW = VK_SAMPLER_ADDRESS_MODE_REPEAT
    );
};

export enum class ImageAspects : VkImageAspectFlags {
    NONE = 0,
    COLOR = VK_IMAGE_ASPECT_COLOR_BIT,
    DEPTH = VK_IMAGE_ASPECT_DEPTH_BIT,
    STENCIL = VK_IMAGE_ASPECT_STENCIL_BIT,
    DEPTH_STENCIL = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
};

export class Image {
    ManagedVmaResource<VkImage, vmaDestroyImage> vk_image;
    VkExtent2D extent;
    VkFormat format;
    u32 mip_level_count;
    u32 layer_count;
    MSAALevel msaa_level;

  public:
    [[nodiscard]] VkImage get_underlying(PassKey<CommandBuffer, ImageView, StructMaker> /* key */) const {
        return vk_image.get_underlying();
    }
    [[nodiscard]] const VkExtent2D& get_extent() const { return extent; }
    [[nodiscard]] VkFormat get_format() const { return format; }
    [[nodiscard]] u32 get_mip_level_count() const { return mip_level_count; }
    [[nodiscard]] u32 get_layer_count() const { return layer_count; }
    [[nodiscard]] MSAALevel get_msaa_level() const { return msaa_level; }

    Image() = delete;

    Image(
        VkExtent2D extent, VkFormat format, VkImageUsageFlags image_usage_flags,
        VkMemoryPropertyFlagBits memory_property_flags, MSAALevel msaa_level, u32 mip_level_count, u32 layer_count
    );

    Image(VkImage img, VkExtent2D extent, VkFormat format, u32 layer_count);

    bool operator==(const Image& other) const noexcept { return vk_image == other.vk_image; }

    bool operator!=(const Image& other) const noexcept { return !(*this == other); }
};

export class ImageView {
    ManagedVkResource<VkImageView, vkDestroyImageView> view;

  public:
    [[nodiscard]] VkImageView get_underlying(PassKey<CommandBuffer, DescriptorSet, StructMaker> /* key */) const {
        return view.get_underlying();
    }

    ImageView() = delete;

    ImageView(const Image& img, ImageAspects aspects, u32 base_mip_level, u32 mip_level_count);

    ImageView(VkImageView vk_view);

    bool operator==(const ImageView& other) const noexcept { return view == other.view; }

    bool operator!=(const ImageView& other) const noexcept { return !(*this == other); }
};

export class GpuSemaphore {
    ManagedVkResource<VkSemaphore, vkDestroySemaphore> semaphore;

  public:
    [[nodiscard]] VkSemaphore get_underlying(PassKey<Swapchain, CommandBuffer> /* key */) const {
        return semaphore.get_underlying();
    }

    GpuSemaphore();
};

export class Swapchain {
    VkSwapchainKHR swapchain{};
    VkFormat image_format{};
    VkExtent2D extent{};
    std::vector<Image> images;
    std::vector<ImageView> image_views;
    std::vector<GpuFence> present_fences;

    void
    initialize_swapchain(SDL_Window* window, VkPresentModeKHR present_mode, VkSwapchainKHR* old_swapchain = nullptr);

  public:
    [[nodiscard]] const std::vector<Image>& get_images() const { return images; }
    [[nodiscard]] const std::vector<ImageView>& get_image_views() const { return image_views; }
    [[nodiscard]] const VkExtent2D& get_extent() const { return extent; }
    [[nodiscard]] VkFormat get_format() const { return image_format; }

    Swapchain(SDL_Window* window, VkPresentModeKHR present_mode);
    ~Swapchain();
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&&) = delete;
    Swapchain& operator=(Swapchain&&) = delete;

    // Returns the index of the next image in the swapchain and the result can be:
    // VK_SUCCESS (proceed normally)
    // VK_ERROR_OUT_OF_DATE_KHR (must remake swapchain and call acquire again)
    // VK_SUBOPTIMAL_KHR (must use the swapchain image normally and then remake the swapchain at a later time)
    // the non-VK_SUCCESS results are typically returned when things like the window being resized or moved to another monitor happen.
    [[nodiscard]] std::tuple<u32, VkResult> acquire_next_image(const GpuSemaphore& signal_sema) const;

    // Only returns VK_SUCCESS, VK_ERROR_OUT_OF_DATE_KHR, or VK_SUBOPTIMAL_KHR
    // Synchronization primitives are affected even if it errors, do not attempt to present again on error
    [[nodiscard]] VkResult present(const GpuSemaphore& wait_sema, u32 swapchain_image_index);
    // Returns true only if a valid swapchain that can be presented to the window can be made.
    // This is true unless the window size or surface extent is {0,0}.
    [[nodiscard]] static bool presentable_swapchain_exists(SDL_Window* window);
    // If there is no VkExtent2D, that means we can't make a swapchain because either the window size or the
    // surface extent is {0,0}.
    [[nodiscard]] static std::optional<VkExtent2D> decide_extent(SDL_Window* window);

    void rebuild_swapchain(SDL_Window* window, VkPresentModeKHR present_mode);
};

export class CommandPool {
    ManagedVkResource<VkCommandPool, vkDestroyCommandPool> pool;

  public:
    [[nodiscard]] VkCommandPool get_underlying(PassKey<StructMaker> /* key */) const { return pool.get_underlying(); }

    CommandPool();
};

export class VulkanBuffer {
    ManagedVmaResource<VkBuffer, vmaDestroyBuffer> buffer;
    u64 capacity_in_bytes = 0;

  protected:
    VkDeviceAddress device_address = 0;

  public:
    [[nodiscard]] VkBuffer get_underlying(PassKey<CommandBuffer, DescriptorSet, StructMaker> /* key */) const {
        return buffer.get_underlying();
    }
    [[nodiscard]] VmaAllocation get_allocation() const { return buffer.get_allocation(); }
    [[nodiscard]] u64 get_capacity_in_bytes() const { return capacity_in_bytes; }
    [[nodiscard]] const VkDeviceAddress& get_device_address() const { return device_address; }

    VulkanBuffer() = delete;

    VulkanBuffer(
        u64 capacity_in_bytes, VkBufferUsageFlags usage_flags, VmaAllocationCreateFlags vma_flags,
        VkMemoryPropertyFlags memory_property_flags_required, VkMemoryPropertyFlags memory_property_flags_preferred = 0
    );

    bool operator==(const VulkanBuffer& other) const { return buffer == other.buffer; }

    bool operator!=(const VulkanBuffer& other) const { return !(*this == other); }
};

// A GPU-side buffer for the gpu to read from and write to.
export class StorageBuffer : public VulkanBuffer {
  public:
    StorageBuffer(u64 size_in_bytes, bool is_transfer_source, bool is_transfer_dest);
};

// Used to get data from Host to Device. For performance reasons, host should write into it sequentially, not at random.
// Copy the contents into a buffer that resides in the Device instead of accessing it from the device directly.
export class StagingBuffer : public VulkanBuffer {
    void* mapped_data;

  public:
    [[nodiscard]] void* get_mapped_data() const {
        return mapped_data;
    } // remember that if you ever have to add defragmentation or begin unmapping/remapping it you'll have to instead fetch this with vmaGetAllocationInfo every time because it might change

    StagingBuffer(u64 size_in_bytes);
};

// The opposite of a staging buffer, can be used to copy gpu resources to the cpu or the gpu might store its result in it directly
// It's fine for the host to access it in whatever way it pleases, but access will be slow for the device (unless we're on a UMA machine)
export class ReadbackBuffer : public VulkanBuffer {
    void* mapped_data;

  public:
    [[nodiscard]] void* get_mapped_data() const {
        return mapped_data;
    } // remember that if you ever have to add defragmentation or begin unmapping/remapping it you'll have to instead fetch this with vmaGetAllocationInfo every time because it might change

    ReadbackBuffer(u64 size_in_bytes);
};

export class IndexBuffer : public VulkanBuffer {
  public:
    static const VkIndexType index_type = VK_INDEX_TYPE_UINT32;
    IndexBuffer(u64 total_indexes);
};

export template <typename VertexStruct>
class VertexBuffer : public VulkanBuffer {
  public:
    static const std::size_t vertex_attribute_count = boost::pfr::tuple_size_v<VertexStruct>;
    static const u32 stride = sizeof(VertexStruct);

    VertexBuffer(u64 element_count)
        : VulkanBuffer(
              sizeof(VertexStruct) * element_count,
              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, 0
          ) {
        static_assert(
            vertex_attribute_count <= 16,
            "Use only up to 16 vertex attributes as some GPUs don't support more than that."
        );
    }

    template <typename T>
    static constexpr VkFormat
    get_vertex_attribute_format(const std::string_view field_name, [[maybe_unused]] const T& field_value) {
        using U = std::remove_cvref_t<T>;

        // clang-format off
        // This list contains only widely supported formats.
        if constexpr      (std::is_same_v<U, i8_norm_t>)     return VK_FORMAT_R8_SNORM;
        else if constexpr (std::is_same_v<U, i16_norm_t>)    return VK_FORMAT_R16_SNORM;
        else if constexpr (std::is_same_v<U, i8>)            return VK_FORMAT_R8_SINT;
        else if constexpr (std::is_same_v<U, i16>)           return VK_FORMAT_R16_SINT;
        else if constexpr (std::is_same_v<U, i32>)           return VK_FORMAT_R32_SINT;
        else if constexpr (std::is_same_v<U, u8_norm_t>)     return VK_FORMAT_R8_UNORM;
        else if constexpr (std::is_same_v<U, u16_norm_t>)    return VK_FORMAT_R16_UNORM;
        else if constexpr (std::is_same_v<U, u8>)            return VK_FORMAT_R8_UINT;
        else if constexpr (std::is_same_v<U, u16>)           return VK_FORMAT_R16_UINT;
        else if constexpr (std::is_same_v<U, u32>)           return VK_FORMAT_R32_UINT;
        else if constexpr (std::is_same_v<U, f32>)           return VK_FORMAT_R32_SFLOAT;
        // else if constexpr (std::is_same_v<U, f64>)           return VK_FORMAT_R64_SFLOAT; unsupported by most hardware (~27% support on gpuinfo.org)

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

        else if constexpr (std::is_same_v<U, i8vec2>)        return VK_FORMAT_R8G8_SINT;
        else if constexpr (std::is_same_v<U, i8vec3>)        return VK_FORMAT_R8G8B8_SINT; // ~82%
        else if constexpr (std::is_same_v<U, i8vec4>)        return VK_FORMAT_R8G8B8A8_SINT;
        else if constexpr (std::is_same_v<U, i16vec2>)       return VK_FORMAT_R16G16_SINT;
        else if constexpr (std::is_same_v<U, i16vec3>)       return VK_FORMAT_R16G16B16_SINT; // ~78%
        else if constexpr (std::is_same_v<U, i16vec4>)       return VK_FORMAT_R16G16B16A16_SINT;
        else if constexpr (std::is_same_v<U, ivec2>)         return VK_FORMAT_R32G32_SINT;
        else if constexpr (std::is_same_v<U, ivec3>)         return VK_FORMAT_R32G32B32_SINT;
        else if constexpr (std::is_same_v<U, ivec4>)         return VK_FORMAT_R32G32B32A32_SINT;
        else if constexpr (std::is_same_v<U, u8vec2>)        return VK_FORMAT_R8G8_UINT;
        else if constexpr (std::is_same_v<U, u8vec3>)        return VK_FORMAT_R8G8B8_UINT; // ~82%
        else if constexpr (std::is_same_v<U, u8vec4>)        return VK_FORMAT_R8G8B8A8_UINT;
        else if constexpr (std::is_same_v<U, u16vec2>)       return VK_FORMAT_R16G16_UINT;
        else if constexpr (std::is_same_v<U, u16vec3>)       return VK_FORMAT_R16G16B16_UINT; // ~78%
        else if constexpr (std::is_same_v<U, u16vec4>)       return VK_FORMAT_R16G16B16A16_UINT;
        else if constexpr (std::is_same_v<U, uvec2>)         return VK_FORMAT_R32G32_UINT;
        else if constexpr (std::is_same_v<U, uvec3>)         return VK_FORMAT_R32G32B32_UINT;
        else if constexpr (std::is_same_v<U, uvec4>)         return VK_FORMAT_R32G32B32A32_UINT;

        else if constexpr (std::is_same_v<U, fvec2>)         return VK_FORMAT_R32G32_SFLOAT;
        else if constexpr (std::is_same_v<U, fvec3>)         return VK_FORMAT_R32G32B32_SFLOAT;
        else if constexpr (std::is_same_v<U, fvec4>)         return VK_FORMAT_R32G32B32A32_SFLOAT;

        else if constexpr (std::is_same_v<U, i32_A2R10G10B10_t>)      return VK_FORMAT_A2R10G10B10_SINT_PACK32;
        else if constexpr (std::is_same_v<U, i32_A2R10G10B10_norm_t>) return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
        else if constexpr (std::is_same_v<U, u32_A2R10G10B10_t>)      return VK_FORMAT_A2R10G10B10_UINT_PACK32;
        else if constexpr (std::is_same_v<U, u32_A2R10G10B10_norm_t>) return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
        // clang-format on

        std::string name{field_name};
        std::println(
            stderr,
            "Your vertex struct conainted a field named {} which has a type that isn't supported as a vertex "
            "attribute.",
            name
        );
        assert(false);
        abort();
        return VK_FORMAT_UNDEFINED;
    }

    [[nodiscard]] static constexpr std::array<VkVertexInputAttributeDescription, vertex_attribute_count>
    get_vertex_attribute_descriptions(u32 binding = 0, u32 first_location = 0) {
        std::array<VkVertexInputAttributeDescription, vertex_attribute_count> vertex_attribute_descriptions;
        u32 curr_location = first_location;
        u32 curr_offset = 0;
        i32 i = 0;
        boost::pfr::for_each_field_with_name(VertexStruct{}, [&](std::string_view name, const auto& value) {
            u32 alignment = alignof(std::remove_cvref_t<decltype(value)>);
            curr_offset =
                (curr_offset + alignment - 1) &
                ~(
                    alignment - 1
                ); // This is a classic trick for rounding curr_offset up to the nearest multiple of "alignment" (if alignment is a power of 2)
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

export class DescriptorSet {
    VkDescriptorSet set;
    VkDescriptorSetLayout layout; // todo is there an actual performance benefit to sharing layouts?
  public:
    DescriptorSet(VkDescriptorSetLayout layout);

    [[nodiscard]] VkDescriptorSet get_underlying(PassKey<CommandBuffer> /* key */) const { return set; }
    [[nodiscard]] VkDescriptorSetLayout get_layout(PassKey<PipelineLayout> /* key */) const { return layout; }
    void update_storage_images(u32 bind, std::span<const ImageView> views);
    void update_storage_image(u32 bind, const ImageView& view);
    void update_sampled_images(u32 bind, std::span<const ImageView> views);
    void update_sampled_image(u32 bind, const ImageView& view);
    void update_samplers(u32 bind, std::span<const Sampler> samplers);
    void update_sampler(u32 bind, const Sampler& sampler);
    void update_storage_buffers(u32 bind, std::span<const StorageBuffer> buffers);
    void update_storage_buffer(u32 bind, const StorageBuffer& buffer);
};

class Shader {
    ManagedVkResource<VkShaderModule, vkDestroyShaderModule> module;

  public:
    [[nodiscard]] VkShaderModule get_underlying(PassKey<StructMaker> /*unused*/) const {
        return module.get_underlying();
    }
    template <typename T>
    [[nodiscard]] VkShaderModule get_underlying(PassKey<GraphicsPipeline<T>> /*unused*/) const {
        return module.get_underlying();
    }

    Shader() = delete;

    Shader(const std::filesystem::path& filepath);
};

export class ComputeShader : public Shader {
  public:
    ComputeShader(const std::filesystem::path& filepath)
        : Shader(filepath) {
        assert(
            std::string_view(filepath.c_str()).ends_with(".comp.spv") &&
            "Compute shaders should have the .comp.spv file extension. You either passed the wrong shader or your "
            "shader does not follow naming convention"
        );
    }
};
export class VertexShader : public Shader {
  public:
    VertexShader(const std::filesystem::path& filepath)
        : Shader(filepath) {
        assert(
            std::string_view(filepath.c_str()).ends_with(".vert.spv") &&
            "Vertex shaders should have the .vert.spv file extension. You either passed the wrong shader or your "
            "shader does not follow naming convention"
        );
    }
};
export class FragmentShader : public Shader {
  public:
    FragmentShader(const std::filesystem::path& filepath)
        : Shader(filepath) {
        assert(
            std::string_view(filepath.c_str()).ends_with(".frag.spv") &&
            "Fragment shaders should have the .frag.spv file extension. You either passed the wrong shader or your "
            "shader does not follow naming convention"
        );
    }
};

export class PipelineLayout {
    VkPipelineLayout layout = VK_NULL_HANDLE;

  public:
    [[nodiscard]] VkPipelineLayout get_underlying(PassKey<CommandBuffer, StructMaker> /*unused*/) const {
        return layout;
    }
    template <typename T>
    [[nodiscard]] VkPipelineLayout get_underlying(PassKey<GraphicsPipeline<T>> /*unused*/) const {
        return layout;
    }
    PipelineLayout() = default;

    PipelineLayout(
        std::initializer_list<DescriptorSet> descriptor_sets,
        const std::optional<std::vector<VkPushConstantRange>>& push_constants
    );
    PipelineLayout(DescriptorSet descriptor_set, const std::optional<std::vector<VkPushConstantRange>>& push_constants);
};

// This struct does not need to be kept alive after being used to create a pipeline, and thus can be reset for use in another pipeline's initialization.
export class SpecializationInfo {
    VkSpecializationInfo info{};
    std::vector<VkSpecializationMapEntry> entries;
    size_t data_buffer_max_capacity_in_bytes{};
    std::vector<std::byte> data;
    bool finalized{};
    // Differs from data_buffer_max_capacity_in_bytes in that this is the actually used memory, while data_buffer_max_capacity_in_bytes includes potentially unused memory
    // and always reflects the actual amount of bytes available in the buffer.
    size_t data_size_in_bytes{};

  public:
    SpecializationInfo(size_t total_data_size = 128);

    template <typename T>
    SpecializationInfo& add_entry(u32 constant_ID, T constant_value) {
        static_assert(
            std::is_same_v<T, i32> || std::is_same_v<T, u32> || std::is_same_v<T, f32> || std::is_same_v<T, f64> ||
                std::is_same_v<T, VkBool32>,
            "Specialization constants can only be of the following types: i32, u32, f32, f64, VkBool32"
        );
        static_assert(sizeof(T) == 4 || sizeof(T) == 8);

        if (data_size_in_bytes + sizeof(T) > data_buffer_max_capacity_in_bytes) {
            assert(false && "Exceeded limit of data buffer while adding entry to specialization info");
            std::println(stderr, "{}", "Exceeded limit of data buffer while adding entry to specialization info");
            abort();
        }
        assert(
            std::none_of(
                entries.begin(),
                entries.end(),
                [&](const VkSpecializationMapEntry& e) { return e.constantID == constant_ID; }
            ) &&
            "Duplicate specialization constant ID: can't set the same constant twice"
        );

        finalized = false;

        const u32 offset = data_size_in_bytes;
        entries.emplace_back(constant_ID, offset, sizeof(T));
        memcpy(data.data() + offset, &constant_value, sizeof(T));
        data_size_in_bytes += sizeof(T);
        return *this;
    }

    SpecializationInfo& reset();

    // DO NOT cache the output of this function. If you ever reset this instance of SpecializationInfo or let it be destroyed,
    // the returned value of this function becomes invalid. Just pass it to a pipeline creation function and let it go.
    const VkSpecializationInfo* get_vk_specialization_info();
};

export class ComputePipeline {
    ManagedVkResource<VkPipeline, vkDestroyPipeline> pipeline;
    PipelineLayout layout;

  public:
    [[nodiscard]] VkPipeline get_underlying(PassKey<CommandBuffer> /*unused*/) const {
        return pipeline.get_underlying();
    }
    [[nodiscard]] const PipelineLayout& get_layout() const { return layout; }

    ComputePipeline(
        const ComputeShader& shader_module, PipelineLayout pipeline_layout,
        SpecializationInfo* /* _Nullable */ specialization_info = nullptr
    );
};

export template <typename T>
class GraphicsPipeline {
    ManagedVkResource<VkPipeline, vkDestroyPipeline> pipeline;
    PipelineLayout layout;

  public:
    [[nodiscard]] VkPipeline get_underlying(PassKey<CommandBuffer> /*unused*/) const {
        return pipeline.get_underlying();
    }
    [[nodiscard]] const PipelineLayout& get_layout() const { return layout; }

    GraphicsPipeline(
        const VertexShader& vert_shader, const FragmentShader& frag_shader, PipelineLayout pipeline_layout,
        const VertexBuffer<T>& vertex_buffer, VkFormat color_attachment_format, VkFormat depth_attachment_format,
        MSAALevel msaa_level, SpecializationInfo* /* _Nullable */ vert_specialization_info = nullptr,
        SpecializationInfo* /* _Nullable */ frag_specialization_info = nullptr,
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    )
        : layout(pipeline_layout) {
        assert(
            topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST || topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
            topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        );
        VkPipelineShaderStageCreateInfo shader_stage_infos[2];

        const auto make_pipeline_shader_stage_info = [](const Shader& shader_module,
                                                        VkShaderStageFlagBits stage,
                                                        SpecializationInfo* /* _Nullable */ specialization_info) {
            return VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags{},
                .stage = stage,
                .module = shader_module.get_underlying(PassKey<GraphicsPipeline<T>>{}),
                .pName = "main",
                .pSpecializationInfo =
                    specialization_info != nullptr ? specialization_info->get_vk_specialization_info() : nullptr,
            };
        };

        shader_stage_infos[0] =
            make_pipeline_shader_stage_info(vert_shader, VK_SHADER_STAGE_VERTEX_BIT, vert_specialization_info);
        shader_stage_infos[1] =
            make_pipeline_shader_stage_info(frag_shader, VK_SHADER_STAGE_FRAGMENT_BIT, frag_specialization_info);

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
            .vertexAttributeDescriptionCount = static_cast<u32>(vertex_attribute_descriptions.size()),
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
            .sampleShadingEnable = VK_FALSE, // true = ssaa, false = msaa
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
            .depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL, // reverse Z. "OR_EQUAL" for when we do z prepass later

            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
            .front = {},
            .back = {},
            .minDepthBounds = 0.0f,
            .maxDepthBounds = 1.0f
        };

        const VkPipelineColorBlendAttachmentState color_blend_attachment_alpha{
            .blendEnable = VK_TRUE,

            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = VK_BLEND_OP_ADD,

            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,

            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT
        };

        VkPipelineColorBlendStateCreateInfo color_blend{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext = nullptr,
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
            .layout = this->layout.get_underlying(PassKey<GraphicsPipeline<T>>{}),
            .renderPass = VK_NULL_HANDLE,
            .subpass{},
            .basePipelineHandle{},
            .basePipelineIndex{},
        };

        VkResult result = (vkCreateGraphicsPipelines(
            g_vk->device,
            VK_NULL_HANDLE,
            1,
            &graphics_pipeline_create_info,
            nullptr,
            this->pipeline.get_underlying_ptr()
        ));
        if (result != VK_SUCCESS) {
            std::println(stderr, "Vulkan error with code: {}", static_cast<int>(result));
            abort();
        }
    }
};

export struct BarrierInfo {
    const Image& img;
    VkImageLayout old_layout_or_undefined_to_discard_current_data;
    VkImageLayout new_layout;
    ImageAspects aspects;
    VkPipelineStageFlags2 src_stage_mask;
    VkAccessFlags2 src_access_mask;
    VkPipelineStageFlags2 dst_stage_mask;
    VkAccessFlags2 dst_access_mask;
    u32 base_mip_level;
    u32 mip_level_count;

    BarrierInfo(
        const Image& img, VkImageLayout old_layout_or_undefined_to_discard_current_data, VkImageLayout new_layout,
        ImageAspects aspects, VkPipelineStageFlags2 src_stage_mask = VK_PIPELINE_STAGE_2_NONE,
        VkAccessFlags2 src_access_mask = VK_ACCESS_2_NONE,
        VkPipelineStageFlags2 dst_stage_mask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VkAccessFlags2 dst_access_mask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        u32 base_mip_level = 0, u32 mip_level_count = VK_REMAINING_MIP_LEVELS
    )
        : img(img),
          old_layout_or_undefined_to_discard_current_data(old_layout_or_undefined_to_discard_current_data),
          new_layout(new_layout),
          aspects(aspects),
          src_stage_mask(src_stage_mask),
          src_access_mask(src_access_mask),
          dst_stage_mask(dst_stage_mask),
          dst_access_mask(dst_access_mask),
          base_mip_level(base_mip_level),
          mip_level_count(mip_level_count) {}

    BarrierInfo() = delete;
    BarrierInfo(const BarrierInfo&) = default;
    BarrierInfo(BarrierInfo&&) noexcept = default;
    BarrierInfo& operator=(const BarrierInfo&) = delete;
    BarrierInfo& operator=(BarrierInfo&&) = delete;
    ~BarrierInfo() = default;
};

export class CommandBuffer {
    VkCommandBuffer buffer;

    static VkImageMemoryBarrier2 image_memory_barrier2(const BarrierInfo& barrier_info) {
        return VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = barrier_info.src_stage_mask,
            .srcAccessMask = barrier_info.src_access_mask,
            .dstStageMask = barrier_info.dst_stage_mask,
            .dstAccessMask = barrier_info.dst_access_mask,
            .oldLayout = barrier_info.old_layout_or_undefined_to_discard_current_data,
            .newLayout = barrier_info.new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = barrier_info.img.get_underlying({}),
            .subresourceRange = {
                .aspectMask = static_cast<VkImageAspectFlags>(barrier_info.aspects),
                .baseMipLevel = barrier_info.base_mip_level,
                .levelCount = barrier_info.mip_level_count,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
    }

    static VkDependencyInfo dependency_info(std::span<VkImageMemoryBarrier2> image_barriers) {
        return VkDependencyInfo{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags{},
            .memoryBarrierCount{},
            .pMemoryBarriers{},
            .bufferMemoryBarrierCount{},
            .pBufferMemoryBarriers{},
            .imageMemoryBarrierCount = static_cast<u32>(image_barriers.size()),
            .pImageMemoryBarriers = image_barriers.data(),
        };
    }

    void bind_descriptor_sets_inner(
        VkPipelineBindPoint bind_point, std::initializer_list<DescriptorSet> sets, const PipelineLayout& layout
    ) const {
        const i32 max_sets = 4;
        if (sets.size() > max_sets) {
            assert(false && "Too many desc sets");
            abort();
        };
        std::array<VkDescriptorSet, max_sets> vk_sets{};
        size_t i = 0;
        for (const auto& set : sets) {
            vk_sets.at(i++) = set.get_underlying({});
        }

        vkCmdBindDescriptorSets(
            this->buffer, bind_point, layout.get_underlying({}), 0, sets.size(), vk_sets.data(), 0, nullptr
        );
    }

  public:
    [[nodiscard]] VkCommandBuffer get_underlying(PassKey<StructMaker> /* key */) const { return buffer; }
    CommandBuffer(const CommandPool& pool);
    CommandBuffer(VkCommandBuffer vk_buffer)
        : buffer(vk_buffer) {}
    static void
    make_command_buffers(std::vector<CommandBuffer>& buffers, const CommandPool& pool, i32 how_many_buffers_to_append);
    void draw_imgui(const ImageView& target_image_view, VkExtent2D draw_extent) const;
    void restart(bool one_time_submit) const;
    void submit(
        std::optional<std::reference_wrapper<const GpuSemaphore>> wait_sema,
        std::optional<VkPipelineStageFlagBits2>
            wait_stage_mask, // Commands in the buffer that use these stages will not run until wait_sema is signaled
        std::optional<std::reference_wrapper<const GpuSemaphore>>
            signal_sema, // Signals after the specified pipeline stages complete
        std::optional<VkPipelineStageFlagBits2>
            signal_stage_mask, // Sets the pipeline stages that trigger the signal operation
        const GpuFence& signal_fence
    ) const;
    void submit(const GpuFence& signal_fence) const;
    void copy_buffer(const VulkanBuffer& src, const VulkanBuffer& dst, std::span<const VkBufferCopy2> ranges) const;
    void copy_buffer(const VulkanBuffer& src, const VulkanBuffer& dst, const VkBufferCopy2& range) const;
    void copy_entire_buffer(const VulkanBuffer& src, const VulkanBuffer& dst) const;
    void copy_buffer_to_image(
        const VulkanBuffer& buffer, const Image& image, VkImageLayout image_layout, u32 region_count,
        const VkBufferImageCopy2* regions
    ) const;
    void copy_buffer_to_image(
        const VulkanBuffer& buffer, const Image& image, u64 buffer_offset, u32 mip_level, u32 base_layer,
        u32 layer_count, ImageAspects aspects, const VkOffset3D& img_offset = VkOffset3D{}
    ) const;
    void copy_buffer_to_image(const VulkanBuffer& buffer, const Image& image, const VkBufferImageCopy2& region) const;
    void copy_buffer_to_image(
        const VulkanBuffer& buffer, const Image& image, std::span<const VkBufferImageCopy2> regions
    ) const;
    void copy_image_to_buffer(
        const Image& image, const VulkanBuffer& buffer, u64 buffer_offset, u32 mip_level, u32 base_layer,
        u32 layer_count, ImageAspects aspects, const VkOffset3D& img_offset
    ) const;
    void copy_image_to_buffer(const Image& image, const VulkanBuffer& buffer, const VkBufferImageCopy2& region) const;
    void copy_image_to_buffer(
        const Image& image, const VulkanBuffer& buffer, std::span<const VkBufferImageCopy2> regions
    ) const;
    void copy_image(
        const Image& src, const Image& dst, ImageAspects src_aspects, ImageAspects dst_aspects, u32 src_mip_level,
        u32 dst_mip_level
    ) const;

    template <typename T>
    void draw_indexed(
        const ImageView& color_attachment, const ImageView& depth_attachment, const ImageView& resolve_image_view,
        VkExtent2D draw_extent, const VertexBuffer<T>& vertex_buffer, const IndexBuffer& index_buffer,
        const std::vector<VkDrawIndexedIndirectCommand>& commands
    ) const {
        VkRenderingAttachmentInfo color_attachment_info{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = color_attachment.get_underlying({}),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT,
            .resolveImageView = resolve_image_view.get_underlying({}),
            .resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue{},
        };

        VkRenderingAttachmentInfo depth_attachment_info{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = depth_attachment.get_underlying({}),
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .resolveMode{},
            .resolveImageView{},
            .resolveImageLayout{},
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue{.depthStencil{.depth = 0.0f, .stencil = 0}},
        };

        const auto make_rendering_info = [](VkExtent2D renderExtent,
                                            VkRenderingAttachmentInfo* colorAttachment,
                                            VkRenderingAttachmentInfo* depthAttachment) {
            return VkRenderingInfo{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags{},
                .renderArea = {.offset = {.x = 0, .y = 0}, .extent = renderExtent},
                .layerCount = 1,
                .viewMask{},
                .colorAttachmentCount = 1,
                .pColorAttachments = colorAttachment,
                .pDepthAttachment = depthAttachment,
                .pStencilAttachment = nullptr,
            };
        };

        VkRenderingInfo rendering_info =
            make_rendering_info(draw_extent, &color_attachment_info, &depth_attachment_info);

        vkCmdBeginRendering(buffer, &rendering_info);

        const VkDeviceSize offsets = 0;
        {
            VkBuffer vb = vertex_buffer.get_underlying({});
            vkCmdBindVertexBuffers(buffer, 0, 1, &vb, &offsets);
        }
        vkCmdBindIndexBuffer(buffer, index_buffer.get_underlying({}), 0, IndexBuffer::index_type);

        const VkViewport viewport = {
            .x = 0,
            .y = 0,
            .width = static_cast<f32>(draw_extent.width),
            .height = static_cast<f32>(draw_extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        vkCmdSetViewport(buffer, 0, 1, &viewport);

        const VkRect2D scissor{
            .offset{.x = 0, .y = 0},
            .extent{draw_extent},
        };
        vkCmdSetScissor(buffer, 0, 1, &scissor);

        for (const auto& command : commands)
            vkCmdDrawIndexed(
                buffer,
                command.indexCount,
                command.instanceCount,
                command.firstIndex,
                command.vertexOffset,
                command.firstInstance
            );

        vkCmdEndRendering(buffer);
    }

    template <typename... BarrierInfo_T>
    void barrier(const BarrierInfo_T&... barrier_infos) const {
        static_assert(
            (std::is_same_v<std::remove_cvref_t<BarrierInfo_T>, BarrierInfo> && ...),
            "Arguments to CommandBuffer::barrier must be BarrierInfo types."
        );
        std::array<VkImageMemoryBarrier2, sizeof...(BarrierInfo_T)> image_barriers;

        i32 i = 0;
        const auto add_image_barrier = [&](const BarrierInfo& barrier_info) {
            image_barriers[i++] = image_memory_barrier2(barrier_info);
        };

        (add_image_barrier(barrier_infos), ...);

        VkDependencyInfo dep_info = dependency_info(image_barriers);
        vkCmdPipelineBarrier2(this->buffer, &dep_info);
    }

    void barrier_span(std::span<const BarrierInfo> barrier_infos) const {
        std::vector<VkImageMemoryBarrier2>
            image_barriers; // todo optimize out the heap allocation with a custom allocator
        image_barriers.reserve(barrier_infos.size());
        for (const auto& info : barrier_infos) {
            image_barriers.push_back(image_memory_barrier2(info));
        }
        const VkDependencyInfo dep_info = dependency_info(image_barriers);
        vkCmdPipelineBarrier2(this->buffer, &dep_info);
    }

    void blit(
        const Image& src, const Image& dst, ivec2 src_top_left, ivec2 src_bottom_right, ivec2 dst_top_left,
        ivec2 dst_bottom_right, ImageAspects aspects
    ) const;

    void blit_entire_images(const Image& source, const Image& destination, ImageAspects aspects) const;
    void bind_pipeline(const ComputePipeline& pipeline) const;

    template <typename T>
    void bind_pipeline(const GraphicsPipeline<T>& pipeline) const {
        vkCmdBindPipeline(this->buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.get_underlying({}));
    }

    void bind_descriptor_sets(const ComputePipeline& pipeline, std::initializer_list<DescriptorSet> sets) const;
    void bind_descriptor_set(const ComputePipeline& pipeline, DescriptorSet set) const;
    template <typename VertexType>
    void bind_descriptor_sets(
        const GraphicsPipeline<VertexType>& pipeline, std::initializer_list<DescriptorSet> sets
    ) const {
        bind_descriptor_sets_inner(VK_PIPELINE_BIND_POINT_GRAPHICS, sets, pipeline.get_layout());
    }
    template <typename VertexType>
    void bind_descriptor_set(const GraphicsPipeline<VertexType>& pipeline, DescriptorSet set) const {
        bind_descriptor_sets(pipeline, {set});
    }

    template <typename T>
    void update_push_constants(const ComputePipeline& pipeline, const PushConstant<T>& push_constant) const {
        assert(
            push_constant.shader_stages & VK_SHADER_STAGE_COMPUTE_BIT &&
            "Updated push constants for a compute pipeline with a push constant that cannot be accessed by compute "
            "shaders."
        );
        vkCmdPushConstants(
            this->buffer,
            pipeline.get_layout().get_underlying({}),
            push_constant.shader_stages,
            push_constant.offset,
            push_constant.size,
            &push_constant.data
        );
    }

    template <typename T, typename U>
    void update_push_constants(const GraphicsPipeline<T>& pipeline, const PushConstant<U>& push_constant) const {
        assert(
            push_constant.shader_stages & (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT) &&
            "Updated push constants for a graphics pipeline with a push constant that cannot be accessed by vertex nor "
            "fragment shaders."
        );
        vkCmdPushConstants(
            this->buffer,
            pipeline.get_layout().get_underlying({}),
            push_constant.shader_stages,
            push_constant.offset,
            push_constant.size,
            &push_constant.data
        );
    }

    void dispatch(u32 x, u32 y, u32 z) const;
    void end() const;
};

// This is what you're meant to use instead of Descriptor Set Layouts and Descriptor Set Layout Bindings. You can
// keep using build as many times as you want just like you would with a Layout.
export class DescriptorSetBuilder {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    VkDescriptorSetLayout layout{};
    bool finalized = false;

  public:
    DescriptorSetBuilder& bind(
        u32 binding, VkDescriptorType type, u32 count = 1,
        VkShaderStageFlags accessible_stages_flags = VK_SHADER_STAGE_ALL
    );

    DescriptorSet build();

    // Don't reset unless you know you won't be making another descriptor set with the same layout again. It's slower otherwise.
    // Objects passed into the creation of other objects must not be destroyed while the new object is in use, so we cannot destroy the layout here.
    // Since we use all descriptor sets until the end of the program, there is no reason to want that, either.
    DescriptorSetBuilder& reset();
};
