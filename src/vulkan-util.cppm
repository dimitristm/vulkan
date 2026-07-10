module;

#include <vulkan/vulkan_core.h>
#include <bits/stdint-uintn.h>

#if !VK_PROJ_USE_IMPORT_STD
#include <bits/std_function.h>
#include <vector>
#include <span>
#endif
export module vulkanUtil;
#if VK_PROJ_USE_IMPORT_STD
import std;
#endif

import vulkanEngine;
import types;


template <typename T>
concept SpanCompatible = requires(const T &t) {
    std::span{t};
};

export class ImmediateSubmitter{
    GpuFence submit_fence;
    CommandBuffer internal_cmd_buffer;
public:
    ImmediateSubmitter(VulkanEngine &vk, const CommandPool &cmd_pool);
    void submit(VulkanEngine &vk, const std::function<void()> &function);
    const CommandBuffer &cmd_buffer(){ return internal_cmd_buffer; }
};

export class HostToDeviceUploader{
    VulkanEngine *const vk;
    StagingBuffer staging_buffer;
    CommandPool cmd_pool;

    std::vector<GpuFence> fence_pool;
    std::vector<CommandBuffer> cmd_buffer_pool;

    struct InProgressUpload{
        CommandBuffer cmd_buffer;
        GpuFence fence;
        InProgressUpload(CommandBuffer &command_buffer, GpuFence &gpu_fence)
        :cmd_buffer(command_buffer), fence(gpu_fence){}
    };

    struct QueuedBufferUpload{
        VulkanBuffer dst;
        std::vector<VkBufferCopy2> copy_info;// todo performance: get rid of the allocation with a memory pool
        QueuedBufferUpload(const VulkanBuffer &dst, const VkBufferCopy2 &copy):dst(dst){
            copy_info.reserve(128);
            copy_info.push_back(copy);
        }
    };
    struct QueuedImageUpload{
        Image dst;
        std::vector<VkBufferImageCopy2> copy_info;
        QueuedImageUpload(const Image dst, const VkBufferImageCopy2 &copy):dst(dst){
            copy_info.reserve(128);
            copy_info.push_back(copy);
        }
    };

    std::vector<InProgressUpload> in_progress_uploads;
    std::vector<QueuedBufferUpload> queued_buffer_uploads;
    std::vector<QueuedImageUpload> queued_image_uploads;
 
    u64 used_staging_buffer_bytes;
    struct FreeStagingRegion{
        u64 offset;
        u64 size;
    };
    FreeStagingRegion get_free_staging_region(u64 desired_size, bool require_desired_size);
    void stage_upload(FreeStagingRegion free_region, const void *src, const VulkanBuffer &dst, u64 dst_offset);
    void stage_upload(FreeStagingRegion free_region, const void *src, const Image &dst, const VkBufferImageCopy2 &copy);

    const VulkanBuffer *active_upload_dst = nullptr;
    u64 active_upload_next_dst_offset = 0;

public:
    HostToDeviceUploader(VulkanEngine *const vk, const CommandPool cmd_pool, u64 staging_buffer_size);

    // The next three functions and their comments describe the core of how this class is used.

    // Usually won't begin the upload, will just write to the staging buffer. You must later call begin_uploads and finish_uploads.
    // If you have multiple uploads for a single destination buffer, there will be less driver overhead if you stage them
    // all before calling begin_uploads.
    // The uploads made with this function are put into the "Queued" category.
    void queue_upload(const void *src, const VulkanBuffer &dst, u64 byte_count, u64 dst_offset = 0);
    // Non-blocking. You must call finish_uploads before doing anything that assumes the data has been uploaded to the GPU.
    // Uploads affected by this function are put into the "In Progress" category. They are REMOVED from the "Queued" category.
    // Uploads may be coalesced into one, so if you have X "Queued" uploads you may get Y "In Progress" uploads where Y<X.
    // If the internal staging buffer becomes full, queued uploads might begin (and finish) without you calling
    // this function. In general, you should not assume that you have control over when the data
    // is uploaded and in what order, unless you use a finish_uploads function between the uploads.
    void begin_uploads();
    // Blocks until all data has made it to GPU memory.
    // Finished uploads are removed from all categories.
    void finish_in_progress_uploads();
    void begin_and_finish_uploads();
    void queue_begin_finish_uploads(const void *src, const VulkanBuffer &dst, u64 byte_count, u64 dst_offset = 0);

    void queue_upload(const void *src, const Image &dst, u64 byte_count, u32 mip_level, u32 base_layer, u32 layer_count, ImageAspects aspects, const VkOffset3D &img_offset = {});

    // Continues from the last upload you queued. Essentially appends to the buffer you were writing to.
    // Intended for small writes. Passing an src larger than the internal staging buffer's size crashes.
    void add_to_last_upload(const void *src, u64 byte_count);
    // An empty queue_upload. Helpful for use with add_to_last_upload.
    void start_queue_upload(const VulkanBuffer &dst, u64 dst_offset = 0);

    bool queued_uploads_exist();
    bool in_progress_uploads_exist();
    bool queued_or_in_progress_uploads_exist();
    size_t queued_uploads_count();
    size_t in_progress_uploads_count();

    // Below are just helpers, no new functionality
    template <typename T> requires (!SpanCompatible<T> && !std::is_pointer_v<std::remove_reference_t<T>>)
    void queue_upload(const T &src, const VulkanBuffer &dst, u64 dst_offset = 0) {
        queue_upload(&src, dst, sizeof(T), dst_offset);
    }

    template <SpanCompatible T>
    void queue_upload(const T &src, const VulkanBuffer &dst, u64 dst_offset = 0) {
        auto s = std::span{src};
        queue_upload(s.data(), dst, s.size_bytes(), dst_offset);
    }

    template <typename T>
        requires (!SpanCompatible<T>)
    void add_to_last_upload(const T &src) {
        add_to_last_upload(&src, sizeof(T));
    }

    template <SpanCompatible T>
    void add_to_last_upload(const T &src) {
        auto s = std::span{src};
        add_to_last_upload(s.data(), s.size_bytes());
    }
};

export struct Texture : public Image{
    Texture(VulkanEngine &vk, u32 width, u32 height, VkFormat format, u32 mip_level_count)
    :Image(vk,
           {.width=width,.height=height},
           format,
           VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
           MSAALevel::OFF,
           mip_level_count,
           1){}
    ImageView make_view(VulkanEngine &vk)
    {
        return  {vk, *this, ImageAspects::COLOR, 0, VK_REMAINING_MIP_LEVELS};
    }
};

static inline constexpr VkFormat drawing_format = VK_FORMAT_R16G16B16A16_SFLOAT;
export struct ResolveImage{
    static constexpr VkFormat format = drawing_format;
    Image img;
    ImageView view;
    ResolveImage(VulkanEngine &vk, const ivec2 &size)
    :img(vk,
           {.width=static_cast<u32>(size.x), .height=static_cast<u32>(size.y)},
           format,
           VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
           | VK_IMAGE_USAGE_STORAGE_BIT
           | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
           MSAALevel::OFF,
           1, 1),
    view(vk, img, ImageAspects::COLOR, 0, 1)
    {}
    void resize(VulkanEngine &vk, const ivec2 &new_window_size){
        change(vk, new_window_size);
    }
    void change(VulkanEngine &vk, const ivec2 &new_window_size){
        img.erase_self(vk);
        view.erase_self(vk);
        *this = ResolveImage(vk, new_window_size);
    }
};

export struct DrawImage{
    static constexpr VkFormat format = drawing_format;
    Image img;
    ImageView view;
    ResolveImage resolve_img;
    DrawImage(VulkanEngine &vk, const ivec2 &size, MSAALevel msaa_level)
    :img(vk,
           {.width=static_cast<u32>(size.x), .height=static_cast<u32>(size.y)},
           format,
           //VK_IMAGE_USAGE_STORAGE_BIT
           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, // transient because it gets resolved immediately during dynamic rendering, and after that we don't care about its contents
           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
           msaa_level,
           1, 1),
    view(vk, img, ImageAspects::COLOR, 0, 1),
    resolve_img(vk, size)
    {}
    void resize(VulkanEngine &vk, const ivec2 &new_window_size){
        change(vk, new_window_size, img.msaa_level);
    }
    void change_msaa(VulkanEngine &vk, MSAALevel msaa_level){
        change(vk, {img.extent.width, img.extent.height}, msaa_level);
    }
    void change(VulkanEngine &vk, const ivec2 &new_window_size, MSAALevel msaa_level){
        img.erase_self(vk);
        view.erase_self(vk);
        *this = DrawImage(vk, new_window_size, msaa_level);
    }
};

export struct DepthImage{
    Image img;
    ImageView view;
    DepthImage(VulkanEngine &vk, const ivec2 &size, MSAALevel msaa_level)
    :img(vk,
           {.width=static_cast<u32>(size.x), .height=static_cast<u32>(size.y)},
           VK_FORMAT_D32_SFLOAT,
           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
           msaa_level,
           1, 1),
    view(vk, img, ImageAspects::DEPTH, 0, 1)
    {}
    void resize(VulkanEngine &vk, const ivec2 &new_window_size){
        change(vk, new_window_size, img.msaa_level);
    }
    void change_msaa(VulkanEngine &vk, MSAALevel msaa_level){
        change(vk, {img.extent.width, img.extent.height}, msaa_level);
    }
    void change(VulkanEngine &vk, const ivec2 &new_window_size, MSAALevel msaa_level){
        img.erase_self(vk);
        view.erase_self(vk);
        *this = DepthImage(vk, new_window_size, msaa_level);
    }
};
