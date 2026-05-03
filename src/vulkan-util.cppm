module;

#include <vulkan/vulkan_core.h>
#include <bits/stdint-uintn.h>

#if !USE_IMPORT_STD
#include <bits/std_function.h>
#include <vector>
#endif
export module vulkanUtil;
#if USE_IMPORT_STD
import std;
#endif

import vulkanEngine;

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
 
    uint64_t used_staging_buffer_bytes;
    struct FreeStagingRegion{
        uint64_t offset;
        uint64_t size;
    };
    FreeStagingRegion get_free_staging_region(uint64_t desired_size, bool require_desired_size);
    void stage_upload(FreeStagingRegion free_region, const void *src, const VulkanBuffer &dst, uint64_t dst_offset);
    void stage_upload(FreeStagingRegion free_region, const void *src, const Image &dst, const VkBufferImageCopy2 &copy);


public:
    HostToDeviceUploader(VulkanEngine *const vk, const CommandPool cmd_pool, uint64_t staging_buffer_size);

    // The next three functions and their comments describe the core of how this class is used.

    // Usually won't begin the upload, will just write to the staging buffer. You must later call begin_uploads and finish_uploads.
    // If you have multiple uploads for a single destination buffer, there will be less driver overhead if you stage them
    // all before calling begin_uploads.
    // The uploads made with this function are put into the "Queued" category.
    void queue_upload(const void *src, const VulkanBuffer &dst, uint64_t byte_count, uint64_t dst_offset = 0);
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
    void queue_begin_finish_uploads(const void *src, const VulkanBuffer &dst, uint64_t byte_count, uint64_t dst_offset = 0);

    bool queued_uploads_exist();
    bool in_progress_uploads_exist();
    bool queued_or_in_progress_uploads_exist();
    size_t queued_uploads_count();
    size_t in_progress_uploads_count();

    void queue_upload(const void *src, const Image &dst, uint64_t byte_count, uint32_t mip_level, uint32_t base_layer, uint32_t layer_count, ImageAspects aspects, const VkOffset3D &img_offset = {});
};

