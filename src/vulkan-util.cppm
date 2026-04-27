module;

#include <vulkan/vulkan_core.h>
#include <bits/stdint-uintn.h>

#if !USE_IMPORT_STD
#include <bits/std_function.h>
#include <vector>
#include <span>
#endif
export module vulkanUtil;
#if USE_IMPORT_STD
import std;
#endif

import vulkanEngine;

export class ImmediateSubmitter{
    GpuFence submit_fence;
    CommandBuffer cmd_buffer;
public:
    ImmediateSubmitter(VulkanEngine &vk, const CommandPool &cmd_pool);
    void submit(VulkanEngine &vk, const std::function<void()> &function);
};

export class HostToDeviceUploader{
    VulkanEngine *const vk;
    StagingBuffer staging_buffer;
    CommandPool cmd_pool;

    struct InProgressUpload{
        CommandBuffer cmd_buffer;
        GpuFence fence;
        std::vector<VkBufferCopy2> copy_info;
        InProgressUpload(VulkanEngine &vk, const CommandPool &cmd_pool);
    };
    struct QueuedUpload{
        VulkanBuffer dst;
        std::vector<VkBufferCopy2> copy_info;//don't forget to std::swap this vector when it goes into an upload slot
        QueuedUpload(const VulkanBuffer &dst, const VkBufferCopy2 &copy):dst(dst){
            copy_info.reserve(128);
            copy_info.push_back(copy);
        }
    };
    std::vector<InProgressUpload> in_progress_uploads; // Each of these can support a running upload
    std::vector<QueuedUpload> queued_uploads;
 
    uint32_t used_staging_buffer_bytes;
    struct FreeStagingRegion{
        uint32_t offset;
        uint32_t size;
    };
    FreeStagingRegion get_free_staging_region(uint32_t desired_size);
    void stage_upload(FreeStagingRegion free_region, const void *src, const VulkanBuffer &dst, uint32_t dst_offset);


public:
    HostToDeviceUploader(VulkanEngine *const vk, const CommandPool cmd_pool, uint32_t staging_buffer_size);

    // The next three functions and their comments describe the core of how this class is used.

    // Usually won't begin the upload, will just write to the staging buffer. You must later call begin_uploads and finish_uploads.
    // If you have multiple uploads for a single destination buffer, there will be less driver overhead if you stage them
    // all before calling begin_uploads.
    // The uploads made with this function are put into the "Queued" category.
    void queue_upload(const void *src, const VulkanBuffer &dst, uint32_t byte_count, uint32_t dst_offset = 0);
    // Non-blocking. You must call finish_uploads before doing anything that assumes the data has been uploaded to the GPU.
    // Uploads affected by this function are put into the "In Progress" category. They are REMOVED from the "Queued" category.
    // If the internal staging buffer becomes full, queued uploads might begin (and finish) without you calling
    // this function. In general, you should not assume that you have control over when the data
    // is uploaded.
    void begin_uploads();
    // Blocks until all data has made it to GPU memory.
    // Finished uploads are removed from all categories.
    void finish_in_progress_uploads();
    void begin_and_finish_uploads();


    bool queued_uploads_exist();
    bool in_progress_uploads_exist();
    bool queued_or_in_progress_uploads_exist();
};

