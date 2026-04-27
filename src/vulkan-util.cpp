module;

#include <bits/stdint-uintn.h>
#include <cassert>
#include <vulkan/vulkan_core.h>

#if !USE_IMPORT_STD
#include <cstring>
#include <vector>
#include <span>
#include <bits/std_function.h>
#endif

module vulkanUtil;

#if USE_IMPORT_STD
import std;
#endif
import vulkanEngine;

ImmediateSubmitter::ImmediateSubmitter(VulkanEngine &vk, const CommandPool &cmd_pool)
:submit_fence(vk, false), cmd_buffer(vk, cmd_pool){}

void ImmediateSubmitter::submit(VulkanEngine &vk, const std::function<void()> &function){
    cmd_buffer.restart(true);

    function();

    cmd_buffer.end();
    cmd_buffer.submit(vk, submit_fence);
    submit_fence.wait(vk);
}

HostToDeviceUploader::InProgressUpload::InProgressUpload(VulkanEngine& vk, const CommandPool &cmd_pool)
:cmd_buffer(vk, cmd_pool), fence(vk, false)
{
    copy_info.reserve(64);
}

HostToDeviceUploader::HostToDeviceUploader(
    VulkanEngine *const vk, const CommandPool cmd_pool,
    uint32_t staging_buffer_size)
:vk(vk),
staging_buffer(*vk, staging_buffer_size),
cmd_pool(cmd_pool),
used_staging_buffer_bytes(0)
{
    in_progress_uploads.reserve(128);
    queued_uploads.reserve(128);
}

HostToDeviceUploader::FreeStagingRegion HostToDeviceUploader::get_free_staging_region(uint32_t desired_size){
    uint32_t remaining_size = staging_buffer.capacity_in_bytes - used_staging_buffer_bytes;
    assert((int32_t)remaining_size > -1);
    if (remaining_size == 0) {
        begin_and_finish_uploads();
        remaining_size = staging_buffer.capacity_in_bytes;
    }

    FreeStagingRegion region{
        .offset = static_cast<uint32_t>(used_staging_buffer_bytes),
        .size = std::min(desired_size, remaining_size),
    };
    used_staging_buffer_bytes += region.size;

    return region;
}
void HostToDeviceUploader::stage_upload(FreeStagingRegion free_region, const void *src, const VulkanBuffer &dst, uint32_t dst_offset){
    std::memcpy((char*)staging_buffer.get_mapped_data() + free_region.offset,
                (char*)src,
                free_region.size);

    VkBufferCopy2 copy{
        .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
        .pNext = nullptr,
        .srcOffset = free_region.offset,
        .dstOffset = dst_offset,
        .size = free_region.size
    };
    for (auto &queued_upload : queued_uploads){
        if (queued_upload.dst.buffer == dst.buffer){
            queued_upload.copy_info.push_back(copy);
            return;
        }
    }
    queued_uploads.emplace_back(dst, copy);
}

void HostToDeviceUploader::queue_upload(const void *src, const VulkanBuffer &dst, uint32_t byte_count, uint32_t dst_offset){
    uint32_t remaining_size = byte_count;
    while (remaining_size > 0){
        FreeStagingRegion free_region = get_free_staging_region(remaining_size);

        uint32_t bytes_already_written = byte_count - remaining_size;
        stage_upload(free_region, (const char*)src + bytes_already_written, dst, dst_offset + bytes_already_written );

        remaining_size -= free_region.size;
    }
}

void HostToDeviceUploader::begin_uploads(){
    for (auto &queued_upload : queued_uploads){
        in_progress_uploads.emplace_back(*vk, cmd_pool);
        InProgressUpload &in_progress_upload = in_progress_uploads.back();
        const CommandBuffer &cmd_buffer = in_progress_upload.cmd_buffer;

        cmd_buffer.copy_buffer(staging_buffer, queued_upload.dst, queued_upload.copy_info);
        cmd_buffer.submit(*vk, in_progress_upload.fence);

        in_progress_upload.copy_info.swap(queued_upload.copy_info);
    }
    queued_uploads.clear();
}

void HostToDeviceUploader::finish_in_progress_uploads(){
    for (auto &in_progress_upload : in_progress_uploads){
        in_progress_upload.fence.wait(*vk);
    }
    in_progress_uploads.clear();

    if (queued_uploads.empty()) { used_staging_buffer_bytes = 0; }
}

void HostToDeviceUploader::begin_and_finish_uploads(){
    begin_uploads();
    finish_in_progress_uploads();
}
