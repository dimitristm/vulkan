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
    InProgressUpload::fence_pool.reserve(128);
    InProgressUpload::cmd_buffer_pool.reserve(128);
    for (int i = 0; i < 16; ++i){
        InProgressUpload::fence_pool.emplace_back(*vk, false);
    }
    CommandBuffer::make_command_buffers(*vk, InProgressUpload::cmd_buffer_pool, cmd_pool, 64);
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

template <typename Container>
static auto pop_back_and_return(Container& c) {
    auto value = std::move(c.back());
    c.pop_back();
    return value;
}

void HostToDeviceUploader::begin_uploads(){
    const auto add_in_progress_upload = [&]{
        auto &cmd_buffer_pool =  InProgressUpload::cmd_buffer_pool;
        auto &fence_pool = InProgressUpload::fence_pool;
        CommandBuffer cmd_buffer = cmd_buffer_pool.empty() ? CommandBuffer(*vk, cmd_pool)
                                                           : pop_back_and_return(cmd_buffer_pool);
        GpuFence fence = fence_pool.empty() ? GpuFence(*vk, false)
                                            : pop_back_and_return(fence_pool);
        in_progress_uploads.emplace_back(cmd_buffer, fence);
        return in_progress_uploads.back();
    };

    const InProgressUpload &in_progress_upload = add_in_progress_upload();
    const CommandBuffer &cmd_buffer = in_progress_upload.cmd_buffer;
    for (auto &queued_upload : queued_uploads){
        cmd_buffer.copy_buffer(staging_buffer, queued_upload.dst, queued_upload.copy_info);
    }
    cmd_buffer.submit(*vk, in_progress_upload.fence);
    queued_uploads.clear();
}

void HostToDeviceUploader::finish_in_progress_uploads(){
    for (auto &in_progress_upload : in_progress_uploads){
        in_progress_upload.fence.wait(*vk);
        InProgressUpload::fence_pool.push_back(in_progress_upload.fence);
        InProgressUpload::cmd_buffer_pool.push_back(in_progress_upload.cmd_buffer);
    }
    in_progress_uploads.clear();

    if (queued_uploads.empty()) { used_staging_buffer_bytes = 0; }
}

void HostToDeviceUploader::begin_and_finish_uploads(){
    begin_uploads();
    finish_in_progress_uploads();
}

bool HostToDeviceUploader::queued_uploads_exist(){
    return !queued_uploads.empty();
}
bool HostToDeviceUploader::in_progress_uploads_exist(){
    return !in_progress_uploads.empty();
}
bool HostToDeviceUploader::queued_or_in_progress_uploads_exist(){
    return queued_uploads_exist() || in_progress_uploads_exist();
}
size_t HostToDeviceUploader::queued_uploads_count(){
    return queued_uploads.size();
}
size_t HostToDeviceUploader::in_progress_uploads_count(){
    return in_progress_uploads.size();
}
