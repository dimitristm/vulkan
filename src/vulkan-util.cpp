module;

#include <bits/stdint-uintn.h>
#include <cassert>
#include <vulkan/vulkan_core.h>

#if !USE_IMPORT_STD
#include <cstring>
#include <vector>
#include <span>
#include <bits/std_function.h>
#include <print>
#endif

module vulkanUtil;

#if USE_IMPORT_STD
import std;
#endif
import vulkanEngine;
import types;

ImmediateSubmitter::ImmediateSubmitter(VulkanEngine &vk, const CommandPool &cmd_pool)
:submit_fence(vk, false), internal_cmd_buffer(vk, cmd_pool){}

void ImmediateSubmitter::submit(VulkanEngine &vk, const std::function<void()> &function){
    internal_cmd_buffer.restart(true);

    function();

    internal_cmd_buffer.end();
    internal_cmd_buffer.submit(vk, submit_fence);
    submit_fence.wait_and_reset(vk);
}

HostToDeviceUploader::HostToDeviceUploader(
    VulkanEngine *const vk, const CommandPool cmd_pool,
    u64 staging_buffer_size)
:vk(vk),
staging_buffer(*vk, staging_buffer_size),
cmd_pool(cmd_pool),
used_staging_buffer_bytes(0)
{
    in_progress_uploads.reserve(128);
    queued_buffer_uploads.reserve(128);
    queued_image_uploads.reserve(128);
    fence_pool.reserve(128);
    cmd_buffer_pool.reserve(128);
    for (int i = 0; i < 16; ++i){
        fence_pool.emplace_back(*vk, false);
    }
    CommandBuffer::make_command_buffers(*vk, cmd_buffer_pool, cmd_pool, 64);
}

HostToDeviceUploader::FreeStagingRegion HostToDeviceUploader::get_free_staging_region(u64 desired_size, bool require_desired_size){
    u64 remaining_size = staging_buffer.capacity_in_bytes - used_staging_buffer_bytes;
    assert((i64)remaining_size > -1);
    if (remaining_size == 0 || (require_desired_size && remaining_size < desired_size)) {
        begin_and_finish_uploads();
        remaining_size = staging_buffer.capacity_in_bytes;
        if (require_desired_size && staging_buffer.capacity_in_bytes < desired_size){
            std::println("Staging buffer not large enough for write with require_desired_size");
            std::abort();
        }
    }

    FreeStagingRegion region{
        .offset = static_cast<u64>(used_staging_buffer_bytes),
        .size = std::min(desired_size, remaining_size),
    };
    used_staging_buffer_bytes += region.size;

    return region;
}

// todo clearer naming convention for dst_offset. which dst? we have the staging as a dst in the memcpy
// and then we have the dst vulkan buffer. maybe just rename it to dst_vk_buffer.
void HostToDeviceUploader::stage_upload(FreeStagingRegion free_region, const void *src, const VulkanBuffer &dst, u64 dst_offset){
    std::memcpy((char*)staging_buffer.get_mapped_data() + free_region.offset,
                (char*)src,
                free_region.size
    );
    VkBufferCopy2 copy{
        .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
        .pNext = nullptr,
        .srcOffset = free_region.offset,
        .dstOffset = dst_offset,
        .size = free_region.size
    };
    for (auto &queued_upload : queued_buffer_uploads){
        if (queued_upload.dst.buffer == dst.buffer){
            queued_upload.copy_info.push_back(copy);
            return;
        }
    }
    queued_buffer_uploads.emplace_back(dst, copy);
}

void HostToDeviceUploader::stage_upload(FreeStagingRegion free_region, const void *src, const Image &dst, const VkBufferImageCopy2 &copy){
    std::memcpy((char*)staging_buffer.get_mapped_data() + free_region.offset,
                (char*)src,
                free_region.size
    );
    for (auto &queued_upload : queued_image_uploads){
        if (queued_upload.dst == dst){
            queued_upload.copy_info.push_back(copy);
            return;
        }
    }
    queued_image_uploads.emplace_back(dst, copy);
}

void HostToDeviceUploader::queue_upload(const void *src, const VulkanBuffer &dst, u64 byte_count, u64 dst_offset){
    assert(byte_count != 0);
    active_upload_dst = &dst;
    active_upload_next_dst_offset = dst_offset;

    u64 remaining_size = byte_count;
    while (remaining_size > 0){
        FreeStagingRegion free_region = get_free_staging_region(remaining_size, false);

        u64 bytes_already_written = byte_count - remaining_size;
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
    // todo: log a warning if we call this and there are no queued uploads
    const auto add_in_progress_upload = [&]{
        CommandBuffer cmd_buffer = cmd_buffer_pool.empty() ? CommandBuffer(*vk, cmd_pool)
                                                           : pop_back_and_return(cmd_buffer_pool);
        GpuFence fence = fence_pool.empty() ? GpuFence(*vk, false)
                                            : pop_back_and_return(fence_pool);
        in_progress_uploads.emplace_back(cmd_buffer, fence);
        return in_progress_uploads.back();
    };

    const InProgressUpload &in_progress_upload = add_in_progress_upload();
    const CommandBuffer &cmd_buffer = in_progress_upload.cmd_buffer;
    cmd_buffer.restart(true);
    for (auto &queued_upload : queued_buffer_uploads){
        cmd_buffer.copy_buffer(staging_buffer, queued_upload.dst, queued_upload.copy_info);
    }
    for (auto &queued_upload : queued_image_uploads){
        cmd_buffer.copy_buffer_to_image(staging_buffer, queued_upload.dst, queued_upload.copy_info);
    }
    cmd_buffer.end();
    cmd_buffer.submit(*vk, in_progress_upload.fence);
    queued_buffer_uploads.clear();
    queued_image_uploads.clear();
}

void HostToDeviceUploader::finish_in_progress_uploads(){
    // todo: log a warning if we call this and there are no in progress uploads
    for (auto &in_progress_upload : in_progress_uploads){
        in_progress_upload.fence.wait_and_reset(*vk);
        fence_pool.push_back(in_progress_upload.fence);
        cmd_buffer_pool.push_back(in_progress_upload.cmd_buffer);
    }
    in_progress_uploads.clear();

    if (!queued_uploads_exist()) { used_staging_buffer_bytes = 0; }
}

void HostToDeviceUploader::begin_and_finish_uploads(){
    begin_uploads();
    finish_in_progress_uploads();
}

void HostToDeviceUploader::queue_begin_finish_uploads(const void *src, const VulkanBuffer &dst, u64 byte_count, u64 dst_offset){
    queue_upload(src, dst, byte_count, dst_offset);
    begin_and_finish_uploads();
}

bool HostToDeviceUploader::queued_uploads_exist(){
    return !queued_buffer_uploads.empty() || !queued_image_uploads.empty();
}
bool HostToDeviceUploader::in_progress_uploads_exist(){
    return !in_progress_uploads.empty();
}
bool HostToDeviceUploader::queued_or_in_progress_uploads_exist(){
    return queued_uploads_exist() || in_progress_uploads_exist();
}
size_t HostToDeviceUploader::queued_uploads_count(){
    return queued_buffer_uploads.size() + queued_image_uploads.size();
}
size_t HostToDeviceUploader::in_progress_uploads_count(){
    return in_progress_uploads.size();
}

// todo unify this with struct_makers from vulkan-engine.cpp and put it in its own file
static VkBufferImageCopy2 buffer_image_copy2(
    const Image &image,
    u64 buffer_offset,
    u32 mip_level,
    u32 base_layer,
    u32 layer_count,
    ImageAspects aspects,
    const VkOffset3D &img_offset = VkOffset3D{})
{
    assert(base_layer + layer_count <= image.layer_count);
    const auto &base_extent = image.extent;
    return VkBufferImageCopy2{
        .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        .pNext = nullptr,
        .bufferOffset = buffer_offset, // for HTDU, we calculate it since we manage the Staging buffer
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource{
            .aspectMask = static_cast<VkImageAspectFlags>(aspects),
            .mipLevel = mip_level,
            .baseArrayLayer = base_layer,
            .layerCount = layer_count,
        },
        .imageOffset = img_offset,
        .imageExtent = {
            .width  = std::max(1u, base_extent.width >> mip_level),
            .height = std::max(1u, base_extent.height >> mip_level),
            .depth  = 1,
        },
    };
}

void HostToDeviceUploader::queue_upload(
    const void *src,
    const Image &dst,
    u64 byte_count,
    u32 mip_level,
    u32 base_layer,
    u32 layer_count,
    ImageAspects aspects,
    const VkOffset3D &img_offset)
{
    FreeStagingRegion free_region = get_free_staging_region(byte_count, true);
    auto img_region = buffer_image_copy2(dst, free_region.offset, mip_level, base_layer, layer_count, aspects, img_offset);
    stage_upload(free_region, src, dst, img_region);
}

void HostToDeviceUploader::start_queue_upload(const VulkanBuffer &dst, u64 dst_offset){
    active_upload_dst = &dst;
    active_upload_next_dst_offset = dst_offset;
}

void HostToDeviceUploader::add_to_last_upload(const void *src, u64 byte_count){
    assert(active_upload_dst != nullptr);
    assert(byte_count != 0);

    FreeStagingRegion free_region = get_free_staging_region(byte_count, true);

    if (queued_buffer_uploads.empty() || queued_buffer_uploads.back().dst != *active_upload_dst){
        VkBufferCopy2 copy{
            .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
            .pNext = nullptr,
            .srcOffset = free_region.offset,
            .dstOffset = active_upload_next_dst_offset,
            .size = byte_count,
        };
        queued_buffer_uploads.emplace_back(*active_upload_dst, copy);
    }
    else {
        QueuedBufferUpload &upload = queued_buffer_uploads.back();
        upload.copy_info.back().size += byte_count;
    }
    std::memcpy((char*)staging_buffer.get_mapped_data() + free_region.offset,
                (const char*)src,
                byte_count
    );
    active_upload_next_dst_offset += byte_count;
}
