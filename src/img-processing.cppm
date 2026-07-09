module;

#include <tbb/parallel_for_each.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <stb/stb_image_resize2.h>

#if !VK_PROJ_USE_IMPORT_STD
#include <print>
#include <mutex>
#include <ranges>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <cstring>
#endif

export module imgProcessing;

#if VK_PROJ_USE_IMPORT_STD
import std;
#endif

import types;
import util;

import bc7enc;
import glm;


inline static std::once_flag init_bc7e_once{};
static void init_img_processing(){
    std::call_once(init_bc7e_once, [](){
        bc7enc_compress_block_init();
        rgbcx::init();
    });
}

export enum class BCnQuality : std::uint8_t {
    MIN,
    BALANCED,
    MAX,
};

export enum class MipQuality : std::uint8_t {
    NO_MIPS,
    MIN,
    BALANCED,
    MAX,
};

export enum class BC7QualityType : std::uint8_t {
    MATHEMATICAL,
    PERCEPTUAL
};

static constexpr bc7enc_compress_block_params create_bc7enc_compress_block_params(BCnQuality quality, BC7QualityType quality_type)
{
    bc7enc_compress_block_params p{};
    p.m_mode_mask = UINT32_MAX;
    p.m_force_alpha = false;
    p.m_force_selectors = false;
    p.m_quant_mode6_endpoints = false;
    p.m_bias_mode1_pbits = false;
    p.m_pbit1_weight = 1.0f;
    p.m_mode1_error_weight = 1.0f;
    p.m_mode5_error_weight = 1.0f;
    p.m_mode6_error_weight = 1.0f;
    p.m_mode7_error_weight = 1.0f;
    p.m_low_frequency_partition_weight = 1.0f;

    if (quality == BCnQuality::MAX)
    {
        p.m_max_partitions = 64; // BC7ENC_MAX_PARTITIONS
        p.m_uber_level = 4;     // BC7ENC_MAX_UBER_LEVEL
        p.m_try_least_squares = true;
        p.m_mode17_partition_estimation_filterbank = false;
    }
    else if (quality == BCnQuality::BALANCED){
        p.m_max_partitions = 64; // BC7ENC_MAX_PARTITIONS
        p.m_uber_level = 0;
        p.m_try_least_squares = true;
        p.m_mode17_partition_estimation_filterbank = true;
    }
    else if (quality == BCnQuality::MIN)
    {
        p.m_max_partitions = 0;
        p.m_uber_level = 0;
        p.m_try_least_squares = false;
        p.m_mode17_partition_estimation_filterbank = true;
    }
    else { std::println("Unrecognised BCnQuality"); abort(); }

    if (quality_type == BC7QualityType::PERCEPTUAL){
        p.m_perceptual = true;
        p.m_weights[0] = 128;
        p.m_weights[1] = 64;
        p.m_weights[2] = 16;
        p.m_weights[3] = 32;
    }
    else if (quality_type == BC7QualityType::MATHEMATICAL)
    {
        p.m_perceptual = false;
        p.m_weights[0] = 1;
        p.m_weights[1] = 1;
        p.m_weights[2] = 1;
        p.m_weights[3] = 1;
    }
    else { std::println("Unrecognised BC7QualityType"); abort(); }

    return p;
}

// Extract one 4x4 RGBA block from src, clamping at image edges
static std::array<uint8_t, 64> extract_BCn_block(
    const uint8_t* pixels_RGBA,
    u32 width, u32 height,
    u32 block_x, u32 block_y)
{
    std::array<uint8_t, 64> out{};
    for (u32 block_pixel_y = 0; block_pixel_y < 4; ++block_pixel_y){
        for (u32 block_pixel_x = 0; block_pixel_x < 4; ++block_pixel_x){
            u32 src_x = std::min(block_x * 4 + block_pixel_x, width - 1);
            u32 src_y = std::min(block_y * 4 + block_pixel_y, height - 1);
            const uint8_t *src_pixel = pixels_RGBA + (u64)(src_y * width + src_x) * 4;
            uint8_t *o = out.data() + (u64)(block_pixel_y * 4 + block_pixel_x) * 4;
            o[0] = src_pixel[0]; o[1] = src_pixel[1]; o[2] = src_pixel[2]; o[3] = src_pixel[3];
        }
    }
    return out;
}

static std::tuple<u32, u32, u64, u64> BCn_image_info(u32 width, u32 height, const u16 BC_level){
    const u32 block_amount_x = (width  + 3) / 4;
    const u32 block_amount_y = (height + 3) / 4;
    const u64 total_blocks = (u64)block_amount_x * block_amount_y;
    // Both BC7 and BC5 encode to 16 bytes per 4x4 block
    u32 bytes_per_block = 0;
    switch (BC_level){
    case (7):
    case (5): bytes_per_block = 16; break;
    default: std::println("BC levels range from 1 to 7"); abort();
    }
    const u64 compressed_image_size = total_blocks * bytes_per_block;
    return std::make_tuple(block_amount_x, block_amount_y, total_blocks, compressed_image_size);
}

export{

std::vector<std::byte> to_bc7(const uint8_t* pixels_RGBA, u32 width, u32 height, const BCnQuality quality, const BC7QualityType quality_type){
    init_img_processing();
    auto [block_amount_x, block_amount_y, total_blocks, compressed_image_size]
        = BCn_image_info(width, height, 7);
    std::vector<std::byte> data;
    data.resize(compressed_image_size);

    auto block_grid = std::views::cartesian_product(
        std::views::iota(0u, block_amount_y),
        std::views::iota(0u, block_amount_x)
    );
    bc7enc_compress_block_params params = create_bc7enc_compress_block_params(quality, quality_type);

    //todo: check that this lays out the threads in a cache efficient way
    tbb::parallel_for_each(block_grid.begin(), block_grid.end(),
        [&](const auto &block_coordinates) {
            auto [block_y, block_x] = block_coordinates;
            std::array<uint8_t, 64> block = extract_BCn_block(pixels_RGBA, width, height, block_x, block_y);
            u32 linear_index = block_y * block_amount_x + block_x;
            std::byte *dst = reinterpret_cast<std::byte*>(data.data()) + (u64)linear_index * 16;
            bc7enc_compress_block(dst, block.data(), &params);
        }
    );
    return data;
}


std::vector<std::byte> to_bc5(const uint8_t *pixels_RGBA, u32 width, u32 height, i32 channel0, i32 channel1, BCnQuality quality){
    init_img_processing();
    auto [block_amount_x, block_amount_y, total_blocks, compressed_image_size]
        = BCn_image_info(width, height, 5);
    std::vector<std::byte> data;
    data.resize(compressed_image_size);

    auto block_grid = std::views::cartesian_product(
        std::views::iota(0u, block_amount_y),
        std::views::iota(0u, block_amount_x)
    );
    if (quality == BCnQuality::MIN){
        tbb::parallel_for_each(block_grid.begin(), block_grid.end(),
            [&](const auto &block_coordinates) {
                auto [block_y, block_x] = block_coordinates;
                std::array<uint8_t, 64> block = extract_BCn_block(pixels_RGBA, width, height, block_x, block_y);
                u32 linear_index = block_y * block_amount_x + block_x;
                std::byte *dst = reinterpret_cast<std::byte*>(data.data()) + (u64)linear_index * 16;
                rgbcx::encode_bc5(dst, block.data(), channel0, channel1, 4);
            }
        );
    } else {
        u32 alpha_search_rad{}, alpha_modes{};
        if (quality == BCnQuality::BALANCED){
            alpha_search_rad = 2;
            alpha_modes = (1 << 0); // mode 0: 8-value, endpoint0 > endpoint1
        }else if (quality == BCnQuality::MAX){
            alpha_search_rad = 8;
            alpha_modes = 3; // both modes
        }
        else{ std::println("Unknown BCnQuality in to_bc5"); abort(); }

        tbb::parallel_for_each(block_grid.begin(), block_grid.end(),
            [&](const auto &block_coordinates) {
                auto [block_y, block_x] = block_coordinates;
                std::array<uint8_t, 64> block = extract_BCn_block(pixels_RGBA, width, height, block_x, block_y);
                u32 linear_index = block_y * block_amount_x + block_x;
                std::byte *dst = reinterpret_cast<std::byte*>(data.data()) + (u64)linear_index * 16;
                // Metallic and Roughness are stored in the G and B channels for gltfs
                rgbcx::encode_bc5_hq(dst, block.data(), channel0, channel1, 4, alpha_search_rad, alpha_modes);
            }
        );
    }
    return data;
}

// Generates a mipmap chain based on the selected quality level, returns all levels
// with mip[0] being the original
std::vector<std::vector<std::byte>> make_mips_color(
    const u8* pixels_RGBA,
    i32 width,
    i32 height,
    bool is_sRGB,
    MipQuality quality)
{
    if ((pixels_RGBA == nullptr) || width <= 0 || height <= 0) {
        return {};
    }

    // Handle NO_MIPS: return a vector of vectors containing only the original image
    if (quality == MipQuality::NO_MIPS) {
        std::vector<std::vector<std::byte>> mips(1);
        mips[0].resize((u64)width * height * 4);
        std::memcpy(mips[0].data(), pixels_RGBA, (u64)width * height * 4);
        return mips;
    }

    i32 mip_levels = 1 + static_cast<i32>(std::floor(std::log2(std::max(width, height))));
    std::vector<std::vector<std::byte>> mips(mip_levels);

    mips[0].resize((u64)width * height * 4);
    std::memcpy(mips[0].data(), pixels_RGBA, (u64)width * height * 4);

    for (i32 level = 1; level < mip_levels; ++level) {
        i32 dst_w = std::max(1, width >> level);
        i32 dst_h = std::max(1, height >> level);
        mips[level].resize((u64)dst_w * dst_h * 4);

        // MIN generates from previous mip level (level - 1)
        // BALANCED and MAX generate from the original image (level 0)
        i32 src_level = (quality == MipQuality::MIN) ? level - 1 : 0;
        i32 src_w = std::max(1, width >> src_level);
        i32 src_h = std::max(1, height >> src_level);

        const void* src_pixels = mips[src_level].data();
        void* dst_pixels = mips[level].data();

        STBIR_RESIZE resize;
        stbir_resize_init(
            &resize,
            src_pixels, src_w, src_h, 0,
            dst_pixels, dst_w, dst_h, 0,
            STBIR_RGBA,
            is_sRGB ? STBIR_TYPE_UINT8_SRGB : STBIR_TYPE_UINT8
        );

        // Configure filter type based on quality
        // MAX uses Mitchell (stb's high-quality cubic windowed filter), others use BOX
        stbir_filter filter_type = (quality == MipQuality::MAX) ? STBIR_FILTER_MITCHELL : STBIR_FILTER_BOX;
        stbir_set_filters(&resize, filter_type, filter_type);

        i32 splits = stbir_build_samplers_with_splits(&resize, tbb::this_task_arena::max_concurrency());
        if (splits > 0) {
            tbb::parallel_for(0, splits, [&resize](i32 i) {
                stbir_resize_extended_split(&resize, i, 1);
            });
        }

        stbir_free_samplers(&resize);
    }
    return mips;
}

std::vector<std::vector<std::byte>> make_mips_normals(
    const u8* pixels_RGBA,
    i32 width,
    i32 height,
    MipQuality quality)
{
    if ((pixels_RGBA == nullptr) || width <= 0 || height <= 0) {
        return {};
    }

    if (quality == MipQuality::NO_MIPS) {
        std::vector<std::vector<std::byte>> mips(1);
        mips[0].resize((u64)width * height * 4);
        std::memcpy(mips[0].data(), pixels_RGBA, (u64)width * height * 4);
        return mips;
    }

    i32 mip_levels = 1 + static_cast<i32>(std::floor(std::log2(std::max(width, height))));
    std::vector<std::vector<std::byte>> mips(mip_levels);

    mips[0].resize((u64)width * height * 4);
    std::memcpy(mips[0].data(), pixels_RGBA, (u64)width * height * 4);

    constexpr i32 x_channel = 0;
    constexpr i32 y_channel = 1;

    for (i32 level = 1; level < mip_levels; ++level) {
        i32 dst_w = std::max(1, width >> level);
        i32 dst_h = std::max(1, height >> level);
        mips[level].resize((u64)dst_w * dst_h * 4);

        i32 src_level = (quality == MipQuality::MIN) ? level - 1 : 0;
        i32 src_w = std::max(1, width >> src_level);
        i32 src_h = std::max(1, height >> src_level);

        const u8* src_pixels = reinterpret_cast<const u8*>(mips[src_level].data());
        u8* dst_pixels = reinterpret_cast<u8*>(mips[level].data());

        auto pixel_grid = std::views::cartesian_product(
            std::views::iota(0, dst_h),
            std::views::iota(0, dst_w)
        );

        tbb::parallel_for_each(pixel_grid.begin(), pixel_grid.end(), [=](const auto& coordinates) {
            auto [dst_y, dst_x] = coordinates;

            i32 src_x_start = (dst_x * src_w) / dst_w;
            i32 src_x_end   = ((dst_x + 1) * src_w) / dst_w;
            i32 src_y_start = (dst_y * src_h) / dst_h;
            i32 src_y_end   = ((dst_y + 1) * src_h) / dst_h;

            fvec3 summed_normal(0.0f);
            i32 sample_count = 0;

            for (i32 src_y = src_y_start; src_y < src_y_end; ++src_y) {
                for (i32 src_x = src_x_start; src_x < src_x_end; ++src_x) {
                    i32 src_pixel_offset = (src_y * src_w + src_x) * 4;

                    // Decode to [-1.0, 1.0]
                    fvec2 decoded_normal_xy(static_cast<f32>(src_pixels[src_pixel_offset + x_channel]),
                                            static_cast<f32>(src_pixels[src_pixel_offset + y_channel]));
                    decoded_normal_xy = (decoded_normal_xy / 127.5f) - 1.0f;

                    // Reconstruct Z
                    f32 reconstructed_z_squared = 1.0f - glm::dot(decoded_normal_xy, decoded_normal_xy);
                    f32 reconstructed_z = (reconstructed_z_squared > 0.0f) ? std::sqrt(reconstructed_z_squared) : 0.0f;

                    summed_normal += fvec3(decoded_normal_xy, reconstructed_z);

                    sample_count++;
                }
            }
            if (sample_count <= 0) {
                std::println("make_mips_normals generated pixel with zero samples");
                abort();
            }

            fvec2 averaged_normal_xy(0.0f);
            f32 summed_normal_length = glm::length(summed_normal);
            if (summed_normal_length > 1e-6f) {// prevent going to Inf
                fvec3 normalized_normal = summed_normal / summed_normal_length;
                averaged_normal_xy = fvec2(normalized_normal.x, normalized_normal.y);
            } // if xy are near zero, z will just be reconstructed as ~1.

            // Encode back to [0, 255]
            i32 dst_pixel_offset = (dst_y * dst_w + dst_x) * 4;
            fvec2 encoded_normal_xy = glm::clamp((averaged_normal_xy * 0.5f + 0.5f) * 255.0f + 0.5f, 0.0f, 255.0f);

            dst_pixels[dst_pixel_offset + x_channel] = static_cast<u8>(encoded_normal_xy.x);
            dst_pixels[dst_pixel_offset + y_channel] = static_cast<u8>(encoded_normal_xy.y);
        });
    }

    return mips;
}

std::vector<std::vector<std::byte>> make_mips_metal_rough(
    const u8* pixels_RGBA,
    i32 width,
    i32 height,
    MipQuality quality)
{
    if ((pixels_RGBA == nullptr) || width <= 0 || height <= 0) {
        return {};
    }

    if (quality == MipQuality::NO_MIPS) {
        std::vector<std::vector<std::byte>> mips(1);
        mips[0].resize((u64)width * height * 4);
        std::memcpy(mips[0].data(), pixels_RGBA, (u64)width * height * 4);
        return mips;
    }

    i32 mip_levels = 1 + static_cast<i32>(std::floor(std::log2(std::max(width, height))));
    std::vector<std::vector<std::byte>> mips(mip_levels);

    mips[0].resize((u64)width * height * 4);
    std::memcpy(mips[0].data(), pixels_RGBA, (u64)width * height * 4);

    // In gltfs, roughness is in green channel and metal in blue
    constexpr i32 roughness_channel = 1;
    constexpr i32 metallic_channel  = 2;

    for (i32 level = 1; level < mip_levels; ++level) {
        i32 dst_w = std::max(1, width >> level);
        i32 dst_h = std::max(1, height >> level);
        mips[level].resize((u64)dst_w * dst_h * 4);

        i32 src_level = (quality == MipQuality::MIN) ? level - 1 : 0;
        i32 src_w = std::max(1, width >> src_level);
        i32 src_h = std::max(1, height >> src_level);

        const u8* src_pixels = reinterpret_cast<const u8*>(mips[src_level].data());
        u8* dst_pixels = reinterpret_cast<u8*>(mips[level].data());

        auto pixel_grid = std::views::cartesian_product(
            std::views::iota(0, dst_h),
            std::views::iota(0, dst_w)
        );
        tbb::parallel_for_each(pixel_grid.begin(), pixel_grid.end(), [=](const auto& coordinates) {
            auto [dst_y, dst_x] = coordinates;

            i32 src_x_start = (dst_x * src_w) / dst_w;
            i32 src_x_end   = ((dst_x + 1) * src_w) / dst_w;
            i32 src_y_start = (dst_y * src_h) / dst_h;
            i32 src_y_end   = ((dst_y + 1) * src_h) / dst_h;

            f32 summed_roughness_sq = 0.0f;
            u32 summed_metallic = 0;
            i32 sample_count = 0;

            for (i32 src_y = src_y_start; src_y < src_y_end; ++src_y) {
                for (i32 src_x = src_x_start; src_x < src_x_end; ++src_x) {
                    i32 src_pixel_offset = (src_y * src_w + src_x) * 4;

                    // Accumulate squared roughness values for RMS calculation
                    // this is the physically correct way to do it
                    f32 r = src_pixels[src_pixel_offset + roughness_channel];
                    summed_roughness_sq += r * r;

                    summed_metallic += src_pixels[src_pixel_offset + metallic_channel];

                    sample_count++;
                }
            }

            if (sample_count <= 0) {
                std::println("make_mips_metal_rough generated pixel with zero samples");
                abort();
            }
            i32 dst_pixel_offset = (dst_y * dst_w + dst_x) * 4;

            // Compute Root-Mean-Square (RMS) for Roughness
            f32 avg_roughness_sq = summed_roughness_sq / static_cast<f32>(sample_count);
            f32 final_roughness = std::sqrt(avg_roughness_sq);
            dst_pixels[dst_pixel_offset + roughness_channel] = static_cast<u8>(std::clamp(final_roughness + 0.5f, 0.0f, 255.0f));

            // Compute Arithmetic Mean for Metallic (with rounding)
            dst_pixels[dst_pixel_offset + metallic_channel] = static_cast<u8>((summed_metallic + sample_count / 2) / sample_count);
        });
    }

    return mips;
}

}
