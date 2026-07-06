module;

//#include <stb/stb_image.h>
#include <tbb/parallel_for_each.h>

#if !VK_PROJ_USE_IMPORT_STD
#include <print>
#include <mutex>
#include <ranges>
#endif

export module imgProcessing;

#if VK_PROJ_USE_IMPORT_STD
import std;
#endif

import types;
import util;

import bc7enc;


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
    std::array<uint8_t, 64> out;
    for (u32 block_pixel_y = 0; block_pixel_y < 4; ++block_pixel_y){
        for (u32 block_pixel_x = 0; block_pixel_x < 4; ++block_pixel_x){
            u32 src_x = std::min(block_x * 4 + block_pixel_x, width - 1);
            u32 src_y = std::min(block_y * 4 + block_pixel_y, height - 1);
            const uint8_t *src_pixel = pixels_RGBA + (src_y * width + src_x) * 4;
            uint8_t *o = out.data() + (block_pixel_y * 4 + block_pixel_x) * 4;
            o[0] = src_pixel[0]; o[1] = src_pixel[1]; o[2] = src_pixel[2]; o[3] = src_pixel[3];
        }
    }
    return out;
}

std::tuple<u32, u32, u64, u64> BCn_image_info(u32 width, u32 height, const u16 BC_level){
    const u32 block_amount_x = (width  + 3) / 4;
    const u32 block_amount_y = (height + 3) / 4;
    const u64 total_blocks = block_amount_x * block_amount_y;
    // Both BC7 and BC5 encode to 16 bytes per 4x4 block
    u32 bytes_per_block = 0;
    switch (BC_level){
    case (7):
    case (5): bytes_per_block = 16; break;
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
            std::byte *dst = reinterpret_cast<std::byte*>(data.data()) + linear_index * 16;
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
                std::byte *dst = reinterpret_cast<std::byte*>(data.data()) + linear_index * 16;
                // Metallic and Roughness are stored in the G and B channels for gltfs
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
                std::byte *dst = reinterpret_cast<std::byte*>(data.data()) + linear_index * 16;
                // Metallic and Roughness are stored in the G and B channels for gltfs
                rgbcx::encode_bc5_hq(dst, block.data(), channel0, channel1, 4, alpha_search_rad, alpha_modes);
            }
        );
    }
    return data;
}

}
