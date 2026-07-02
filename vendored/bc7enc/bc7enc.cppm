/*
    NOTE: not a file that originally existed in the bc7enc codebase. This exists to offer a module version
    of this library.
*/

module;

#include "bc7enc.h"
#include "rgbcx.h"

export module bc7enc;

export {
    using ::color_rgba;
    using ::bc7enc_compress_block_params;

    using ::bc7enc_compress_block_params_init_linear_weights;
    using ::bc7enc_compress_block_params_init_perceptual_weights;
    using ::bc7enc_compress_block_params_init;

    using ::bc7enc_compress_block_init;
    using ::bc7enc_compress_block;
}

export namespace rgbcx
{
    using ::rgbcx::bc1_approx_mode;
    using ::rgbcx::eNoClamp;
    using ::rgbcx::color32;

    using ::rgbcx::maximum;
    using ::rgbcx::minimum;

    using ::rgbcx::init;

    using ::rgbcx::encode_bc1_solid_block;

    using ::rgbcx::encode_bc1;
    using ::rgbcx::encode_bc3;
    using ::rgbcx::encode_bc3_hq;
    using ::rgbcx::encode_bc4;
    using ::rgbcx::encode_bc4_hq;
    using ::rgbcx::encode_bc5;
    using ::rgbcx::encode_bc5_hq;

    using ::rgbcx::unpack_bc1_block_colors;
    using ::rgbcx::unpack_bc1;
    using ::rgbcx::unpack_bc3;
    using ::rgbcx::unpack_bc4;
    using ::rgbcx::unpack_bc5;

    using ::rgbcx::bc1_block;
    using ::rgbcx::bc4_block;

    using ::rgbcx::dxt_constants;

    using ::rgbcx::MIN_TOTAL_ORDERINGS;
    using ::rgbcx::MAX_TOTAL_ORDERINGS3;
    using ::rgbcx::MAX_TOTAL_ORDERINGS4;
    using ::rgbcx::DEFAULT_TOTAL_ORDERINGS_TO_TRY;
    using ::rgbcx::DEFAULT_TOTAL_ORDERINGS_TO_TRY3;

    using ::rgbcx::MIN_LEVEL;
    using ::rgbcx::MAX_LEVEL;

    using ::rgbcx::BC4_DEFAULT_SEARCH_RAD;
    using ::rgbcx::BC4_USE_MODE8_FLAG;
    using ::rgbcx::BC4_USE_MODE6_FLAG;
    using ::rgbcx::BC4_USE_ALL_MODES;
}
