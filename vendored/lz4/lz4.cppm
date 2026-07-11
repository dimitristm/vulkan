/*
    Not included in the original LZ4 library, this is here to provide a module interface to LZ4.
*/
module;

#include "lz4.h"
#include "lz4hc.h"

export module lz4;

#undef LZ4_VERSION_MAJOR
#undef LZ4_VERSION_MINOR
#undef LZ4_VERSION_RELEASE
#undef LZ4_VERSION_NUMBER
#undef LZ4_VERSION_STRING
#undef LZ4_MAX_INPUT_SIZE
#undef LZ4_COMPRESSBOUND
#undef LZ4_DECODER_RING_BUFFER_SIZE

#undef LZ4HC_CLEVEL_MIN
#undef LZ4HC_CLEVEL_DEFAULT
#undef LZ4HC_CLEVEL_OPT_MIN
#undef LZ4HC_CLEVEL_MAX

export {
    // Types
    using ::LZ4_stream_t;
    using ::LZ4_streamDecode_t;
    using ::LZ4_streamHC_t;

    // Defines
    constexpr int LZ4_VERSION_MAJOR = 1;
    constexpr int LZ4_VERSION_MINOR = 10;
    constexpr int LZ4_VERSION_RELEASE = 0;
    constexpr int LZ4_VERSION_NUMBER = 11000;
    constexpr const char* LZ4_VERSION_STRING = "1.10.0";

    constexpr int LZ4_MAX_INPUT_SIZE = 0x7E000000;

    constexpr int LZ4HC_CLEVEL_MIN = 2;
    constexpr int LZ4HC_CLEVEL_DEFAULT = 9;
    constexpr int LZ4HC_CLEVEL_OPT_MIN = 10;
    constexpr int LZ4HC_CLEVEL_MAX = 12;

    // Macros
    constexpr int LZ4_COMPRESSBOUND(int isize) {
        return (static_cast<unsigned>(isize) > static_cast<unsigned>(LZ4_MAX_INPUT_SIZE))
            ? 0
            : (isize + (isize / 255) + 16);
    }

    constexpr int LZ4_DECODER_RING_BUFFER_SIZE(int maxBlockSize) {
        return 65536 + 14 + maxBlockSize;
    }

    // lz4.h functions
    using ::LZ4_versionNumber;
    using ::LZ4_versionString;
    using ::LZ4_compress_default;
    using ::LZ4_decompress_safe;
    using ::LZ4_compressBound;
    using ::LZ4_compress_fast;
    using ::LZ4_sizeofState;
    using ::LZ4_compress_fast_extState;
    using ::LZ4_compress_destSize;
    using ::LZ4_decompress_safe_partial;

    // Streaming Compression
    // Excluded because they heap allocate
    // using ::LZ4_createStream;
    // using ::LZ4_freeStream;

    using ::LZ4_resetStream_fast;
    using ::LZ4_loadDict;
    using ::LZ4_loadDictSlow;
    using ::LZ4_attach_dictionary;
    using ::LZ4_compress_fast_continue;
    using ::LZ4_saveDict;
    using ::LZ4_initStream;

    // Streaming Decompression
    // Excluded because they heap allocate
    // using ::LZ4_createStreamDecode;
    // using ::LZ4_freeStreamDecode;

    using ::LZ4_setStreamDecode;
    using ::LZ4_decoderRingBufferSize;
    using ::LZ4_decompress_safe_continue;
    using ::LZ4_decompress_safe_usingDict;
    using ::LZ4_decompress_safe_partial_usingDict;

    // lz4hc.h functions
    using ::LZ4_compress_HC;
    using ::LZ4_sizeofStateHC;
    using ::LZ4_compress_HC_extStateHC;
    using ::LZ4_compress_HC_destSize;

    // HC Streaming
    // Excluded because they heap allocate
    // using ::LZ4_createStreamHC;
    // using ::LZ4_freeStreamHC;

    using ::LZ4_resetStreamHC_fast;
    using ::LZ4_loadDictHC;
    using ::LZ4_compress_HC_continue;
    using ::LZ4_compress_HC_continue_destSize;
    using ::LZ4_saveDictHC;
    using ::LZ4_attach_HC_dictionary;
    using ::LZ4_initStreamHC;
}
