
/*
    NOTE: not a file that originally existed in the XXHash codebase. This exists to offer a module version
    of this library.
*/

module;

#define XXH_STATIC_LINKING_ONLY
#include "xxhash.h"

export module xxhash;
export {
    using ::XXH32_hash_t;
    using ::XXH64_hash_t;
    using ::XXH_errorcode;
    using ::XXH_OK;
    using ::XXH_ERROR;
    using ::XXH_versionNumber;

    using ::XXH32_canonical_t;
    using ::XXH32_state_t;
    using ::XXH32;
    using ::XXH32_createState;
    using ::XXH32_freeState;
    using ::XXH32_copyState;
    using ::XXH32_reset;
    using ::XXH32_update;
    using ::XXH32_digest;
    using ::XXH32_canonicalFromHash;
    using ::XXH32_hashFromCanonical;

    using ::XXH64_canonical_t;
    using ::XXH64_state_t;
    using ::XXH64;
    using ::XXH64_createState;
    using ::XXH64_freeState;
    using ::XXH64_copyState;
    using ::XXH64_reset;
    using ::XXH64_update;
    using ::XXH64_digest;
    using ::XXH64_canonicalFromHash;
    using ::XXH64_hashFromCanonical;

    using ::XXH128_hash_t;
    using ::XXH128_canonical_t;
    using ::XXH3_state_t;

    using ::XXH3_64bits;
    using ::XXH3_64bits_withSeed;
    using ::XXH3_64bits_withSecret;
    using ::XXH3_createState;
    using ::XXH3_freeState;
    using ::XXH3_copyState;
    using ::XXH3_64bits_reset;
    using ::XXH3_64bits_reset_withSeed;
    using ::XXH3_64bits_reset_withSecret;
    using ::XXH3_64bits_update;
    using ::XXH3_64bits_digest;

    using ::XXH3_128bits;
    using ::XXH3_128bits_withSeed;
    using ::XXH3_128bits_withSecret;
    using ::XXH3_128bits_reset;
    using ::XXH3_128bits_reset_withSeed;
    using ::XXH3_128bits_reset_withSecret;
    using ::XXH3_128bits_update;
    using ::XXH3_128bits_digest;

    using ::XXH128_isEqual;
    using ::XXH128_cmp;
    using ::XXH128_canonicalFromHash;
    using ::XXH128_hashFromCanonical;

    using ::XXH3_64bits_reset_withSecretandSeed;
    using ::XXH3_128bits_withSecretandSeed;
    using ::XXH3_128bits_reset_withSecretandSeed;
    using ::XXH128;
    using ::XXH3_generateSecret;
    using ::XXH3_generateSecret_fromSeed;
}
