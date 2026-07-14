module;

#include "meshoptimizer.h"

export module meshoptimizer;

#undef MESHOPTIMIZER_VERSION

export {
    // Types
    using ::meshopt_Stream;
    using ::meshopt_VertexCacheStatistics;
    using ::meshopt_VertexFetchStatistics;
    using ::meshopt_OverdrawStatistics;
    using ::meshopt_CoverageStatistics;
    using ::meshopt_Meshlet;
    using ::meshopt_Bounds;

    // Defines
    constexpr int MESHOPTIMIZER_VERSION = 1020;

    // Enums
    using ::meshopt_EncodeExpMode;
    using ::meshopt_EncodeExpSeparate;
    using ::meshopt_EncodeExpSharedVector;
    using ::meshopt_EncodeExpSharedComponent;
    using ::meshopt_EncodeExpClamped;

    // Simplification Options
    using ::meshopt_SimplifyLockBorder;
    using ::meshopt_SimplifySparse;
    using ::meshopt_SimplifyErrorAbsolute;
    using ::meshopt_SimplifyPrune;
    using ::meshopt_SimplifyRegularize;
    using ::meshopt_SimplifyPermissive;
    using ::meshopt_SimplifyRegularizeLight;

    // Simplification Vertex Flags
    using ::meshopt_SimplifyVertex_Lock;
    using ::meshopt_SimplifyVertex_Protect;
    using ::meshopt_SimplifyVertex_Priority;

    // Functions (Brings in both C API and C++ template overloads)
    using ::meshopt_generateVertexRemap;
    using ::meshopt_generateVertexRemapMulti;
    using ::meshopt_generateVertexRemapCustom;
    using ::meshopt_remapVertexBuffer;
    using ::meshopt_remapIndexBuffer;
    using ::meshopt_filterIndexBuffer;
    using ::meshopt_filterIndexBufferMulti;
    using ::meshopt_generateShadowIndexBuffer;
    using ::meshopt_generateShadowIndexBufferMulti;
    using ::meshopt_generatePositionRemap;
    using ::meshopt_generateAdjacencyIndexBuffer;
    using ::meshopt_generateTessellationIndexBuffer;
    using ::meshopt_generateProvokingIndexBuffer;
    using ::meshopt_optimizeVertexCache;
    using ::meshopt_optimizeVertexCacheStrip;
    using ::meshopt_optimizeVertexCacheFifo;
    using ::meshopt_optimizeOverdraw;
    using ::meshopt_optimizeVertexFetch;
    using ::meshopt_optimizeVertexFetchRemap;
    using ::meshopt_encodeIndexBuffer;
    using ::meshopt_encodeIndexBufferBound;
    using ::meshopt_encodeIndexVersion;
    using ::meshopt_decodeIndexBuffer;
    using ::meshopt_decodeIndexVersion;
    using ::meshopt_encodeIndexSequence;
    using ::meshopt_encodeIndexSequenceBound;
    using ::meshopt_decodeIndexSequence;
    using ::meshopt_encodeMeshlet;
    using ::meshopt_encodeMeshletBound;
    using ::meshopt_decodeMeshlet;
    using ::meshopt_decodeMeshletRaw;
    using ::meshopt_encodeVertexBuffer;
    using ::meshopt_encodeVertexBufferBound;
    using ::meshopt_encodeVertexBufferLevel;
    using ::meshopt_encodeVertexVersion;
    using ::meshopt_decodeVertexBuffer;
    using ::meshopt_decodeVertexVersion;
    using ::meshopt_decodeFilterOct;
    using ::meshopt_decodeFilterQuat;
    using ::meshopt_decodeFilterExp;
    using ::meshopt_decodeFilterColor;
    using ::meshopt_encodeFilterOct;
    using ::meshopt_encodeFilterQuat;
    using ::meshopt_encodeFilterExp;
    using ::meshopt_encodeFilterColor;
    using ::meshopt_simplify;
    using ::meshopt_simplifyWithAttributes;
    using ::meshopt_simplifyWithUpdate;
    using ::meshopt_simplifySloppy;
    using ::meshopt_simplifyPrune;
    using ::meshopt_simplifyPoints;
    using ::meshopt_simplifyScale;
    using ::meshopt_stripify;
    using ::meshopt_stripifyBound;
    using ::meshopt_unstripify;
    using ::meshopt_unstripifyBound;
    using ::meshopt_analyzeVertexCache;
    using ::meshopt_analyzeVertexFetch;
    using ::meshopt_analyzeOverdraw;
    using ::meshopt_analyzeCoverage;
    using ::meshopt_buildMeshlets;
    using ::meshopt_buildMeshletsScan;
    using ::meshopt_buildMeshletsBound;
    using ::meshopt_buildMeshletsFlex;
    using ::meshopt_buildMeshletsSpatial;
    using ::meshopt_optimizeMeshlet;
    using ::meshopt_optimizeMeshletLevel;
    using ::meshopt_computeClusterBounds;
    using ::meshopt_computeMeshletBounds;
    using ::meshopt_computeSphereBounds;
    using ::meshopt_extractMeshletIndices;
    using ::meshopt_partitionClusters;
    using ::meshopt_spatialSortRemap;
    using ::meshopt_spatialSortTriangles;
    using ::meshopt_spatialClusterPoints;
    using ::meshopt_opacityMapMeasure;
    using ::meshopt_opacityMapRasterize;
    using ::meshopt_opacityMapEntrySize;
    using ::meshopt_opacityMapCompact;
    using ::meshopt_generateTangents;
    using ::meshopt_quantizeHalf;
    using ::meshopt_quantizeFloat;
    using ::meshopt_dequantizeHalf;
    using ::meshopt_computePositionExponent;
    using ::meshopt_setAllocator;
    
    // Inline C++ Quantization Functions
    using ::meshopt_quantizeUnorm;
    using ::meshopt_quantizeSnorm;
}
