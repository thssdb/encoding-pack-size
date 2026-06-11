#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bp {

struct SIMTBatchMeta {
    int numChunks = 0;
    int totalValues = 0;
    std::vector<int> packSizes;
    std::vector<int> chunkLens;
    std::vector<int> chunkValOff;
    std::vector<int> groupOff;
    std::vector<int> encOffsets;
    std::vector<int> out32Offsets;
    std::vector<int> flatBW;
    std::vector<long long> flatGBS;
    int totalGroups = 0;
    int totalEncBytesActual = 0;
    int totalEncWords = 0;
    int totalEncBytes = 0;
    std::vector<uint8_t> headerBuf;
};

SIMTBatchMeta makeSIMTBatchMetaBP(const std::vector<int>& allValues, int chunkSize);
SIMTBatchMeta makeSIMTBatchMetaBPFixed(const std::vector<int>& allValues, int chunkSize, int packSize);
SIMTBatchMeta makeSIMTBatchMetaPrune(const std::vector<int>& allValues, int chunkSize);

std::vector<uint8_t> batchEncodeSIMT(const std::vector<int>& allValues, const SIMTBatchMeta& m);
std::vector<int> batchDecodeSIMT(const std::vector<uint8_t>& flatEncoded, const SIMTBatchMeta& m);

int findOptimalPackSizeV5SIMT(const std::vector<int>& values);

std::vector<uint8_t> encodeBitPackingV2SIMT(const std::vector<int>& originalArray,
                                            const std::vector<int>& bitWidths,
                                            int pack_size);

std::vector<int> decodeBitPackingV2SIMT(const uint8_t* encodedData,
                                       size_t encodedLen,
                                       const std::vector<int>& bitWidths,
                                       int pack_size,
                                       int n);

} // namespace bp
