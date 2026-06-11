#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bp {

int findOptimalPackSizeV5SIMD(const std::vector<int>& values);

std::vector<uint8_t> encodeBitPackingV2SIMD(const std::vector<int>& originalArray,
                                            const std::vector<int>& bitWidths,
                                            int pack_size);

std::vector<int> decodeBitPackingV2SIMD(const uint8_t* encodedData,
                                       size_t encodedLen,
                                       const std::vector<int>& bitWidths,
                                       int pack_size,
                                       int n);

} // namespace bp
