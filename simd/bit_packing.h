#pragma once

#include <cstdint>
#include <vector>

namespace bp {

inline int bit_width(int64_t v) {
    if (v <= 0) return 0;
    return 64 - __builtin_clzll(static_cast<uint64_t>(v));
}

// Pruning table for pack-size search (matches Java DynamicPacking).
extern const int PREV_ARRAY[1025];

int findOptimalPackSizeV5(const std::vector<int>& values);

} // namespace bp
