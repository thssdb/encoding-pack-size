// SIMD-accelerated bit packing and findOptimalPackSize
// Apple M1 / AArch64: NEON; x86: AVX2 (simdcomp-style block SIMD for bit-width + sparse table)
// Reference: https://github.com/fast-pack/simdcomp
#include "bit_packing_simd.h"
#include "bit_packing.h"
#include "bit_writer_msb_fast.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define USE_SSE4 1
#endif

#if (defined(__aarch64__) || defined(_M_ARM64)) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define BP_USE_NEON_OPT_PACK 1
#endif

namespace bp {

const int PREV_ARRAY[1025] = {
#include "prev_array.inc"
};

int findOptimalPackSizeV5(const std::vector<int>& values) {
    const int n = static_cast<int>(values.size());
    if (n < 8) return n;

    std::vector<int> bitWidths(static_cast<size_t>(n));
    int globalMax = 0;
    for (int i = 0; i < n; ++i) {
        int v = values[static_cast<size_t>(i)];
        if (v > globalMax) globalMax = v;
        bitWidths[static_cast<size_t>(i)] = bit_width(static_cast<int64_t>(std::max(1, v)));
    }

    const int bitWidthGlobal = bit_width(static_cast<int64_t>(std::max(1, globalMax)));
    const int z = static_cast<int>(std::ceil(std::log2(static_cast<double>(bitWidthGlobal + 1))));

    const int logN = (n <= 1) ? 0 : (32 - __builtin_clz(static_cast<unsigned>(n)));
    std::vector<std::vector<int>> st(static_cast<size_t>(logN), std::vector<int>(static_cast<size_t>(n)));
    for (int j = 0; j < n; ++j) st[0][static_cast<size_t>(j)] = bitWidths[static_cast<size_t>(j)];

    for (int k = 1; k < logN; ++k) {
        const int step = 1 << (k - 1);
        for (int i = 0; i + (1 << k) <= n; ++i) {
            st[static_cast<size_t>(k)][static_cast<size_t>(i)] =
                std::max(st[static_cast<size_t>(k - 1)][static_cast<size_t>(i)],
                         st[static_cast<size_t>(k - 1)][static_cast<size_t>(i + step)]);
        }
    }

    std::vector<int> log2_table(static_cast<size_t>(n) + 1, 0);
    for (int t = 2; t <= n; ++t) log2_table[static_cast<size_t>(t)] = log2_table[static_cast<size_t>(t / 2)] + 1;

    std::vector<int64_t> cost(static_cast<size_t>(n) + 1, 0);
    std::vector<bool> isIncreased(static_cast<size_t>(n) + 1, false);
    int bestPackSize = n;
    int64_t bestCost = INT64_MAX;

    for (int p = 1; p <= n; ++p) {
        const int prev = (p <= 1024) ? PREV_ARRAY[p] : 0;
        if (prev != 0 && isIncreased[static_cast<size_t>(prev)]) {
            isIncreased[static_cast<size_t>(p)] = true;
            continue;
        }

        const int m = (n + p - 1) / p;
        int64_t currentCost = 0;

        for (int g = 0; g < m - 1; ++g) {
            const int s = g * p;
            const int e = s + p - 1;
            const int kk = log2_table[static_cast<size_t>(p)];
            currentCost += static_cast<int64_t>(p) *
                           std::max(st[static_cast<size_t>(kk)][static_cast<size_t>(s)],
                                    st[static_cast<size_t>(kk)][static_cast<size_t>(e - (1 << kk) + 1)]);
        }
        if (m > 0) {
            const int ls = (m - 1) * p;
            const int r = n - ls;
            if (r > 0) {
                const int kk = log2_table[static_cast<size_t>(r)];
                currentCost +=
                    static_cast<int64_t>(r) *
                    std::max(st[static_cast<size_t>(kk)][static_cast<size_t>(ls)],
                             st[static_cast<size_t>(kk)][static_cast<size_t>(n - 1 - (1 << kk) + 1)]);
            }
        }

        currentCost += static_cast<int64_t>(m) * z;
        cost[static_cast<size_t>(p)] = currentCost;

        if (prev != 0 && currentCost > cost[static_cast<size_t>(prev)]) {
            isIncreased[static_cast<size_t>(p)] = true;
            continue;
        }
        if (currentCost < bestCost) {
            bestCost = currentCost;
            bestPackSize = p;
        }
    }
    return bestPackSize;
}


#if defined(BP_USE_NEON_OPT_PACK)
static inline int32x4_t neon_bit_width_u32(int32x4_t v_nonneg) {
    uint32x4_t u = vreinterpretq_u32_s32(v_nonneg);
    uint32x4_t clz = vclzq_u32(u);
    return vreinterpretq_s32_u32(vsubq_u32(vdupq_n_u32(32), clz));
}

static int findOptimalPackSizeV5_NEON(const std::vector<int>& values) {
    const int n = static_cast<int>(values.size());
    if (n < 8) return n;

    const int* vals = values.data();
    std::vector<int> bitWidths(static_cast<size_t>(n));
    int globalMax = 0;

    int i = 0;
    const int32x4_t one = vdupq_n_s32(1);
    for (; i + 4 <= n; i += 4) {
        int32x4_t v = vld1q_s32(vals + i);
        globalMax = std::max(globalMax, static_cast<int>(vmaxvq_s32(v)));
        int32x4_t vmax = vmaxq_s32(v, one);
        int32x4_t bw = neon_bit_width_u32(vmax);
        vst1q_s32(bitWidths.data() + i, bw);
    }
    for (; i < n; ++i) {
        int v = vals[i];
        if (v > globalMax) globalMax = v;
        bitWidths[static_cast<size_t>(i)] = bit_width(static_cast<int64_t>(std::max(1, v)));
    }

    const int bitWidthGlobal = bit_width(static_cast<int64_t>(std::max(1, globalMax)));
    const int z = static_cast<int>(std::ceil(std::log2(static_cast<double>(bitWidthGlobal + 1))));

    const int logN = (n <= 1) ? 0 : (32 - __builtin_clz(static_cast<unsigned>(n)));
    std::vector<std::vector<int>> st(static_cast<size_t>(logN), std::vector<int>(static_cast<size_t>(n)));

    for (int j = 0; j < n; ++j) st[0][static_cast<size_t>(j)] = bitWidths[static_cast<size_t>(j)];

    for (int k = 1; k < logN; ++k) {
        const int step = 1 << (k - 1);
        const int imax = n - (1 << k);
        int j = 0;
        for (; j + 3 <= imax; j += 4) {
            int32x4_t a = vld1q_s32(st[static_cast<size_t>(k - 1)].data() + j);
            int32x4_t b = vld1q_s32(st[static_cast<size_t>(k - 1)].data() + j + step);
            vst1q_s32(st[static_cast<size_t>(k)].data() + j, vmaxq_s32(a, b));
        }
        for (; j <= imax; ++j) {
            st[static_cast<size_t>(k)][static_cast<size_t>(j)] =
                std::max(st[static_cast<size_t>(k - 1)][static_cast<size_t>(j)],
                         st[static_cast<size_t>(k - 1)][static_cast<size_t>(j + step)]);
        }
    }

    std::vector<int> log2_table(static_cast<size_t>(n) + 1, 0);
    for (int t = 2; t <= n; ++t) log2_table[static_cast<size_t>(t)] = log2_table[static_cast<size_t>(t / 2)] + 1;

    std::vector<int64_t> cost(static_cast<size_t>(n) + 1, 0);
    std::vector<bool> isIncreased(static_cast<size_t>(n) + 1, false);
    const int64_t maxCost = INT64_MAX;
    int bestPackSize = n;
    int64_t bestCost = maxCost;

    for (int p = 1; p <= n; ++p) {
        const int prev = (p <= 1024) ? PREV_ARRAY[p] : 0;
        if (prev != 0 && isIncreased[static_cast<size_t>(prev)]) {
            isIncreased[static_cast<size_t>(p)] = true;
            continue;
        }

        const int m = (n + p - 1) / p;
        int64_t currentCost = 0;
        for (int g = 0; g < m - 1; ++g) {
            const int s = g * p;
            const int e = s + p - 1;
            const int kk = log2_table[static_cast<size_t>(p)];
            currentCost += static_cast<int64_t>(p) *
                           std::max(st[static_cast<size_t>(kk)][static_cast<size_t>(s)],
                                    st[static_cast<size_t>(kk)][static_cast<size_t>(e - (1 << kk) + 1)]);
        }
        if (m > 0) {
            const int ls = (m - 1) * p;
            const int r = n - ls;
            if (r > 0) {
                const int kk = log2_table[static_cast<size_t>(r)];
                currentCost +=
                    static_cast<int64_t>(r) *
                    std::max(st[static_cast<size_t>(kk)][static_cast<size_t>(ls)],
                             st[static_cast<size_t>(kk)][static_cast<size_t>(n - 1 - (1 << kk) + 1)]);
            }
        }
        currentCost += static_cast<int64_t>(m) * z;
        cost[static_cast<size_t>(p)] = currentCost;
        if (prev != 0 && currentCost > cost[static_cast<size_t>(prev)]) {
            isIncreased[static_cast<size_t>(p)] = true;
            continue;
        }
        if (currentCost < bestCost) {
            bestCost = currentCost;
            bestPackSize = p;
        }
    }
    return bestPackSize;
}
#endif // BP_USE_NEON_OPT_PACK

#if defined(USE_AVX2)
static inline int hmax_epi32_avx2(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i m = _mm_max_epi32(lo, hi);
    m = _mm_max_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_max_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(m);
}

static inline __m256i avx2_bit_width(__m256i vals) {
    __m256i clamped = _mm256_max_epi32(vals, _mm256_set1_epi32(1));
    __m256d dlo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(clamped));
    __m256d dhi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(clamped, 1));
    const __m256i sub1022 = _mm256_set1_epi64x(1022LL);
    __m256i elo = _mm256_sub_epi64(_mm256_srli_epi64(_mm256_castpd_si256(dlo), 52), sub1022);
    __m256i ehi = _mm256_sub_epi64(_mm256_srli_epi64(_mm256_castpd_si256(dhi), 52), sub1022);
    const __m256i perm = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
    __m128i bwlo = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(elo, perm));
    __m128i bwhi = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(ehi, perm));
    return _mm256_set_m128i(bwhi, bwlo);
}

static int findOptimalPackSizeV5_AVX2(const std::vector<int>& values) {
    const int n = static_cast<int>(values.size());
    if (n < 8) return n;

    const int* vals = values.data();
    std::vector<int> bitWidths(static_cast<size_t>(n));
    __m256i vGMax = _mm256_setzero_si256();

    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals + i));
        vGMax = _mm256_max_epi32(vGMax, v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(bitWidths.data() + i), avx2_bit_width(v));
    }

    int globalMax = hmax_epi32_avx2(vGMax);
    for (; i < n; ++i) {
        int v = vals[i];
        if (v > globalMax) globalMax = v;
        bitWidths[static_cast<size_t>(i)] = bit_width(static_cast<int64_t>(std::max(1, v)));
    }

    const int bitWidthGlobal = bit_width(static_cast<int64_t>(std::max(1, globalMax)));
    const int z = static_cast<int>(std::ceil(std::log2(static_cast<double>(bitWidthGlobal + 1))));

    const int logN = (n <= 1) ? 0 : (32 - __builtin_clz(static_cast<unsigned>(n)));
    std::vector<std::vector<int>> st(static_cast<size_t>(logN), std::vector<int>(static_cast<size_t>(n)));
    for (int j = 0; j < n; ++j) st[0][static_cast<size_t>(j)] = bitWidths[static_cast<size_t>(j)];

    for (int k = 1; k < logN; ++k) {
        const int step = 1 << (k - 1);
        const int imax = n - (1 << k);
        int j = 0;
        for (; j + 7 <= imax; j += 8) {
            __m256i a =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(st[static_cast<size_t>(k - 1)].data() + j));
            __m256i b = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(st[static_cast<size_t>(k - 1)].data() + j + step));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(st[static_cast<size_t>(k)].data() + j),
                                _mm256_max_epi32(a, b));
        }
        for (; j <= imax; ++j) {
            st[static_cast<size_t>(k)][static_cast<size_t>(j)] =
                std::max(st[static_cast<size_t>(k - 1)][static_cast<size_t>(j)],
                         st[static_cast<size_t>(k - 1)][static_cast<size_t>(j + step)]);
        }
    }

    std::vector<int> log2_table(static_cast<size_t>(n) + 1, 0);
    for (int t = 2; t <= n; ++t) log2_table[static_cast<size_t>(t)] = log2_table[static_cast<size_t>(t / 2)] + 1;

    std::vector<int64_t> cost(static_cast<size_t>(n) + 1, 0);
    std::vector<bool> isIncreased(static_cast<size_t>(n) + 1, false);
    int bestPackSize = n;
    int64_t bestCost = INT64_MAX;

    for (int p = 1; p <= n; ++p) {
        const int prev = (p <= 1024) ? PREV_ARRAY[p] : 0;
        if (prev != 0 && isIncreased[static_cast<size_t>(prev)]) {
            isIncreased[static_cast<size_t>(p)] = true;
            continue;
        }
        const int m = (n + p - 1) / p;
        int64_t currentCost = 0;
        for (int g = 0; g < m - 1; ++g) {
            const int s = g * p;
            const int e = s + p - 1;
            const int kk = log2_table[static_cast<size_t>(p)];
            currentCost += static_cast<int64_t>(p) *
                           std::max(st[static_cast<size_t>(kk)][static_cast<size_t>(s)],
                                    st[static_cast<size_t>(kk)][static_cast<size_t>(e - (1 << kk) + 1)]);
        }
        if (m > 0) {
            const int ls = (m - 1) * p;
            const int r = n - ls;
            if (r > 0) {
                const int kk = log2_table[static_cast<size_t>(r)];
                currentCost +=
                    static_cast<int64_t>(r) *
                    std::max(st[static_cast<size_t>(kk)][static_cast<size_t>(ls)],
                             st[static_cast<size_t>(kk)][static_cast<size_t>(n - 1 - (1 << kk) + 1)]);
            }
        }
        currentCost += static_cast<int64_t>(m) * z;
        cost[static_cast<size_t>(p)] = currentCost;
        if (prev != 0 && currentCost > cost[static_cast<size_t>(prev)]) {
            isIncreased[static_cast<size_t>(p)] = true;
            continue;
        }
        if (currentCost < bestCost) {
            bestCost = currentCost;
            bestPackSize = p;
        }
    }
    return bestPackSize;
}
#endif // USE_AVX2

int findOptimalPackSizeV5SIMD(const std::vector<int>& values) {
#if defined(USE_AVX2)
    return findOptimalPackSizeV5_AVX2(values);
#elif defined(BP_USE_NEON_OPT_PACK)
    return findOptimalPackSizeV5_NEON(values);
#else
    return findOptimalPackSizeV5(values);
#endif
}


std::vector<uint8_t> encodeBitPackingV2SIMD(const std::vector<int>& originalArray,
                                            const std::vector<int>& bitWidths,
                                            int pack_size) {
    const int totalGroups = static_cast<int>(bitWidths.size());
    const int n = static_cast<int>(originalArray.size());
    const int* src = originalArray.data();

    int maxBitWidth = 0;
    for (int bw : bitWidths)
        if (bw > maxBitWidth) maxBitWidth = bw;
    const int bitsForBitWidth = (maxBitWidth == 0) ? 1 : (32 - __builtin_clz(static_cast<unsigned>(maxBitWidth)));

    BitWriterMSBFast w;
    w.writeBits(static_cast<uint64_t>(bitsForBitWidth), 6);
    for (int bw : bitWidths) w.writeBits(static_cast<uint64_t>(bw), bitsForBitWidth);

    for (int group = 0; group < totalGroups; ++group) {
        const int startIndex = group * pack_size;
        const int bitWidth = bitWidths[group];
        const int valuesInGroup = std::min(pack_size, n - startIndex);
        if (valuesInGroup <= 0) break;
        const uint32_t mask = (bitWidth >= 32) ? 0xFFFFFFFFu : ((1u << bitWidth) - 1u);

        int i = 0;
        const bool useChunk8 = (8 * bitWidth <= 64);
        const bool useChunk4 = (4 * bitWidth <= 64);
#if defined(USE_AVX2)
        {
            const __m256i vmask8 = _mm256_set1_epi32(static_cast<int>(mask));
            for (; i + 8 <= valuesInGroup; i += 8) {
                __m256i vv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + startIndex + i));
                __m256i vm = _mm256_and_si256(vv, vmask8);
                uint32_t tmp[8];
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), vm);
                if (useChunk8) {
                    const uint64_t c =
                        (static_cast<uint64_t>(tmp[0]) << (7 * bitWidth)) |
                        (static_cast<uint64_t>(tmp[1]) << (6 * bitWidth)) |
                        (static_cast<uint64_t>(tmp[2]) << (5 * bitWidth)) |
                        (static_cast<uint64_t>(tmp[3]) << (4 * bitWidth)) |
                        (static_cast<uint64_t>(tmp[4]) << (3 * bitWidth)) |
                        (static_cast<uint64_t>(tmp[5]) << (2 * bitWidth)) |
                        (static_cast<uint64_t>(tmp[6]) << (1 * bitWidth)) |
                        (static_cast<uint64_t>(tmp[7]));
                    w.writeBits(c, 8 * bitWidth);
                } else if (useChunk4) {
                    const uint64_t c0 =
                        (static_cast<uint64_t>(tmp[0]) << (3 * bitWidth)) |
                        (static_cast<uint64_t>(tmp[1]) << (2 * bitWidth)) |
                        (static_cast<uint64_t>(tmp[2]) << bitWidth) |
                        (static_cast<uint64_t>(tmp[3]));
                    const uint64_t c1 =
                        (static_cast<uint64_t>(tmp[4]) << (3 * bitWidth)) |
                        (static_cast<uint64_t>(tmp[5]) << (2 * bitWidth)) |
                        (static_cast<uint64_t>(tmp[6]) << bitWidth) |
                        (static_cast<uint64_t>(tmp[7]));
                    w.writeBits(c0, 4 * bitWidth);
                    w.writeBits(c1, 4 * bitWidth);
                } else {
                    for (int j = 0; j < 8; ++j) w.writeBits(static_cast<uint64_t>(tmp[j]), bitWidth);
                }
            }
            for (; i + 4 <= valuesInGroup; i += 4) {
                if (useChunk4) {
                    const uint64_t combined =
                        (static_cast<uint64_t>(static_cast<uint32_t>(src[startIndex + i]) & mask)
                         << (3 * bitWidth)) |
                        (static_cast<uint64_t>(static_cast<uint32_t>(src[startIndex + i + 1]) & mask)
                         << (2 * bitWidth)) |
                        (static_cast<uint64_t>(static_cast<uint32_t>(src[startIndex + i + 2]) & mask)
                         << bitWidth) |
                        (static_cast<uint64_t>(static_cast<uint32_t>(src[startIndex + i + 3]) & mask));
                    w.writeBits(combined, 4 * bitWidth);
                } else {
                    w.writeBits(static_cast<uint64_t>(src[startIndex + i]) & mask, bitWidth);
                    w.writeBits(static_cast<uint64_t>(src[startIndex + i + 1]) & mask, bitWidth);
                    w.writeBits(static_cast<uint64_t>(src[startIndex + i + 2]) & mask, bitWidth);
                    w.writeBits(static_cast<uint64_t>(src[startIndex + i + 3]) & mask, bitWidth);
                }
            }
        }
#elif defined(BP_USE_NEON_OPT_PACK)
        for (; i + 4 <= valuesInGroup; i += 4) {
            int32x4_t vv = vld1q_s32(src + startIndex + i);
            uint32x4_t mq = vdupq_n_u32(mask);
            uint32x4_t vm = vandq_u32(vreinterpretq_u32_s32(vv), mq);
            if (useChunk4) {
                const uint32_t v0 = vgetq_lane_u32(vm, 0);
                const uint32_t v1 = vgetq_lane_u32(vm, 1);
                const uint32_t v2 = vgetq_lane_u32(vm, 2);
                const uint32_t v3 = vgetq_lane_u32(vm, 3);
                const uint64_t combined = (static_cast<uint64_t>(v0) << (3 * bitWidth)) |
                                         (static_cast<uint64_t>(v1) << (2 * bitWidth)) |
                                         (static_cast<uint64_t>(v2) << bitWidth) |
                                         (static_cast<uint64_t>(v3));
                w.writeBits(combined, 4 * bitWidth);
            } else {
                w.writeBits(static_cast<uint64_t>(vgetq_lane_u32(vm, 0)), bitWidth);
                w.writeBits(static_cast<uint64_t>(vgetq_lane_u32(vm, 1)), bitWidth);
                w.writeBits(static_cast<uint64_t>(vgetq_lane_u32(vm, 2)), bitWidth);
                w.writeBits(static_cast<uint64_t>(vgetq_lane_u32(vm, 3)), bitWidth);
            }
        }
#else
        for (; i + 4 <= valuesInGroup; i += 4) {
            if (useChunk4) {
                const uint32_t v0 = static_cast<uint32_t>(src[startIndex + i]) & mask;
                const uint32_t v1 = static_cast<uint32_t>(src[startIndex + i + 1]) & mask;
                const uint32_t v2 = static_cast<uint32_t>(src[startIndex + i + 2]) & mask;
                const uint32_t v3 = static_cast<uint32_t>(src[startIndex + i + 3]) & mask;
                const uint64_t combined = (static_cast<uint64_t>(v0) << (3 * bitWidth)) |
                                         (static_cast<uint64_t>(v1) << (2 * bitWidth)) |
                                         (static_cast<uint64_t>(v2) << bitWidth) |
                                         (static_cast<uint64_t>(v3));
                w.writeBits(combined, 4 * bitWidth);
            } else {
                w.writeBits(static_cast<uint64_t>(src[startIndex + i]) & mask, bitWidth);
                w.writeBits(static_cast<uint64_t>(src[startIndex + i + 1]) & mask, bitWidth);
                w.writeBits(static_cast<uint64_t>(src[startIndex + i + 2]) & mask, bitWidth);
                w.writeBits(static_cast<uint64_t>(src[startIndex + i + 3]) & mask, bitWidth);
            }
        }
#endif
        for (; i < valuesInGroup; ++i)
            w.writeBits(static_cast<uint64_t>(src[startIndex + i]) & mask, bitWidth);
    }
    return w.toByteArray();
}

class BitReaderMSB64 {
public:
    explicit BitReaderMSB64(const uint8_t* data, size_t len)
        : data_(data), len_(len), pos_(0), buf_(0), bits_(0) {}

    uint64_t readBits(int n) {
        if (n <= 0) return 0;
        if (n > 64) n = 64;
        if (n == 64) return (readBits(32) << 32) | readBits(32);
        while (bits_ < n && bits_ <= 56 && pos_ < len_) {
            buf_ = (buf_ << 8) | static_cast<uint64_t>(data_[pos_++]);
            bits_ += 8;
        }
        if (bits_ < n) return 0;
        uint64_t v = (buf_ >> static_cast<unsigned>(bits_ - n)) & ((1ULL << n) - 1ULL);
        bits_ -= n;
        if (bits_ == 0)
            buf_ = 0;
        else if (bits_ == 64)
            buf_ &= ~0ULL;
        else
            buf_ &= (1ULL << bits_) - 1ULL;
        return v;
    }

private:
    const uint8_t* data_;
    size_t len_;
    size_t pos_;
    uint64_t buf_;
    int bits_;
};

std::vector<int> decodeBitPackingV2SIMD(const uint8_t* encodedData, size_t encodedLen,
                                       const std::vector<int>& bitWidths,
                                       int pack_size, int n) {
    if (bitWidths.empty()) throw std::invalid_argument("bitWidths required");
    const int totalGroups = static_cast<int>(bitWidths.size());
    std::vector<int> decoded(static_cast<size_t>(n), 0);
    int* dst = decoded.data();

    int maxBW = 0;
    for (int bw : bitWidths)
        if (bw > maxBW) maxBW = bw;
    const int bitsForBitWidth = (maxBW == 0) ? 1 : (32 - __builtin_clz(static_cast<unsigned>(maxBW)));
    long long gbs = 6LL + static_cast<long long>(totalGroups) * bitsForBitWidth;

    for (int group = 0; group < totalGroups; ++group) {
        const int startIndex = group * pack_size;
        const int bw = bitWidths[group];
        const int valuesInGroup = std::min(pack_size, n - startIndex);
        if (valuesInGroup <= 0) break;

        const long long groupBitStart = gbs;
        gbs += static_cast<long long>(valuesInGroup) * bw;

        const uint64_t mask = (bw >= 32) ? 0xFFFFFFFFULL : ((1ULL << bw) - 1ULL);
        int i = 0;

#if defined(USE_AVX2)
        if (bw <= 14) {
            const __m256i perm64to32 = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
            const __m256i mk64 = _mm256_set1_epi64x(static_cast<long long>(mask));

            if (bw <= 7) {
                for (; i + 8 <= valuesInGroup; i += 8) {
                    const long long b0 = groupBitStart + static_cast<long long>(i) * bw;
                    const int byte_start = static_cast<int>(b0 >> 3);
                    const int bit_offset = static_cast<int>(b0 & 7);
                    const int nbytes = (bit_offset + 8 * bw + 7) / 8;
                    uint64_t window = 0;
                    for (int bi = 0; bi < nbytes; ++bi)
                        window =
                            (window << 8) | static_cast<uint64_t>(encodedData[byte_start + bi]);
                    const int bs = nbytes * 8 - bit_offset - bw;
                    const __m256i vw = _mm256_set1_epi64x(static_cast<long long>(window));
                    const __m256i sh_lo =
                        _mm256_set_epi64x(bs - 3 * bw, bs - 2 * bw, bs - bw, bs);
                    const __m256i sh_hi =
                        _mm256_set_epi64x(bs - 7 * bw, bs - 6 * bw, bs - 5 * bw, bs - 4 * bw);
                    const __m256i r_lo = _mm256_and_si256(_mm256_srlv_epi64(vw, sh_lo), mk64);
                    const __m256i r_hi = _mm256_and_si256(_mm256_srlv_epi64(vw, sh_hi), mk64);
                    const __m128i v_lo =
                        _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(r_lo, perm64to32));
                    const __m128i v_hi =
                        _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(r_hi, perm64to32));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + startIndex + i),
                                        _mm256_set_m128i(v_hi, v_lo));
                }
            }

            for (; i + 4 <= valuesInGroup; i += 4) {
                const long long b0 = groupBitStart + static_cast<long long>(i) * bw;
                const int byte_start = static_cast<int>(b0 >> 3);
                const int bit_offset = static_cast<int>(b0 & 7);
                const int nbytes = (bit_offset + 4 * bw + 7) / 8;
                uint64_t window = 0;
                for (int bi = 0; bi < nbytes; ++bi)
                    window =
                        (window << 8) | static_cast<uint64_t>(encodedData[byte_start + bi]);
                const int bs = nbytes * 8 - bit_offset - bw;
                const __m256i vw = _mm256_set1_epi64x(static_cast<long long>(window));
                const __m256i sh = _mm256_set_epi64x(bs - 3 * bw, bs - 2 * bw, bs - bw, bs);
                const __m256i res = _mm256_and_si256(_mm256_srlv_epi64(vw, sh), mk64);
                _mm_storeu_si128(
                    reinterpret_cast<__m128i*>(dst + startIndex + i),
                    _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(res, perm64to32)));
            }
        }
#endif
        for (; i < valuesInGroup; ++i) {
            const long long b0 = groupBitStart + static_cast<long long>(i) * bw;
            const int byte_start = static_cast<int>(b0 >> 3);
            const int bit_offset = static_cast<int>(b0 & 7);
            const int nbytes = (bit_offset + bw + 7) / 8;
            uint64_t window = 0;
            for (int bi = 0; bi < nbytes; ++bi)
                window = (window << 8) | static_cast<uint64_t>(encodedData[byte_start + bi]);
            dst[startIndex + i] =
                static_cast<int>((window >> (nbytes * 8 - bit_offset - bw)) & mask);
        }
    }
    (void)encodedLen;
    return decoded;
}

} // namespace bp
