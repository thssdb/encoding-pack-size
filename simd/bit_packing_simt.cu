// SIMT (CUDA) bit packing — optimized version.
// Stream format matches scalar/SIMD (MSB-first, same header layout).
#include "bit_packing_simt.h"
#include "bit_packing.h"
#include "bit_writer_msb_fast.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace bp {

namespace {

int* g_d_vals = nullptr;
int* g_d_dec = nullptr;
unsigned int* g_d_out32 = nullptr;
uint8_t* g_d_enc = nullptr;
int* g_d_meta = nullptr;
long long* g_d_gbs = nullptr;
size_t g_cap_vals = 0;
size_t g_cap_out = 0;
size_t g_cap_meta = 0;
size_t g_cap_gbs = 0;

void ensureVals(size_t n) {
    if (n > g_cap_vals) {
        cudaFree(g_d_vals);
        cudaFree(g_d_dec);
        cudaMalloc(&g_d_vals, n * sizeof(int));
        cudaMalloc(&g_d_dec, n * sizeof(int));
        g_cap_vals = n;
    }
}

void ensureOut(size_t bytes) {
    if (bytes > g_cap_out) {
        cudaFree(g_d_out32);
        cudaFree(g_d_enc);
        cudaMalloc(&g_d_out32, ((bytes + 3) / 4) * sizeof(unsigned int));
        cudaMalloc(&g_d_enc, bytes);
        g_cap_out = bytes;
    }
}

void ensureMeta(size_t nInts, size_t nGBS) {
    if (nInts > g_cap_meta) {
        cudaFree(g_d_meta);
        cudaMalloc(&g_d_meta, nInts * sizeof(int));
        g_cap_meta = nInts;
    }
    if (nGBS > g_cap_gbs) {
        cudaFree(g_d_gbs);
        cudaMalloc(&g_d_gbs, nGBS * sizeof(long long));
        g_cap_gbs = nGBS;
    }
}

} // namespace

__global__ void kBitWidths(const int* vals, int* bw, int* gmax, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int v = vals[i];
    atomicMax(gmax, v);
    int mv = (v > 1) ? v : 1;
    bw[i] = 32 - __clz(static_cast<unsigned>(mv));
}

__global__ void kBatchEncode(const int* allVals,
                             unsigned int* allOut32,
                             const int* chunkValOff,
                             const int* chunkLens,
                             const int* packSizes,
                             const int* flatBW,
                             const long long* flatGBS,
                             const int* groupOff,
                             const int* out32Off) {
    const int chunk = blockIdx.x;
    const int tid = threadIdx.x;
    const int vLen = chunkLens[chunk];
    if (tid >= vLen) return;

    const int ps = packSizes[chunk];
    const int gOff = groupOff[chunk];
    const int wOff = out32Off[chunk];
    const int g = tid / ps;
    const int pos = tid % ps;
    const int w = flatBW[gOff + g];

    const unsigned mask = (w >= 32) ? 0xFFFFFFFFu : ((1u << w) - 1u);
    const unsigned v = static_cast<unsigned>(allVals[chunkValOff[chunk] + tid]) & mask;

    const long long bit0 = flatGBS[gOff + g] + static_cast<long long>(pos) * w;
    const int byte_start = static_cast<int>(bit0 >> 3);
    const int bit_offset = static_cast<int>(bit0 & 7);

    const uint64_t window = static_cast<uint64_t>(v) << (64 - bit_offset - w);

    const int nbytes = (bit_offset + w + 7) >> 3;
    const int word0 = byte_start >> 2;
    unsigned lo = 0, hi = 0;
    for (int bi = 0; bi < nbytes; ++bi) {
        const uint8_t bval = static_cast<uint8_t>(window >> (56 - bi * 8));
        const int gb = byte_start + bi;
        const int shift = (gb & 3) * 8;
        if ((gb >> 2) == word0)
            lo |= (static_cast<unsigned>(bval) << shift);
        else
            hi |= (static_cast<unsigned>(bval) << shift);
    }
    if (lo != 0u) atomicOr(&allOut32[wOff + word0], lo);
    if (hi != 0u) atomicOr(&allOut32[wOff + word0 + 1], hi);
}

__global__ void kBatchDecode(const uint8_t* allEnc,
                             int* allDec,
                             const int* chunkValOff,
                             const int* chunkLens,
                             const int* packSizes,
                             const int* flatBW,
                             const long long* flatGBS,
                             const int* groupOff,
                             const int* encOff) {
    const int chunk = blockIdx.x;
    const int tid = threadIdx.x;
    const int vLen = chunkLens[chunk];
    if (tid >= vLen) return;

    const int ps = packSizes[chunk];
    const int gOff = groupOff[chunk];
    const int eOff = encOff[chunk];
    const int g = tid / ps;
    const int pos = tid % ps;
    const int w = flatBW[gOff + g];

    const long long bit0 = flatGBS[gOff + g] + static_cast<long long>(pos) * w;
    const int byte_start = static_cast<int>(bit0 >> 3) + eOff;
    const int bit_offset = static_cast<int>(bit0 & 7);

    const int nbytes = (bit_offset + w + 7) >> 3;
    uint64_t window = 0;
    for (int bi = 0; bi < nbytes; ++bi)
        window = (window << 8) | static_cast<uint64_t>(allEnc[byte_start + bi]);

    const int shift = nbytes * 8 - w - bit_offset;
    const uint64_t mask64 = (w < 64) ? ((1ULL << w) - 1ULL) : ~0ULL;
    allDec[chunkValOff[chunk] + tid] = static_cast<int>((window >> shift) & mask64);
}

static void writeChunkHeader(uint8_t* outBuf, int byteOffset, const std::vector<int>& bws) {
    int maxBW = 0;
    for (int b : bws)
        if (b > maxBW) maxBW = b;
    const int bfbw = (maxBW == 0) ? 1 : (32 - __builtin_clz(static_cast<unsigned>(maxBW)));
    BitWriterMSBFast w;
    w.writeBits(static_cast<uint64_t>(bfbw), 6);
    for (int b : bws) w.writeBits(static_cast<uint64_t>(b), bfbw);
    auto hdr = w.toByteArray();
    std::memcpy(outBuf + byteOffset, hdr.data(), hdr.size());
}

static int fillChunkMeta(const std::vector<int>& vals,
                         int chunkLen,
                         int packSize,
                         std::vector<int>& flatBW_out,
                         std::vector<long long>& flatGBS_out) {
    const int numGroups = (chunkLen + packSize - 1) / packSize;
    std::vector<int> bws(static_cast<size_t>(numGroups));
    int maxBW = 0;
    for (int g = 0; g < numGroups; ++g) {
        const int gs = g * packSize;
        const int ge = std::min(gs + packSize, chunkLen);
        int mx = 0;
        for (int j = gs; j < ge; ++j)
            if (vals[static_cast<size_t>(j)] > mx) mx = vals[static_cast<size_t>(j)];
        bws[static_cast<size_t>(g)] = bp::bit_width(static_cast<int64_t>(std::max(1, mx)));
        if (bws[static_cast<size_t>(g)] > maxBW) maxBW = bws[static_cast<size_t>(g)];
    }
    const int bfbw = (maxBW == 0) ? 1 : (32 - __builtin_clz(static_cast<unsigned>(maxBW)));
    const int headerBits = 6 + numGroups * bfbw;

    long long dataBits = 0;
    for (int g = 0; g < numGroups; ++g) {
        flatGBS_out.push_back(static_cast<long long>(headerBits) + dataBits);
        flatBW_out.push_back(bws[static_cast<size_t>(g)]);
        const int vg = std::min(packSize, chunkLen - g * packSize);
        dataBits += static_cast<long long>(vg) * bws[static_cast<size_t>(g)];
    }
    const long long totalBits = static_cast<long long>(headerBits) + dataBits;
    return static_cast<int>((totalBits + 7) / 8);
}

static SIMTBatchMeta buildBatchMeta(const std::vector<int>& allValues,
                                    int chunkSize,
                                    const std::vector<int>& packSizes) {
    SIMTBatchMeta m;
    const int nTotal = static_cast<int>(allValues.size());
    m.numChunks = (nTotal + chunkSize - 1) / chunkSize;
    m.totalValues = nTotal;
    m.packSizes = packSizes;
    m.chunkLens.resize(static_cast<size_t>(m.numChunks));
    m.chunkValOff.resize(static_cast<size_t>(m.numChunks));
    m.groupOff.resize(static_cast<size_t>(m.numChunks));
    m.encOffsets.resize(static_cast<size_t>(m.numChunks));
    m.out32Offsets.resize(static_cast<size_t>(m.numChunks));

    int runningGroups = 0, runningActualBytes = 0, runningWords = 0;
    for (int ci = 0; ci < m.numChunks; ++ci) {
        const int cs = ci * chunkSize;
        const int clen = std::min(chunkSize, nTotal - cs);
        m.chunkLens[static_cast<size_t>(ci)] = clen;
        m.chunkValOff[static_cast<size_t>(ci)] = cs;
        m.groupOff[static_cast<size_t>(ci)] = runningGroups;
        m.out32Offsets[static_cast<size_t>(ci)] = runningWords;
        m.encOffsets[static_cast<size_t>(ci)] = runningWords * 4;

        std::vector<int> chunk(allValues.begin() + cs, allValues.begin() + cs + clen);
        const int encBytes =
            fillChunkMeta(chunk, clen, packSizes[static_cast<size_t>(ci)], m.flatBW, m.flatGBS);
        const int ng = (clen + packSizes[static_cast<size_t>(ci)] - 1) / packSizes[static_cast<size_t>(ci)];

        runningGroups += ng;
        runningActualBytes += encBytes;
        runningWords += (encBytes + 3) / 4;
    }
    m.totalGroups = runningGroups;
    m.totalEncBytesActual = runningActualBytes;
    m.totalEncWords = runningWords;
    m.totalEncBytes = runningWords * 4;

    m.headerBuf.assign(static_cast<size_t>(m.totalEncBytes), 0);
    for (int ci = 0; ci < m.numChunks; ++ci) {
        const int clen = m.chunkLens[static_cast<size_t>(ci)];
        const int ps = m.packSizes[static_cast<size_t>(ci)];
        const int ng = (clen + ps - 1) / ps;
        std::vector<int> bws(m.flatBW.begin() + m.groupOff[static_cast<size_t>(ci)],
                             m.flatBW.begin() + m.groupOff[static_cast<size_t>(ci)] + ng);
        writeChunkHeader(m.headerBuf.data(), m.encOffsets[static_cast<size_t>(ci)], bws);
    }
    return m;
}

SIMTBatchMeta makeSIMTBatchMetaBP(const std::vector<int>& allValues, int chunkSize) {
    const int nTotal = static_cast<int>(allValues.size());
    const int numChunks = (nTotal + chunkSize - 1) / chunkSize;
    std::vector<int> ps(static_cast<size_t>(numChunks));
    for (int ci = 0; ci < numChunks; ++ci) ps[static_cast<size_t>(ci)] = std::min(chunkSize, nTotal - ci * chunkSize);
    return buildBatchMeta(allValues, chunkSize, ps);
}

SIMTBatchMeta makeSIMTBatchMetaBPFixed(const std::vector<int>& allValues, int chunkSize, int packSize) {
    const int nTotal = static_cast<int>(allValues.size());
    const int numChunks = (nTotal + chunkSize - 1) / chunkSize;
    std::vector<int> ps(static_cast<size_t>(numChunks), packSize);
    return buildBatchMeta(allValues, chunkSize, ps);
}

SIMTBatchMeta makeSIMTBatchMetaPrune(const std::vector<int>& allValues, int chunkSize) {
    const int nTotal = static_cast<int>(allValues.size());
    const int numChunks = (nTotal + chunkSize - 1) / chunkSize;
    std::vector<int> ps(static_cast<size_t>(numChunks));
    for (int ci = 0; ci < numChunks; ++ci) {
        const int cs = ci * chunkSize;
        const int clen = std::min(chunkSize, nTotal - cs);
        std::vector<int> chunk(allValues.begin() + cs, allValues.begin() + cs + clen);
        ps[static_cast<size_t>(ci)] = std::max(1, findOptimalPackSizeV5(chunk));
    }
    return buildBatchMeta(allValues, chunkSize, ps);
}

std::vector<uint8_t> batchEncodeSIMT(const std::vector<int>& allValues, const SIMTBatchMeta& m) {
    const int C = m.numChunks;
    const int G = m.totalGroups;
    const int nMetaInts = 5 * C + G;

    ensureVals(static_cast<size_t>(m.totalValues));
    ensureOut(static_cast<size_t>(m.totalEncBytes));
    ensureMeta(static_cast<size_t>(nMetaInts), static_cast<size_t>(G));

    std::vector<int> h_meta(static_cast<size_t>(nMetaInts));
    std::copy(m.chunkValOff.begin(), m.chunkValOff.end(), h_meta.begin());
    std::copy(m.chunkLens.begin(), m.chunkLens.end(), h_meta.begin() + C);
    std::copy(m.packSizes.begin(), m.packSizes.end(), h_meta.begin() + 2 * C);
    std::copy(m.groupOff.begin(), m.groupOff.end(), h_meta.begin() + 3 * C);
    std::copy(m.out32Offsets.begin(), m.out32Offsets.end(), h_meta.begin() + 4 * C);
    std::copy(m.flatBW.begin(), m.flatBW.end(), h_meta.begin() + 5 * C);

    std::vector<unsigned int> h_header32(static_cast<size_t>(m.totalEncWords), 0);
    std::memcpy(h_header32.data(), m.headerBuf.data(), static_cast<size_t>(m.totalEncBytes));

    cudaMemcpy(g_d_vals, allValues.data(), static_cast<size_t>(m.totalValues) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_meta, h_meta.data(), static_cast<size_t>(nMetaInts) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_gbs, m.flatGBS.data(), static_cast<size_t>(G) * sizeof(long long),
               cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_out32, h_header32.data(),
               static_cast<size_t>(m.totalEncWords) * sizeof(unsigned int), cudaMemcpyHostToDevice);

    kBatchEncode<<<C, 1024>>>(g_d_vals, g_d_out32, g_d_meta, g_d_meta + C, g_d_meta + 2 * C,
                              g_d_meta + 5 * C, g_d_gbs, g_d_meta + 3 * C, g_d_meta + 4 * C);

    cudaMemcpy(h_header32.data(), g_d_out32,
               static_cast<size_t>(m.totalEncWords) * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    std::vector<uint8_t> result(static_cast<size_t>(m.totalEncBytes));
    std::memcpy(result.data(), h_header32.data(), static_cast<size_t>(m.totalEncBytes));
    return result;
}

std::vector<int> batchDecodeSIMT(const std::vector<uint8_t>& flatEncoded, const SIMTBatchMeta& m) {
    const int C = m.numChunks;
    const int G = m.totalGroups;
    const int nMetaInts = 5 * C + G;

    ensureVals(static_cast<size_t>(m.totalValues));
    ensureOut(static_cast<size_t>(m.totalEncBytes));
    ensureMeta(static_cast<size_t>(nMetaInts), static_cast<size_t>(G));

    std::vector<int> h_meta(static_cast<size_t>(nMetaInts));
    std::copy(m.chunkValOff.begin(), m.chunkValOff.end(), h_meta.begin());
    std::copy(m.chunkLens.begin(), m.chunkLens.end(), h_meta.begin() + C);
    std::copy(m.packSizes.begin(), m.packSizes.end(), h_meta.begin() + 2 * C);
    std::copy(m.groupOff.begin(), m.groupOff.end(), h_meta.begin() + 3 * C);
    std::copy(m.encOffsets.begin(), m.encOffsets.end(), h_meta.begin() + 4 * C);
    std::copy(m.flatBW.begin(), m.flatBW.end(), h_meta.begin() + 5 * C);

    cudaMemcpy(g_d_enc, flatEncoded.data(), static_cast<size_t>(m.totalEncBytes),
               cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_meta, h_meta.data(), static_cast<size_t>(nMetaInts) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_gbs, m.flatGBS.data(), static_cast<size_t>(G) * sizeof(long long),
               cudaMemcpyHostToDevice);

    kBatchDecode<<<C, 1024>>>(g_d_enc, g_d_dec, g_d_meta, g_d_meta + C, g_d_meta + 2 * C,
                             g_d_meta + 5 * C, g_d_gbs, g_d_meta + 3 * C, g_d_meta + 4 * C);

    std::vector<int> result(static_cast<size_t>(m.totalValues));
    cudaMemcpy(result.data(), g_d_dec, static_cast<size_t>(m.totalValues) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return result;
}

int findOptimalPackSizeV5SIMT(const std::vector<int>& values) {
    const int n = static_cast<int>(values.size());
    if (n <= 0) return n;

    ensureVals(static_cast<size_t>(n));
    ensureMeta(static_cast<size_t>(n + 1), 1);

    cudaMemcpy(g_d_vals, values.data(), static_cast<size_t>(n) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(g_d_meta, 0, static_cast<size_t>(n + 1) * sizeof(int));

    kBitWidths<<<(n + 255) / 256, 256>>>(g_d_vals, g_d_meta, g_d_meta + n, n);

    std::vector<int> bitwidths(static_cast<size_t>(n));
    int globalMax = 0;
    cudaMemcpy(bitwidths.data(), g_d_meta, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&globalMax, g_d_meta + n, sizeof(int), cudaMemcpyDeviceToHost);

    const int bitWidthGlobal = bit_width(static_cast<int64_t>(std::max(1, globalMax)));
    const int z = static_cast<int>(std::ceil(std::log2(static_cast<double>(bitWidthGlobal + 1))));
    const int logN = (n <= 1) ? 0 : (32 - __builtin_clz(static_cast<unsigned>(n)));

    std::vector<std::vector<int>> st(static_cast<size_t>(logN), std::vector<int>(static_cast<size_t>(n)));
    for (int i = 0; i < n; ++i) st[0][static_cast<size_t>(i)] = bitwidths[static_cast<size_t>(i)];
    for (int k = 1; k < logN; ++k) {
        const int step = 1 << (k - 1);
        for (int i = 0; i + (1 << k) <= n; ++i)
            st[static_cast<size_t>(k)][static_cast<size_t>(i)] =
                std::max(st[static_cast<size_t>(k - 1)][static_cast<size_t>(i)],
                         st[static_cast<size_t>(k - 1)][static_cast<size_t>(i + step)]);
    }

    std::vector<int> log2_table(static_cast<size_t>(n) + 1, 0);
    for (int i = 2; i <= n; ++i) log2_table[static_cast<size_t>(i)] = log2_table[static_cast<size_t>(i / 2)] + 1;

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
                             st[static_cast<size_t>(kk)][static_cast<size_t>(n - (1 << kk))]);
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

std::vector<uint8_t> encodeBitPackingV2SIMT(const std::vector<int>& originalArray,
                                            const std::vector<int>& bitWidths,
                                            int pack_size) {
    const int n = static_cast<int>(originalArray.size());
    const int totalGroups = static_cast<int>(bitWidths.size());

    int maxBW = 0;
    for (int bw : bitWidths)
        if (bw > maxBW) maxBW = bw;
    const int bfbw = (maxBW == 0) ? 1 : (32 - __builtin_clz(static_cast<unsigned>(maxBW)));
    const int headerBits = 6 + totalGroups * bfbw;

    std::vector<long long> groupBitStart(static_cast<size_t>(totalGroups));
    long long dataBits = 0;
    for (int g = 0; g < totalGroups; ++g) {
        const int vg = std::min(pack_size, n - g * pack_size);
        groupBitStart[static_cast<size_t>(g)] = static_cast<long long>(headerBits) + dataBits;
        dataBits += static_cast<long long>(vg) * bitWidths[static_cast<size_t>(g)];
    }

    const long long totalBits = static_cast<long long>(headerBits) + dataBits;
    const int totalBytes = static_cast<int>((totalBits + 7) / 8);
    const int totalWords = (totalBytes + 3) / 4;

    std::vector<uint8_t> outBytes(static_cast<size_t>(totalBytes), 0);
    {
        BitWriterMSBFast w;
        w.writeBits(static_cast<uint64_t>(bfbw), 6);
        for (int bw : bitWidths) w.writeBits(static_cast<uint64_t>(bw), bfbw);
        auto hdr = w.toByteArray();
        std::memcpy(outBytes.data(), hdr.data(), hdr.size());
    }
    std::vector<unsigned int> outWords(static_cast<size_t>(totalWords), 0);
    std::memcpy(outWords.data(), outBytes.data(), static_cast<size_t>(totalBytes));

    ensureVals(static_cast<size_t>(n));
    ensureOut(static_cast<size_t>(totalBytes));
    ensureMeta(static_cast<size_t>(totalGroups), static_cast<size_t>(totalGroups));

    cudaMemcpy(g_d_vals, originalArray.data(), static_cast<size_t>(n) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_out32, outWords.data(), static_cast<size_t>(totalWords) * sizeof(unsigned int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_meta, bitWidths.data(), static_cast<size_t>(totalGroups) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_gbs, groupBitStart.data(), static_cast<size_t>(totalGroups) * sizeof(long long),
               cudaMemcpyHostToDevice);

    int* d_tmp = nullptr;
    cudaMalloc(&d_tmp, 5 * sizeof(int));
    const int h_tmp[5] = {0, n, pack_size, 0, 0};
    cudaMemcpy(d_tmp, h_tmp, 5 * sizeof(int), cudaMemcpyHostToDevice);

    kBatchEncode<<<1, static_cast<unsigned>(std::min(n, 1024))>>>(
        g_d_vals, g_d_out32, d_tmp, d_tmp + 1, d_tmp + 2, g_d_meta, g_d_gbs, d_tmp + 3, d_tmp + 4);
    cudaFree(d_tmp);

    cudaMemcpy(outWords.data(), g_d_out32, static_cast<size_t>(totalWords) * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    std::memcpy(outBytes.data(), outWords.data(), static_cast<size_t>(totalBytes));
    return outBytes;
}

std::vector<int> decodeBitPackingV2SIMT(const uint8_t* encodedData,
                                       size_t encodedLen,
                                       const std::vector<int>& bitWidths,
                                       int pack_size,
                                       int n) {
    if (bitWidths.empty()) throw std::invalid_argument("bitWidths required");
    const int totalGroups = static_cast<int>(bitWidths.size());

    int maxBW = 0;
    for (int bw : bitWidths)
        if (bw > maxBW) maxBW = bw;
    const int bfbw = (maxBW == 0) ? 1 : (32 - __builtin_clz(static_cast<unsigned>(maxBW)));
    const int headerBits = 6 + totalGroups * bfbw;

    std::vector<long long> groupBitStart(static_cast<size_t>(totalGroups));
    long long dataBits = 0;
    for (int g = 0; g < totalGroups; ++g) {
        const int vg = std::min(pack_size, n - g * pack_size);
        groupBitStart[static_cast<size_t>(g)] = static_cast<long long>(headerBits) + dataBits;
        dataBits += static_cast<long long>(vg) * bitWidths[static_cast<size_t>(g)];
    }

    ensureVals(static_cast<size_t>(n));
    ensureOut(encodedLen);
    ensureMeta(static_cast<size_t>(totalGroups), static_cast<size_t>(totalGroups));

    cudaMemcpy(g_d_enc, encodedData, encodedLen, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_meta, bitWidths.data(), static_cast<size_t>(totalGroups) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_gbs, groupBitStart.data(), static_cast<size_t>(totalGroups) * sizeof(long long),
               cudaMemcpyHostToDevice);

    int* d_tmp = nullptr;
    cudaMalloc(&d_tmp, 5 * sizeof(int));
    const int h_tmp[5] = {0, n, pack_size, 0, 0};
    cudaMemcpy(d_tmp, h_tmp, 5 * sizeof(int), cudaMemcpyHostToDevice);

    kBatchDecode<<<1, static_cast<unsigned>(std::min(n, 1024))>>>(
        g_d_enc, g_d_dec, d_tmp, d_tmp + 1, d_tmp + 2, g_d_meta, g_d_gbs, d_tmp + 3, d_tmp + 4);
    cudaFree(d_tmp);

    std::vector<int> decoded(static_cast<size_t>(n));
    cudaMemcpy(decoded.data(), g_d_dec, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost);
    return decoded;
}

} // namespace bp
