#pragma once

#include <cstdint>
#include <vector>

// MSB-first bit writer (same bit order as Java BitWriterV2 in this repo).
class BitWriterMSBFast {
public:
    void writeBits(uint64_t value, int numBits) {
        if (numBits <= 0) return;
        for (int i = numBits - 1; i >= 0; --i) {
            int bit = static_cast<int>((value >> i) & 1ULL);
            if (bit)
                current_byte_ |= (1 << bit_position_);
            bit_position_--;
            if (bit_position_ < 0) {
                bytes_.push_back(static_cast<uint8_t>(current_byte_));
                current_byte_ = 0;
                bit_position_ = 7;
            }
        }
    }

    std::vector<uint8_t> toByteArray() {
        if (bit_position_ != 7) {
            bytes_.push_back(static_cast<uint8_t>(current_byte_));
        }
        return bytes_;
    }

private:
    std::vector<uint8_t> bytes_;
    int current_byte_ = 0;
    int bit_position_ = 7;
};
