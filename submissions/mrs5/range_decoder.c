/* Five-symbol, 63-bit arithmetic decoding; all probabilities come from Python.
 * Portable C11: no wide-integer extensions, platform intrinsics or payload data.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define TOTAL (UINT64_C(1) << 31)
#define QUARTER (UINT64_C(1) << 61)
#define HALF (UINT64_C(1) << 62)
#define TOP ((UINT64_C(1) << 63) - 1)

/* floor(width * cumulative / 2^31), without a 94-bit intermediate.
 * width <= 2^63 and cumulative <= 2^31 make both products fit uint64_t.
 */
static uint64_t interval_offset(uint64_t width, uint64_t cumulative) {
    return (width >> 31) * cumulative
         + (((width & (TOTAL - 1)) * cumulative) >> 31);
}

static unsigned read_bit(const uint8_t *payload, size_t size, uint64_t *position) {
    uint64_t byte = *position >> 3;
    unsigned shift = 7 - (unsigned)(*position & 7);
    ++*position;
    return byte < size ? (payload[byte] >> shift) & 1u : 0u;
}

/* State is [low, high, code, bit_position]. Commit it only on success.
 * Return -1 for arguments/state, -2 for probabilities, -3 for frequencies,
 * and -4 for an impossible arithmetic interval. Python raises on any error.
 */
int decode_group(uint64_t *state, const uint8_t *payload, size_t size,
                 const float *probabilities, size_t count, int32_t *symbols) {
    uint64_t low, high, code, position;
    size_t index;
    if (!state || !payload || !size || !probabilities || !count || !symbols)
        return -1;
    low = state[0];
    high = state[1];
    code = state[2];
    position = state[3];
    if (high > TOP || low > code || code > high) return -1;

    for (index = 0; index < count; ++index) {
        const float *row = probabilities + 5 * index;
        uint64_t frequency[5], frequency_sum = 0;
        uint64_t width, cumulative = 0, lower = 0, upper = 0;
        double probability_sum = 0.0;
        int64_t adjusted;
        unsigned symbol, winner = 0;

        /* Match float32 -> float64, truncation, and first-maximum tie handling. */
        for (symbol = 0; symbol < 5; ++symbol) {
            double value = (double)row[symbol];
            if (!isfinite(value) || value <= 0.0 || value > 1.00002) return -2;
            probability_sum += value;
            if (row[symbol] > row[winner]) winner = symbol;
            frequency[symbol] = (uint64_t)(value * (double)TOTAL);
            if (!frequency[symbol]) frequency[symbol] = 1;
            frequency_sum += frequency[symbol];
        }
        if (probability_sum < 0.99998 || probability_sum > 1.00002) return -2;
        adjusted = (int64_t)frequency[winner] + (int64_t)TOTAL - (int64_t)frequency_sum;
        if (adjusted <= 0 || adjusted >= (int64_t)TOTAL) return -3;
        frequency[winner] = (uint64_t)adjusted;
        for (symbol = 0; symbol < 5; ++symbol)
            if (!frequency[symbol] || frequency[symbol] >= TOTAL) return -3;

        if (low > code || code > high) return -4;
        width = high - low + 1;
        for (symbol = 0; symbol < 5; ++symbol) {
            cumulative += frequency[symbol];
            upper = interval_offset(width, cumulative);
            /* Equivalent to scaled_code < cumulative, without wide division:
             * code-low < floor(width*cumulative/TOTAL).
             */
            if (code - low < upper) break;
            lower = upper;
        }
        if (symbol == 5 || upper <= lower) return -4;
        high = low + upper - 1;
        low += lower;
        for (;;) {
            uint64_t offset;
            if (high < HALF) offset = 0;
            else if (low >= HALF) offset = HALF;
            else if (low >= QUARTER && high < 3 * QUARTER) offset = QUARTER;
            else break;
            low = (low - offset) << 1;
            high = ((high - offset) << 1) | 1;
            code = ((code - offset) << 1) | read_bit(payload, size, &position);
        }
        symbols[index] = (int32_t)symbol;
    }
    state[0] = low;
    state[1] = high;
    state[2] = code;
    state[3] = position;
    return 0;
}
