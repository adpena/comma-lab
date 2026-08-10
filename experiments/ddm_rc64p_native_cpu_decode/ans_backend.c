#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Byte-compatible decoder for constriction's DefaultAnsCoder<u32, u64>
 * together with Categorical(perfect=False), float32 probabilities, and the
 * five-symbol lc2 HPAC alphabet.  The probability precision is 24 bits.
 *
 * This file contains no learned or video-derived constants.  The only fixed
 * values describe the public entropy grammar used by the counted payload.
 */

#define LC2_ANS_ALPHABET 5u
#define LC2_ANS_PRECISION 24u
#define LC2_ANS_TOTAL ((uint32_t)1u << LC2_ANS_PRECISION)
#define LC2_ANS_MASK (LC2_ANS_TOTAL - 1u)
#define LC2_ANS_REFILL_THRESHOLD ((uint64_t)1u << 32u)

typedef struct {
    const uint8_t *payload;
    size_t payload_bytes;
    size_t words_remaining;
    uint64_t state;
    int error;
} lc2_ans_decoder;

static uint32_t read_le_u32(const uint8_t *source) {
    return ((uint32_t)source[0]) |
           ((uint32_t)source[1] << 8u) |
           ((uint32_t)source[2] << 16u) |
           ((uint32_t)source[3] << 24u);
}

static uint32_t pop_word(lc2_ans_decoder *decoder) {
    decoder->words_remaining--;
    return read_le_u32(decoder->payload + 4u * decoder->words_remaining);
}

void *lc2_ans_decoder_create(const uint8_t *payload, size_t payload_bytes) {
    lc2_ans_decoder *decoder;
    uint32_t first;

    if (!payload || !payload_bytes || payload_bytes % 4u != 0u) return NULL;
    decoder = (lc2_ans_decoder *)calloc(1u, sizeof(lc2_ans_decoder));
    if (!decoder) return NULL;
    decoder->payload = payload;
    decoder->payload_bytes = payload_bytes;
    decoder->words_remaining = payload_bytes / 4u;

    first = pop_word(decoder);
    if (!first) {
        free(decoder);
        return NULL;
    }
    decoder->state = first;
    while (
        decoder->words_remaining &&
        decoder->state < LC2_ANS_REFILL_THRESHOLD
    ) {
        decoder->state = (decoder->state << 32u) | pop_word(decoder);
    }
    return decoder;
}

void lc2_ans_decoder_destroy(void *opaque) {
    free(opaque);
}

static int categorical_interval(
    const float *probabilities,
    uint32_t quantile,
    uint32_t *symbol_out,
    uint32_t *left_out,
    uint32_t *frequency_out
) {
    float normalization = 0.0f;
    float scale;
    float cumulative = 0.0f;
    uint32_t left = 0u;
    unsigned symbol;

    for (symbol = 0u; symbol < LC2_ANS_ALPHABET; ++symbol) {
        float value = probabilities[symbol];
        if (!isfinite(value) || value < 0.0f) return -1;
        normalization += value;
    }
    if (!isnormal(normalization) || !(normalization > 0.0f)) return -2;

    /*
     * constriction's fast leaky categorical model reserves one count for
     * each symbol and distributes the remaining 2^24 - alphabet counts in
     * proportion to the supplied float32 PMF.
     */
    scale = (float)(LC2_ANS_TOTAL - LC2_ANS_ALPHABET) / normalization;
    for (symbol = 0u; symbol < LC2_ANS_ALPHABET; ++symbol) {
        uint32_t right;
        cumulative += probabilities[symbol];
        if (symbol + 1u == LC2_ANS_ALPHABET) {
            right = LC2_ANS_TOTAL;
        } else {
            right = (uint32_t)(cumulative * scale) + symbol + 1u;
        }
        if (right <= left) return -3;
        if (quantile < right) {
            *symbol_out = symbol;
            *left_out = left;
            *frequency_out = right - left;
            return 0;
        }
        left = right;
    }
    return -4;
}

int lc2_ans_decoder_decode_probabilities(
    void *opaque,
    const float *probabilities,
    size_t count,
    int32_t *symbols
) {
    lc2_ans_decoder *decoder = (lc2_ans_decoder *)opaque;
    size_t index;

    if (!decoder || (!probabilities && count) || (!symbols && count)) return -1;
    if (decoder->error) return -2;
    for (index = 0u; index < count; ++index) {
        uint32_t quantile = (uint32_t)decoder->state & LC2_ANS_MASK;
        uint32_t symbol;
        uint32_t left;
        uint32_t frequency;
        uint32_t remainder;
        int status = categorical_interval(
            probabilities + index * LC2_ANS_ALPHABET,
            quantile,
            &symbol,
            &left,
            &frequency
        );
        if (status) {
            decoder->error = status;
            return status - 10;
        }
        remainder = quantile - left;
        decoder->state =
            (decoder->state >> LC2_ANS_PRECISION) * (uint64_t)frequency +
            (uint64_t)remainder;
        if (
            decoder->state < LC2_ANS_REFILL_THRESHOLD &&
            decoder->words_remaining
        ) {
            decoder->state = (decoder->state << 32u) | pop_word(decoder);
        }
        symbols[index] = (int32_t)symbol;
    }
    return 0;
}

int lc2_ans_decoder_is_empty(const void *opaque) {
    const lc2_ans_decoder *decoder = (const lc2_ans_decoder *)opaque;
    return decoder && !decoder->error && decoder->state == 0u;
}

size_t lc2_ans_decoder_words_remaining(const void *opaque) {
    const lc2_ans_decoder *decoder = (const lc2_ans_decoder *)opaque;
    return decoder ? decoder->words_remaining : 0u;
}

uint64_t lc2_ans_decoder_state(const void *opaque) {
    const lc2_ans_decoder *decoder = (const lc2_ans_decoder *)opaque;
    return decoder ? decoder->state : 0u;
}

size_t lc2_ans_decoder_snapshot_words(
    const void *opaque,
    uint32_t *output,
    size_t capacity
) {
    const lc2_ans_decoder *decoder = (const lc2_ans_decoder *)opaque;
    size_t state_words;
    size_t required;
    uint32_t low;
    uint32_t high;
    size_t index;

    if (!decoder || decoder->error) return SIZE_MAX;
    low = (uint32_t)decoder->state;
    high = (uint32_t)(decoder->state >> 32u);
    state_words = high ? 2u : (low ? 1u : 0u);
    required = decoder->words_remaining + state_words;
    if (!output) return required;
    if (capacity < required) return SIZE_MAX;

    for (index = 0u; index < decoder->words_remaining; ++index) {
        output[index] = read_le_u32(decoder->payload + 4u * index);
    }
    if (state_words) output[decoder->words_remaining] = low;
    if (state_words == 2u) output[decoder->words_remaining + 1u] = high;
    return required;
}

uint32_t lc2_ans_precision(void) {
    return LC2_ANS_PRECISION;
}

uint32_t lc2_ans_alphabet(void) {
    return LC2_ANS_ALPHABET;
}
