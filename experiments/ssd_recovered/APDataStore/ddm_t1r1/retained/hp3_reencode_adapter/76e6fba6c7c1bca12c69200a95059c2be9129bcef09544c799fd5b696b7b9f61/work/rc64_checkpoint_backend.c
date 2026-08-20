#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Streaming 63-bit arithmetic coder for the fixed five-symbol HPAC alphabet.
 *
 * The public probability contract is one positive-frequency row per symbol,
 * with every row summing to 2^31.  A 63-bit interval leaves enough headroom
 * for the 94-bit range/CDF product when GCC/Clang's unsigned __int128 is used.
 * The bitstream has no platform-sized fields and is therefore endian-neutral.
 */

#define RC64_ALPHABET 5u
#define RC64_TOTAL ((uint64_t)1u << 31)
#define RC64_TOP (((uint64_t)1u << 63) - 1u)
#define RC64_FIRST_QTR ((uint64_t)1u << 61)
#define RC64_HALF ((uint64_t)1u << 62)
#define RC64_THIRD_QTR (RC64_FIRST_QTR * 3u)

typedef struct {
    uint64_t low;
    uint64_t high;
    uint64_t pending;
    uint8_t *data;
    size_t size;
    size_t capacity;
    uint8_t partial;
    uint8_t partial_bits;
    int error;
    int finished;
} rc64_encoder;

typedef struct {
    uint64_t low;
    uint64_t high;
    uint64_t code;
    const uint8_t *data;
    size_t size;
    size_t bit_position;
    int error;
} rc64_decoder;

static int rc64_reserve(rc64_encoder *encoder, size_t extra) {
    size_t required;
    size_t capacity;
    uint8_t *replacement;
    if (encoder->error) return 0;
    if (extra > SIZE_MAX - encoder->size) {
        encoder->error = 1;
        return 0;
    }
    required = encoder->size + extra;
    if (required <= encoder->capacity) return 1;
    capacity = encoder->capacity ? encoder->capacity : 4096u;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) {
            capacity = required;
            break;
        }
        capacity *= 2u;
    }
    replacement = (uint8_t *)realloc(encoder->data, capacity);
    if (!replacement) {
        encoder->error = 1;
        return 0;
    }
    encoder->data = replacement;
    encoder->capacity = capacity;
    return 1;
}

static void rc64_put_bit(rc64_encoder *encoder, unsigned bit) {
    if (encoder->error) return;
    encoder->partial = (uint8_t)((encoder->partial << 1u) | (bit & 1u));
    encoder->partial_bits++;
    if (encoder->partial_bits == 8u) {
        if (!rc64_reserve(encoder, 1u)) return;
        encoder->data[encoder->size++] = encoder->partial;
        encoder->partial = 0u;
        encoder->partial_bits = 0u;
    }
}

static void rc64_put_bit_with_pending(rc64_encoder *encoder, unsigned bit) {
    rc64_put_bit(encoder, bit);
    while (encoder->pending && !encoder->error) {
        rc64_put_bit(encoder, bit ^ 1u);
        encoder->pending--;
    }
}

void *rc64_encoder_create(void) {
    rc64_encoder *encoder = (rc64_encoder *)calloc(1u, sizeof(rc64_encoder));
    if (!encoder) return NULL;
    encoder->high = RC64_TOP;
    return encoder;
}

void rc64_encoder_destroy(void *opaque) {
    rc64_encoder *encoder = (rc64_encoder *)opaque;
    if (!encoder) return;
    free(encoder->data);
    free(encoder);
}


int rc64_encoder_snapshot(
    const void *opaque,
    uint64_t *low,
    uint64_t *high,
    uint64_t *pending,
    uint8_t *partial,
    uint8_t *partial_bits,
    const uint8_t **data,
    size_t *size
) {
    const rc64_encoder *encoder = (const rc64_encoder *)opaque;
    if (!encoder || encoder->error || encoder->finished || !low || !high ||
        !pending || !partial || !partial_bits || !data || !size) return -1;
    *low = encoder->low;
    *high = encoder->high;
    *pending = encoder->pending;
    *partial = encoder->partial;
    *partial_bits = encoder->partial_bits;
    *data = encoder->data;
    *size = encoder->size;
    return 0;
}

void *rc64_encoder_resume(
    uint64_t low,
    uint64_t high,
    uint64_t pending,
    uint8_t partial,
    uint8_t partial_bits,
    const uint8_t *data,
    size_t size
) {
    rc64_encoder *encoder = (rc64_encoder *)rc64_encoder_create();
    if (!encoder || partial_bits > 7u || low > high || high > RC64_TOP) {
        rc64_encoder_destroy(encoder);
        return NULL;
    }
    if (size && (!data || !rc64_reserve(encoder, size))) {
        rc64_encoder_destroy(encoder);
        return NULL;
    }
    if (size) memcpy(encoder->data, data, size);
    encoder->size = size;
    encoder->low = low;
    encoder->high = high;
    encoder->pending = pending;
    encoder->partial = partial;
    encoder->partial_bits = partial_bits;
    return encoder;
}

int rc64_encoder_encode(
    void *opaque,
    const int32_t *symbols,
    const uint32_t *frequencies,
    size_t count
) {
    rc64_encoder *encoder = (rc64_encoder *)opaque;
    size_t index;
    if (!encoder || (!symbols && count) || (!frequencies && count)) return -1;
    if (encoder->error || encoder->finished) return -2;
    for (index = 0u; index < count; ++index) {
        const uint32_t *row = frequencies + index * RC64_ALPHABET;
        int32_t symbol = symbols[index];
        uint64_t cumulative_low = 0u;
        uint64_t cumulative_high;
        uint64_t total = 0u;
        uint64_t width;
        uint64_t lower_offset;
        uint64_t upper_offset;
        unsigned item;
        if (symbol < 0 || symbol >= (int32_t)RC64_ALPHABET) return -3;
        for (item = 0u; item < RC64_ALPHABET; ++item) {
            if (!row[item]) return -4;
            total += row[item];
            if (item < (unsigned)symbol) cumulative_low += row[item];
        }
        if (total != RC64_TOTAL) return -5;
        cumulative_high = cumulative_low + row[(unsigned)symbol];
        width = encoder->high - encoder->low + 1u;
        lower_offset = (uint64_t)(((__uint128_t)width * cumulative_low) >> 31u);
        upper_offset = (uint64_t)(((__uint128_t)width * cumulative_high) >> 31u);
        if (upper_offset <= lower_offset) return -6;
        encoder->high = encoder->low + upper_offset - 1u;
        encoder->low += lower_offset;
        for (;;) {
            if (encoder->high < RC64_HALF) {
                rc64_put_bit_with_pending(encoder, 0u);
            } else if (encoder->low >= RC64_HALF) {
                rc64_put_bit_with_pending(encoder, 1u);
                encoder->low -= RC64_HALF;
                encoder->high -= RC64_HALF;
            } else if (
                encoder->low >= RC64_FIRST_QTR &&
                encoder->high < RC64_THIRD_QTR
            ) {
                encoder->pending++;
                encoder->low -= RC64_FIRST_QTR;
                encoder->high -= RC64_FIRST_QTR;
            } else {
                break;
            }
            encoder->low <<= 1u;
            encoder->high = (encoder->high << 1u) | 1u;
            if (encoder->error) return -7;
        }
    }
    return encoder->error ? -7 : 0;
}

int rc64_encoder_finish(void *opaque) {
    rc64_encoder *encoder = (rc64_encoder *)opaque;
    if (!encoder || encoder->error) return -1;
    if (!encoder->finished) {
        encoder->pending++;
        rc64_put_bit_with_pending(
            encoder,
            encoder->low < RC64_FIRST_QTR ? 0u : 1u
        );
        if (encoder->partial_bits) {
            encoder->partial <<= (uint8_t)(8u - encoder->partial_bits);
            if (!rc64_reserve(encoder, 1u)) return -2;
            encoder->data[encoder->size++] = encoder->partial;
            encoder->partial = 0u;
            encoder->partial_bits = 0u;
        }
        encoder->finished = 1;
    }
    return encoder->error ? -2 : 0;
}

const uint8_t *rc64_encoder_data(const void *opaque) {
    const rc64_encoder *encoder = (const rc64_encoder *)opaque;
    return encoder && encoder->finished && !encoder->error ? encoder->data : NULL;
}

size_t rc64_encoder_size(const void *opaque) {
    const rc64_encoder *encoder = (const rc64_encoder *)opaque;
    return encoder && encoder->finished && !encoder->error ? encoder->size : 0u;
}

static unsigned rc64_read_bit(rc64_decoder *decoder) {
    size_t byte_index = decoder->bit_position >> 3u;
    unsigned bit_index = (unsigned)(decoder->bit_position & 7u);
    unsigned bit = 0u;
    if (byte_index < decoder->size) {
        bit = (unsigned)((decoder->data[byte_index] >> (7u - bit_index)) & 1u);
    }
    decoder->bit_position++;
    return bit;
}

void *rc64_decoder_create(const uint8_t *data, size_t size) {
    rc64_decoder *decoder;
    unsigned bit;
    if (!data || !size) return NULL;
    decoder = (rc64_decoder *)calloc(1u, sizeof(rc64_decoder));
    if (!decoder) return NULL;
    decoder->data = data;
    decoder->size = size;
    decoder->high = RC64_TOP;
    for (bit = 0u; bit < 63u; ++bit) {
        decoder->code = (decoder->code << 1u) | rc64_read_bit(decoder);
    }
    return decoder;
}

void rc64_decoder_destroy(void *opaque) {
    free(opaque);
}

static int rc64_decoder_decode_row(
    rc64_decoder *decoder,
    const uint32_t *row,
    int32_t *output
) {
    uint64_t total = 0u;
    uint64_t width = decoder->high - decoder->low + 1u;
    uint64_t scaled;
    uint64_t cumulative_low = 0u;
    uint64_t cumulative_high = 0u;
    uint64_t lower_offset;
    uint64_t upper_offset;
    unsigned symbol;
    for (symbol = 0u; symbol < RC64_ALPHABET; ++symbol) {
        if (!row[symbol]) return -1;
        total += row[symbol];
    }
    if (
        total != RC64_TOTAL ||
        decoder->code < decoder->low ||
        decoder->code > decoder->high
    ) return -2;
    scaled = (uint64_t)(
        (((__uint128_t)(decoder->code - decoder->low + 1u) * RC64_TOTAL) - 1u) /
        width
    );
    for (symbol = 0u; symbol < RC64_ALPHABET; ++symbol) {
        cumulative_high += row[symbol];
        if (scaled < cumulative_high) break;
        cumulative_low = cumulative_high;
    }
    if (symbol == RC64_ALPHABET) return -3;
    lower_offset = (uint64_t)(((__uint128_t)width * cumulative_low) >> 31u);
    upper_offset = (uint64_t)(((__uint128_t)width * cumulative_high) >> 31u);
    if (upper_offset <= lower_offset) return -4;
    decoder->high = decoder->low + upper_offset - 1u;
    decoder->low += lower_offset;
    for (;;) {
        if (decoder->high < RC64_HALF) {
            /* no offset */
        } else if (decoder->low >= RC64_HALF) {
            decoder->code -= RC64_HALF;
            decoder->low -= RC64_HALF;
            decoder->high -= RC64_HALF;
        } else if (
            decoder->low >= RC64_FIRST_QTR &&
            decoder->high < RC64_THIRD_QTR
        ) {
            decoder->code -= RC64_FIRST_QTR;
            decoder->low -= RC64_FIRST_QTR;
            decoder->high -= RC64_FIRST_QTR;
        } else {
            break;
        }
        decoder->low <<= 1u;
        decoder->high = (decoder->high << 1u) | 1u;
        decoder->code = (decoder->code << 1u) | rc64_read_bit(decoder);
    }
    *output = (int32_t)symbol;
    return 0;
}

int rc64_decoder_decode(
    void *opaque,
    const uint32_t *frequencies,
    size_t count,
    int32_t *symbols
) {
    rc64_decoder *decoder = (rc64_decoder *)opaque;
    size_t index;
    int status;
    if (!decoder || (!symbols && count) || (!frequencies && count)) return -1;
    if (decoder->error) return -2;
    for (index = 0u; index < count; ++index) {
        status = rc64_decoder_decode_row(
            decoder,
            frequencies + index * RC64_ALPHABET,
            symbols + index
        );
        if (status) return status - 2;
    }
    return 0;
}

/*
 * Fuse the production float32-to-frequency conversion with decoding.  Every
 * input float is promoted exactly to double and multiplied by a power of two,
 * matching NumPy's float64 floor path without materializing temporary arrays.
 */
int rc64_decoder_decode_probabilities(
    void *opaque,
    const float *probabilities,
    size_t count,
    int32_t *symbols
) {
    rc64_decoder *decoder = (rc64_decoder *)opaque;
    size_t index;
    if (!decoder || (!symbols && count) || (!probabilities && count)) return -1;
    if (decoder->error) return -2;
    for (index = 0u; index < count; ++index) {
        const float *row = probabilities + index * RC64_ALPHABET;
        uint32_t frequencies[RC64_ALPHABET];
        uint64_t frequency_sum = 0u;
        double probability_sum = 0.0;
        unsigned winner = 0u;
        unsigned symbol;
        int64_t balance;
        int64_t adjusted;
        int status;
        for (symbol = 0u; symbol < RC64_ALPHABET; ++symbol) {
            double value = (double)row[symbol];
            uint64_t frequency;
            if (!isfinite(value) || value <= 0.0) return -3;
            probability_sum += value;
            if (row[symbol] > row[winner]) winner = symbol;
            if (value > 1.00002) return -4;
            frequency = (uint64_t)(value * (double)RC64_TOTAL);
            if (frequency < 1u) frequency = 1u;
            frequencies[symbol] = (uint32_t)frequency;
            frequency_sum += frequency;
        }
        if (probability_sum < 0.99998 || probability_sum > 1.00002) return -5;
        balance = (int64_t)RC64_TOTAL - (int64_t)frequency_sum;
        adjusted = (int64_t)frequencies[winner] + balance;
        if (adjusted <= 0 || adjusted >= (int64_t)RC64_TOTAL) return -6;
        frequencies[winner] = (uint32_t)adjusted;
        for (symbol = 0u; symbol < RC64_ALPHABET; ++symbol) {
            if (!frequencies[symbol] || frequencies[symbol] >= RC64_TOTAL) {
                return -6;
            }
        }
        status = rc64_decoder_decode_row(decoder, frequencies, symbols + index);
        if (status) return status - 6;
    }
    return 0;
}

size_t rc64_decoder_bit_position(const void *opaque) {
    const rc64_decoder *decoder = (const rc64_decoder *)opaque;
    return decoder ? decoder->bit_position : 0u;
}

uint64_t rc64_total_frequency(void) {
    return RC64_TOTAL;
}
