#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Generic native lowering of F26's integer sparse-HPAC -> probability -> RC64
 * receiver.  Learned and video-derived values are supplied by the archive via
 * the Python binding; this source contains only the fixed decoder algorithm.
 */

#define F26_ALPHABET 5
#define F26_TOTAL ((uint64_t)1u << 31)
#define F26_TOP (((uint64_t)1u << 63) - 1u)
#define F26_FIRST_QTR ((uint64_t)1u << 61)
#define F26_HALF ((uint64_t)1u << 62)
#define F26_THIRD_QTR (F26_FIRST_QTR * 3u)

typedef struct {
    uint64_t low;
    uint64_t high;
    uint64_t code;
    const uint8_t *data;
    size_t size;
    size_t bit_position;
    int error;
} f26_rc64_decoder;

typedef struct {
    uint64_t low;
    uint64_t high;
    uint64_t code;
    uint64_t bit_position;
    int32_t error;
} f26_rc64_state;

typedef struct {
    int32_t height;
    int32_t width;
    int32_t patch;
    int32_t patch_rows;
    int32_t patch_cols;
    int32_t patch_count;
    int32_t channels;
    int32_t groups;
    int32_t a_taps;
    int32_t b1_taps;
    int32_t b2_taps;
    int32_t logit_precision;

    /* conv_a_weight is [7, a_taps, channels], channel-contiguous. */
    const int8_t *conv_a_weight;
    /* One exact coordinate contribution per [concatenated h position, channel]. */
    const int32_t *conv_a_coordinate;
    const int16_t *conv_a_bias;
    const int8_t *conv_a_exponent;
    const int8_t *b1_weight;
    const int16_t *b1_bias;
    const int8_t *b1_exponent;
    const int8_t *b2_weight;
    const int16_t *b2_bias;
    const int8_t *b2_exponent;
    const int8_t *head_weight;
    const int16_t *head_bias;
    const int8_t *head_exponent;
    const float *residual_table;

    const int16_t *a_offsets;
    const int32_t *group_h_offsets;
    const int32_t *group_b1_offsets;
    const int32_t *group_target_offsets;
    const int32_t *group_output_offsets;
    const int16_t *h_positions;
    const int16_t *b1_gather;
    const int16_t *b2_gather;
    const int32_t *output_order;
    const int32_t *flat_positions;
} f26_hpac_model;

static unsigned f26_read_bit(f26_rc64_decoder *decoder) {
    size_t byte_index = decoder->bit_position >> 3u;
    unsigned bit_index = (unsigned)(decoder->bit_position & 7u);
    unsigned bit = 0u;
    if (byte_index < decoder->size) {
        bit = (unsigned)((decoder->data[byte_index] >> (7u - bit_index)) & 1u);
    }
    decoder->bit_position++;
    return bit;
}

void *f26_rc64_create(const uint8_t *data, size_t size) {
    f26_rc64_decoder *decoder;
    unsigned bit;
    if (!data || !size) return NULL;
    decoder = (f26_rc64_decoder *)calloc(1u, sizeof(f26_rc64_decoder));
    if (!decoder) return NULL;
    decoder->data = data;
    decoder->size = size;
    decoder->high = F26_TOP;
    for (bit = 0u; bit < 63u; ++bit) {
        decoder->code = (decoder->code << 1u) | f26_read_bit(decoder);
    }
    return decoder;
}

void f26_rc64_destroy(void *opaque) {
    free(opaque);
}

int f26_rc64_get_state(const void *opaque, f26_rc64_state *state) {
    const f26_rc64_decoder *decoder = (const f26_rc64_decoder *)opaque;
    if (!decoder || !state) return -1;
    state->low = decoder->low;
    state->high = decoder->high;
    state->code = decoder->code;
    state->bit_position = (uint64_t)decoder->bit_position;
    state->error = decoder->error;
    return 0;
}

int f26_rc64_set_state(void *opaque, const f26_rc64_state *state) {
    f26_rc64_decoder *decoder = (f26_rc64_decoder *)opaque;
    if (!decoder || !state || state->bit_position < 63u) return -1;
    decoder->low = state->low;
    decoder->high = state->high;
    decoder->code = state->code;
    decoder->bit_position = (size_t)state->bit_position;
    decoder->error = state->error;
    return 0;
}

size_t f26_rc64_bit_position(const void *opaque) {
    const f26_rc64_decoder *decoder = (const f26_rc64_decoder *)opaque;
    return decoder ? decoder->bit_position : 0u;
}

static int f26_decode_row(
    f26_rc64_decoder *decoder,
    const uint32_t *frequencies,
    uint8_t *output
) {
    uint64_t total = 0u;
    uint64_t width = decoder->high - decoder->low + 1u;
    uint64_t scaled;
    uint64_t cumulative_low = 0u;
    uint64_t cumulative_high = 0u;
    uint64_t lower_offset;
    uint64_t upper_offset;
    unsigned symbol;

    for (symbol = 0u; symbol < F26_ALPHABET; ++symbol) {
        if (!frequencies[symbol]) return -1;
        total += frequencies[symbol];
    }
    if (
        total != F26_TOTAL || decoder->code < decoder->low || decoder->code > decoder->high
    ) return -2;
    scaled = (uint64_t)(
        (((__uint128_t)(decoder->code - decoder->low + 1u) * F26_TOTAL) - 1u) / width
    );
    for (symbol = 0u; symbol < F26_ALPHABET; ++symbol) {
        cumulative_high += frequencies[symbol];
        if (scaled < cumulative_high) break;
        cumulative_low = cumulative_high;
    }
    if (symbol == F26_ALPHABET) return -3;
    lower_offset = (uint64_t)(((__uint128_t)width * cumulative_low) >> 31u);
    upper_offset = (uint64_t)(((__uint128_t)width * cumulative_high) >> 31u);
    if (upper_offset <= lower_offset) return -4;
    decoder->high = decoder->low + upper_offset - 1u;
    decoder->low += lower_offset;
    for (;;) {
        if (decoder->high < F26_HALF) {
            /* No offset. */
        } else if (decoder->low >= F26_HALF) {
            decoder->code -= F26_HALF;
            decoder->low -= F26_HALF;
            decoder->high -= F26_HALF;
        } else if (decoder->low >= F26_FIRST_QTR && decoder->high < F26_THIRD_QTR) {
            decoder->code -= F26_FIRST_QTR;
            decoder->low -= F26_FIRST_QTR;
            decoder->high -= F26_FIRST_QTR;
        } else {
            break;
        }
        decoder->low <<= 1u;
        decoder->high = (decoder->high << 1u) | 1u;
        decoder->code = (decoder->code << 1u) | f26_read_bit(decoder);
    }
    *output = (uint8_t)symbol;
    return 0;
}

static int32_t clamp_i32(int64_t value, int32_t low, int32_t high) {
    if (value < low) return low;
    if (value > high) return high;
    return (int32_t)value;
}

/* Round signed numerator / 2^shift to nearest, with ties to even. */
static int64_t round_div_pow2_even(int64_t numerator, unsigned shift) {
    uint64_t magnitude;
    uint64_t quotient;
    uint64_t remainder;
    uint64_t half;
    int negative;
    if (!shift) return numerator;
    negative = numerator < 0;
    magnitude = negative ? (uint64_t)(-numerator) : (uint64_t)numerator;
    quotient = magnitude >> shift;
    remainder = magnitude & (((uint64_t)1u << shift) - 1u);
    half = (uint64_t)1u << (shift - 1u);
    if (remainder > half || (remainder == half && (quotient & 1u))) quotient++;
    return negative ? -(int64_t)quotient : (int64_t)quotient;
}

static int32_t requantize_affine(
    int64_t sum,
    int16_t bias,
    int8_t exponent,
    unsigned shift,
    int32_t low,
    int32_t high
) {
    unsigned exponent_shift;
    int64_t numerator;
    if (exponent > 0 || exponent < -20) return low - 1;
    exponent_shift = (unsigned)(-exponent);
    numerator = sum + ((int64_t)bias << exponent_shift);
    return clamp_i32(
        round_div_pow2_even(numerator, shift + exponent_shift), low, high
    );
}

static int probability_and_frequencies(
    const float corrected[F26_ALPHABET],
    int precision,
    float probability[F26_ALPHABET],
    uint32_t frequency[F26_ALPHABET]
) {
    double values[F26_ALPHABET];
    double maximum;
    double sum = 0.0;
    uint64_t frequency_sum = 0u;
    int winner = 0;
    int symbol;
    int64_t balance;
    int64_t adjusted;

    for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
        float scaled = corrected[symbol] * (float)precision;
        long quantized = lrintf(scaled);
        if (quantized < -32768) quantized = -32768;
        if (quantized > 32767) quantized = 32767;
        values[symbol] = (double)((float)quantized / (float)precision);
    }
    maximum = values[0];
    for (symbol = 1; symbol < F26_ALPHABET; ++symbol) {
        if (values[symbol] > maximum) maximum = values[symbol];
    }
    for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
        values[symbol] = exp(values[symbol] - maximum);
        sum += values[symbol];
    }
    for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
        double value;
        uint64_t item;
        probability[symbol] = (float)(values[symbol] / sum);
        value = (double)probability[symbol];
        if (!isfinite(value) || value <= 0.0 || value > 1.00002) return -1;
        if (probability[symbol] > probability[winner]) winner = symbol;
        item = (uint64_t)(value * (double)F26_TOTAL);
        if (item < 1u) item = 1u;
        frequency[symbol] = (uint32_t)item;
        frequency_sum += item;
    }
    balance = (int64_t)F26_TOTAL - (int64_t)frequency_sum;
    adjusted = (int64_t)frequency[winner] + balance;
    if (adjusted <= 0 || adjusted >= (int64_t)F26_TOTAL) return -2;
    frequency[winner] = (uint32_t)adjusted;
    return 0;
}

int f26_hpac_decode_frame(
    void *opaque_decoder,
    const f26_hpac_model *model,
    const uint8_t *boundary,
    const float *shift,
    const float *past,
    const float *scale,
    const float *spm,
    uint8_t *current,
    float *corrected_trace,
    float *probability_trace
) {
    f26_rc64_decoder *decoder = (f26_rc64_decoder *)opaque_decoder;
    int16_t *hidden = NULL;
    int16_t *b1 = NULL;
    int16_t *logits = NULL;
    size_t hidden_capacity;
    size_t b1_capacity;
    size_t logits_capacity;
    int32_t max_h = 0;
    int32_t max_b1 = 0;
    int32_t max_targets = 0;
    int32_t group;
    int status = 0;

    if (
        !decoder || !model || !boundary || !shift || !past || !scale || !spm ||
        !current || !corrected_trace || !probability_trace
    ) return -1;
    if (
        model->channels != 64 || model->patch_count != model->patch_rows * model->patch_cols ||
        model->height != model->patch_rows * model->patch ||
        model->width != model->patch_cols * model->patch || model->groups <= 0
    ) return -2;
    if (fesetround(FE_TONEAREST)) return -3;

    for (group = 0; group < model->groups; ++group) {
        int32_t h_count = model->group_h_offsets[group + 1] - model->group_h_offsets[group];
        int32_t b1_count = model->group_b1_offsets[group + 1] - model->group_b1_offsets[group];
        int32_t target_count = model->group_target_offsets[group + 1] - model->group_target_offsets[group];
        if (h_count > max_h) max_h = h_count;
        if (b1_count > max_b1) max_b1 = b1_count;
        if (target_count > max_targets) max_targets = target_count;
    }
    hidden_capacity = (size_t)model->patch_count * (size_t)(max_h + 1) * (size_t)model->channels;
    b1_capacity = (size_t)model->patch_count * (size_t)(max_b1 + 1) * (size_t)model->channels;
    logits_capacity = (size_t)model->patch_count * (size_t)max_targets * F26_ALPHABET;
    hidden = (int16_t *)malloc(hidden_capacity * sizeof(int16_t));
    b1 = (int16_t *)malloc(b1_capacity * sizeof(int16_t));
    logits = (int16_t *)malloc(logits_capacity * sizeof(int16_t));
    if (!hidden || !b1 || !logits) {
        status = -4;
        goto cleanup;
    }

    memset(current, 0, (size_t)model->height * (size_t)model->width);
    for (group = 0; group < model->groups; ++group) {
        int32_t h_start = model->group_h_offsets[group];
        int32_t h_count = model->group_h_offsets[group + 1] - h_start;
        int32_t b1_start = model->group_b1_offsets[group];
        int32_t b1_count = model->group_b1_offsets[group + 1] - b1_start;
        int32_t target_start = model->group_target_offsets[group];
        int32_t target_count = model->group_target_offsets[group + 1] - target_start;
        int32_t output_start = model->group_output_offsets[group];
        int32_t output_count = model->group_output_offsets[group + 1] - output_start;
        int32_t patch_index;
        int32_t local_index;

        if (output_count != model->patch_count * target_count) {
            status = -5;
            goto cleanup;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4)
#endif
        for (patch_index = 0; patch_index < model->patch_count; ++patch_index) {
            int32_t patch_row = patch_index / model->patch_cols;
            int32_t patch_col = patch_index % model->patch_cols;
            for (local_index = 0; local_index < h_count; ++local_index) {
                int32_t local_row = model->h_positions[(h_start + local_index) * 2];
                int32_t local_col = model->h_positions[(h_start + local_index) * 2 + 1];
                int32_t accum[64];
                int32_t tap;
                int32_t channel;
                const int32_t *coordinate = model->conv_a_coordinate +
                    (size_t)(h_start + local_index) * (size_t)model->channels;
                for (channel = 0; channel < model->channels; ++channel) accum[channel] = coordinate[channel];
                for (tap = 0; tap < model->a_taps; ++tap) {
                    int32_t row = local_row + model->a_offsets[tap * 2];
                    int32_t col = local_col + model->a_offsets[tap * 2 + 1];
                    int32_t global_row;
                    int32_t global_col;
                    uint8_t class_id;
                    const int8_t *weight;
                    if (row < 0 || row >= model->patch || col < 0 || col >= model->patch) continue;
                    global_row = patch_row * model->patch + row;
                    global_col = patch_col * model->patch + col;
                    class_id = current[global_row * model->width + global_col];
                    if (class_id >= F26_ALPHABET) {
                        status = -6;
                        goto cleanup;
                    }
                    weight = model->conv_a_weight +
                        ((size_t)class_id * (size_t)model->a_taps + (size_t)tap) *
                        (size_t)model->channels;
                    for (channel = 0; channel < model->channels; ++channel) accum[channel] += weight[channel];
                }
                for (channel = 0; channel < model->channels; ++channel) {
                    int32_t value = requantize_affine(
                        accum[channel], model->conv_a_bias[channel],
                        model->conv_a_exponent[channel], 1u, -127, 127
                    );
                    int32_t context_offset =
                        ((patch_index * model->channels + channel) * model->patch + local_row) *
                        model->patch + local_col;
                    int32_t film_offset = patch_index * model->channels + channel;
                    int64_t scaled = (int64_t)value * (16 + (int32_t)lrintf(scale[film_offset]));
                    value = clamp_i32(round_div_pow2_even(scaled, 4u), -127, 127);
                    {
                        int32_t past_value = (int32_t)lrintf(past[context_offset]);
                        past_value += (int32_t)lrintf(spm[context_offset]);
                        /* Match the reference's distinct past+SPM requantization. */
                        past_value = clamp_i32(past_value, -127, 127);
                        value += (int32_t)lrintf(shift[film_offset]);
                        value += past_value;
                    }
                    value = clamp_i32(value, -127, 127);
                    if (value < 0) value = 0;
                    hidden[
                        ((size_t)patch_index * (size_t)(h_count + 1) + (size_t)local_index) *
                        (size_t)model->channels + (size_t)channel
                    ] = (int16_t)value;
                }
            }
            memset(
                hidden + ((size_t)patch_index * (size_t)(h_count + 1) + (size_t)h_count) *
                    (size_t)model->channels,
                0,
                (size_t)model->channels * sizeof(int16_t)
            );

            for (local_index = 0; local_index < b1_count; ++local_index) {
                int32_t channel;
                for (channel = 0; channel < model->channels; ++channel) {
                    int64_t sum = 0;
                    int32_t tap;
                    for (tap = 0; tap < model->b1_taps; ++tap) {
                        int32_t gather = model->b1_gather[
                            ((size_t)b1_start + (size_t)local_index) * (size_t)model->b1_taps + (size_t)tap
                        ];
                        int16_t value = hidden[
                            ((size_t)patch_index * (size_t)(h_count + 1) + (size_t)gather) *
                            (size_t)model->channels + (size_t)channel
                        ];
                        sum += (int64_t)value * model->b1_weight[tap * model->channels + channel];
                    }
                    {
                        int32_t value = requantize_affine(
                            sum, model->b1_bias[channel], model->b1_exponent[channel],
                            3u, -127, 127
                        );
                        if (value < 0) value = 0;
                        b1[
                            ((size_t)patch_index * (size_t)(b1_count + 1) + (size_t)local_index) *
                            (size_t)model->channels + (size_t)channel
                        ] = (int16_t)value;
                    }
                }
            }
            memset(
                b1 + ((size_t)patch_index * (size_t)(b1_count + 1) + (size_t)b1_count) *
                    (size_t)model->channels,
                0,
                (size_t)model->channels * sizeof(int16_t)
            );

            for (local_index = 0; local_index < target_count; ++local_index) {
                int16_t b2_hidden[64];
                int32_t channel;
                int32_t symbol;
                for (channel = 0; channel < model->channels; ++channel) {
                    int64_t sum = 0;
                    int32_t tap;
                    for (tap = 0; tap < model->b2_taps; ++tap) {
                        int32_t gather = model->b2_gather[
                            ((size_t)target_start + (size_t)local_index) * (size_t)model->b2_taps + (size_t)tap
                        ];
                        int16_t value = b1[
                            ((size_t)patch_index * (size_t)(b1_count + 1) + (size_t)gather) *
                            (size_t)model->channels + (size_t)channel
                        ];
                        sum += (int64_t)value * model->b2_weight[tap * model->channels + channel];
                    }
                    {
                        int32_t value = requantize_affine(
                            sum, model->b2_bias[channel], model->b2_exponent[channel],
                            3u, -127, 127
                        );
                        if (value < 0) value = 0;
                        b2_hidden[channel] = (int16_t)value;
                    }
                }
                for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
                    int64_t sum = 0;
                    for (channel = 0; channel < model->channels; ++channel) {
                        sum += (int64_t)b2_hidden[channel] *
                            model->head_weight[symbol * model->channels + channel];
                    }
                    logits[
                        ((size_t)patch_index * (size_t)target_count + (size_t)local_index) *
                        F26_ALPHABET + (size_t)symbol
                    ] = (int16_t)requantize_affine(
                        sum, model->head_bias[symbol], model->head_exponent[symbol],
                        3u, -32768, 32767
                    );
                }
            }
        }

        for (local_index = 0; local_index < output_count; ++local_index) {
            int32_t patch_major = model->output_order[output_start + local_index];
            int32_t flat_position = model->flat_positions[output_start + local_index];
            const int16_t *row = logits + (size_t)patch_major * F26_ALPHABET;
            float corrected[F26_ALPHABET];
            float probability[F26_ALPHABET];
            uint32_t frequency[F26_ALPHABET];
            int32_t predicted = 0;
            int32_t symbol;
            uint8_t decoded;
            int32_t feature;
            for (symbol = 1; symbol < F26_ALPHABET; ++symbol) {
                if (row[symbol] > row[predicted]) predicted = symbol;
            }
            feature = (int32_t)boundary[flat_position] * F26_ALPHABET + predicted;
            if (feature < 0 || feature >= 25) {
                status = -7;
                goto cleanup;
            }
            for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
                corrected[symbol] = (float)row[symbol] / 8.0f +
                    model->residual_table[feature * F26_ALPHABET + symbol];
                corrected_trace[((size_t)output_start + (size_t)local_index) * F26_ALPHABET + symbol] =
                    corrected[symbol];
            }
            status = probability_and_frequencies(
                corrected, model->logit_precision, probability, frequency
            );
            if (status) {
                status = -20 + status;
                goto cleanup;
            }
            for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
                probability_trace[((size_t)output_start + (size_t)local_index) * F26_ALPHABET + symbol] =
                    probability[symbol];
            }
            status = f26_decode_row(decoder, frequency, &decoded);
            if (status) {
                status -= 30;
                goto cleanup;
            }
            current[flat_position] = decoded;
        }
    }

cleanup:
    free(hidden);
    free(b1);
    free(logits);
    return status;
}

uint64_t f26_total_frequency(void) {
    return F26_TOTAL;
}
