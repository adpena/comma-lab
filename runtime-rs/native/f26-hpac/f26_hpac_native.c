#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * ddm_wc2c ISA gating.
 *
 * arm64 makes NEON mandatory, so the presence of the target IS the gate -- there
 * is no arm64 without it and a probe would be theatre.  x86 is the opposite: the
 * contest silicon is unknown, the library may be prebuilt and shipped rather
 * than compiled on the run host, and AVX-512 is absent from most cloud SKUs.  So
 * x86 headers are included unconditionally (they are available regardless of
 * -march) and the kernel is selected by ``__builtin_cpu_supports`` AT RUNTIME.
 * Every kernel below is pure integer arithmetic, where SIMD lanes are exact and
 * addition is associative, so the dispatch cannot change a decoded value.
 */
#if defined(F26_FORCE_SCALAR)
/* The portable twin: no intrinsics at all, the identity oracle for every path. */
#elif defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define F26_HAVE_NEON 1
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define F26_HAVE_X86 1
#endif

#include <pthread.h>

/*
 * ---------------------------------------------------------------------------
 * ddm_wc2c worker pool.
 *
 * WHY NOT OpenMP.  The decoder is loaded into a process that has already loaded
 * PyTorch, which links its own OpenMP runtime.  On macOS that is an immediate
 * hard abort ("multiple copies of the OpenMP runtime ... can silently produce
 * incorrect results"), and the only escape hatch upstream offers is
 * KMP_DUPLICATE_LIB_OK, which is documented as unsafe.  Building against OpenMP
 * also drags two decode-time dependencies into ``inflate.sh`` -- ``brew --prefix
 * libomp`` on Darwin and ``-fopenmp`` on Linux -- either of which failing turns
 * the accelerator off at the exact moment we need it.  pthreads is in libc on
 * both, so the dependency disappears.
 *
 * WHY IT IS DETERMINISTIC.  Every parallel region here partitions PATCHES, and
 * each patch writes only its own slice of hidden / b1 / logits / conv_state.
 * There is no cross-thread reduction anywhere, so the decoded field does not
 * depend on the thread count -- a run with F26_HPAC_THREADS=1 and a run with 8
 * produce byte-identical output, which the identity driver checks rather than
 * assumes.  The float64 corrector, where a reordered reduction WOULD change the
 * result, is not in this file at all.
 * ---------------------------------------------------------------------------
 */

#define F26_MAX_THREADS 32

typedef void (*f26_range_fn)(void *context, int32_t begin, int32_t end);

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    pthread_cond_t done;
    f26_range_fn body;
    void *context;
    int32_t count;
    int32_t slices;
    int32_t pool_slices;
    uint64_t generation;
    int32_t finished;
    int started;
    int shutdown;
    pthread_t threads[F26_MAX_THREADS];
} f26_pool;

static f26_pool f26_worker_pool = {
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER,
    NULL, NULL, 0, 1, 1, 0, 0, 0, 0, {0}
};
static pthread_once_t f26_pool_once = PTHREAD_ONCE_INIT;

/* Contiguous static slice, so slice k always covers the same items. */
static void f26_slice_bounds(
    int32_t count, int32_t slices, int32_t index, int32_t *begin, int32_t *end
) {
    int32_t base = count / slices;
    int32_t extra = count % slices;
    int32_t start = index * base + (index < extra ? index : extra);
    int32_t width = base + (index < extra ? 1 : 0);
    *begin = start;
    *end = start + width;
}

static void *f26_worker_main(void *argument) {
    intptr_t slice = (intptr_t)argument;
    uint64_t seen = 0u;
    for (;;) {
        f26_range_fn body;
        void *context;
        int32_t count;
        int32_t slices;
        int32_t begin;
        int32_t end;

        pthread_mutex_lock(&f26_worker_pool.mutex);
        while (!f26_worker_pool.shutdown && f26_worker_pool.generation == seen) {
            pthread_cond_wait(&f26_worker_pool.ready, &f26_worker_pool.mutex);
        }
        if (f26_worker_pool.shutdown) {
            pthread_mutex_unlock(&f26_worker_pool.mutex);
            return NULL;
        }
        seen = f26_worker_pool.generation;
        body = f26_worker_pool.body;
        context = f26_worker_pool.context;
        count = f26_worker_pool.count;
        slices = f26_worker_pool.slices;
        pthread_mutex_unlock(&f26_worker_pool.mutex);

        if ((int32_t)slice < slices) {
            f26_slice_bounds(count, slices, (int32_t)slice, &begin, &end);
            if (end > begin) body(context, begin, end);
        }

        pthread_mutex_lock(&f26_worker_pool.mutex);
        f26_worker_pool.finished++;
        pthread_cond_signal(&f26_worker_pool.done);
        pthread_mutex_unlock(&f26_worker_pool.mutex);
    }
}

static int32_t f26_requested_threads(void) {
    const char *text = getenv("F26_HPAC_THREADS");
    long value = 4;
    if (text && *text) {
        char *stop = NULL;
        long parsed = strtol(text, &stop, 10);
        if (stop && stop != text && parsed > 0) value = parsed;
    }
    if (value < 1) value = 1;
    if (value > F26_MAX_THREADS) value = F26_MAX_THREADS;
    return (int32_t)value;
}

static void f26_pool_start(void) {
    int32_t threads = f26_requested_threads();
    intptr_t index;
    f26_worker_pool.pool_slices = threads;
    /* The caller runs slice 0 itself, so only threads-1 workers are spawned. */
    for (index = 1; index < (intptr_t)threads; ++index) {
        if (pthread_create(
                &f26_worker_pool.threads[index], NULL, f26_worker_main, (void *)index
            )) {
            /* Fail closed to fewer slices rather than to a wrong answer: the
             * partition is recomputed from the job's `slices`, so a short pool
             * is still a complete, identical traversal. */
            f26_worker_pool.pool_slices = (int32_t)index;
            break;
        }
    }
    f26_worker_pool.started = 1;
}

static void f26_parallel_for(int32_t count, f26_range_fn body, void *context) {
    int32_t slices;
    int32_t begin;
    int32_t end;

    if (count <= 0) return;
    pthread_once(&f26_pool_once, f26_pool_start);
    slices = f26_worker_pool.pool_slices;
    if (slices <= 1) {
        body(context, 0, count);
        return;
    }

    pthread_mutex_lock(&f26_worker_pool.mutex);
    f26_worker_pool.body = body;
    f26_worker_pool.context = context;
    f26_worker_pool.count = count;
    f26_worker_pool.slices = slices;
    f26_worker_pool.finished = 0;
    f26_worker_pool.generation++;
    pthread_cond_broadcast(&f26_worker_pool.ready);
    pthread_mutex_unlock(&f26_worker_pool.mutex);

    f26_slice_bounds(count, slices, 0, &begin, &end);
    if (end > begin) body(context, begin, end);

    pthread_mutex_lock(&f26_worker_pool.mutex);
    while (f26_worker_pool.finished < slices - 1) {
        pthread_cond_wait(&f26_worker_pool.done, &f26_worker_pool.mutex);
    }
    pthread_mutex_unlock(&f26_worker_pool.mutex);
}

int32_t f26_hpac_thread_count(void) {
    pthread_once(&f26_pool_once, f26_pool_start);
    return f26_worker_pool.pool_slices;
}

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

static void f26_select_dispatch(void);

static double f26_last_timing[5];

static double f26_now_seconds(void) {
    struct timespec stamp;
    timespec_get(&stamp, TIME_UTC);
    return (double)stamp.tv_sec + (double)stamp.tv_nsec * 1.0e-9;
}

typedef struct {
    uint64_t low;
    uint64_t high;
    uint64_t code;
    const uint8_t *data;
    size_t size;
    size_t bit_position;
    int error;
    int16_t *hidden_workspace;
    int16_t *b1_workspace;
    int16_t *logits_workspace;
    int16_t *shift_workspace;
    int16_t *past_workspace;
    int16_t *scale_workspace;
    int16_t *spm_workspace;
    int16_t *spm_dw_workspace;
    int32_t *pool_workspace;
    int32_t *conv_state_workspace;
    uint8_t *decoded_workspace;
    int8_t *patch_status_workspace;
    size_t patch_status_capacity;
    size_t hidden_capacity;
    size_t b1_capacity;
    size_t logits_capacity;
    size_t shift_capacity;
    size_t past_capacity;
    size_t scale_capacity;
    size_t spm_capacity;
    size_t spm_dw_capacity;
    size_t pool_capacity;
    size_t conv_state_capacity;
    size_t decoded_capacity;
    /* Group geometry, recomputed per call (see f26_cache_geometry). */
    int32_t max_h;
    int32_t max_b1;
    int32_t max_targets;
    /* Set by f26_hpac_frame_begin; refused by the group entry points until then. */
    int32_t frame_ready;
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
    int32_t num_frames;
    int32_t frame_dim;
    int32_t has_frame_scale;
    int32_t has_spm;

    /* conv_a_weight is [7, a_taps, channels], channel-contiguous. */
    const int8_t *conv_a_weight;
    /* Exact [5, a_taps, channels] class-minus-class-zero deltas. */
    const int16_t *conv_a_delta;
    /* Exact class-zero plus coordinate accumulator per [patch position, channel]. */
    const int32_t *conv_a_initial;
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

    const int8_t *frame_codes;
    const int8_t *frame_shift_weight;
    const int16_t *frame_shift_bias;
    const int8_t *frame_shift_exponent;
    const int8_t *frame_scale_weight;
    const int16_t *frame_scale_bias;
    const int8_t *frame_scale_exponent;
    /* conv_past_weight is [3, 3, 5, channels], channel-contiguous. */
    const int8_t *conv_past_weight;
    const int16_t *conv_past_bias;
    const int8_t *conv_past_exponent;
    /* spm_dw_weight is [3, 3, channels], channel-contiguous. */
    const int8_t *spm_dw_weight;
    const int16_t *spm_dw_bias;
    const int8_t *spm_dw_exponent;
    /* spm_pw_weight is [channels, channels], output-major. */
    const int8_t *spm_pw_weight;
    const int16_t *spm_pw_bias;
    const int8_t *spm_pw_exponent;

    const int16_t *a_offsets;
    const int32_t *group_h_offsets;
    const int32_t *group_b1_offsets;
    const int32_t *group_target_offsets;
    const int32_t *group_output_offsets;
    const int16_t *h_positions;
    const int16_t *b1_gather;
    const int16_t *b2_gather;
    const int16_t *target_positions;
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
    f26_select_dispatch();
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
    f26_rc64_decoder *decoder = (f26_rc64_decoder *)opaque;
    if (!decoder) return;
    free(decoder->hidden_workspace);
    free(decoder->b1_workspace);
    free(decoder->logits_workspace);
    free(decoder->shift_workspace);
    free(decoder->past_workspace);
    free(decoder->scale_workspace);
    free(decoder->spm_workspace);
    free(decoder->spm_dw_workspace);
    free(decoder->pool_workspace);
    free(decoder->conv_state_workspace);
    free(decoder->decoded_workspace);
    free(decoder->patch_status_workspace);
    free(decoder);
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

/*
 * ---------------------------------------------------------------------------
 * Integer channel kernels.
 *
 * All three operate on the fixed 64-channel width and are EXACT: int8/int16
 * inputs whose products and 64-term sums are bounded far inside int32
 * (hidden and b1 activations are clamped to [0, 127] and weights to int8, so a
 * 64-tap accumulation cannot exceed 64 * 127 * 128 = 1,040,384).  Integer
 * addition is associative, so lane order and reduction order are free here --
 * unlike the float64 corrector, where they are not.  That asymmetry is the whole
 * reason the corrector stayed in numpy and only this half was lowered.
 * ---------------------------------------------------------------------------
 */

#define F26_CHANNELS_FIXED 64

static void add_i8x64_scalar(int32_t destination[64], const int8_t source[64]) {
    int channel;
    for (channel = 0; channel < F26_CHANNELS_FIXED; ++channel) {
        destination[channel] += (int32_t)source[channel];
    }
}

static void add_i16x64_scalar(int32_t destination[64], const int16_t source[64]) {
    int channel;
    for (channel = 0; channel < F26_CHANNELS_FIXED; ++channel) {
        destination[channel] += (int32_t)source[channel];
    }
}

/* sums[c] += values[c] * weights[c], 64 channels, int16 x int8 -> int32. */
static void mac_i16_i8x64_scalar(
    int32_t sums[64], const int16_t *values, const int8_t *weights
) {
    int channel;
    for (channel = 0; channel < F26_CHANNELS_FIXED; ++channel) {
        sums[channel] += (int32_t)values[channel] * (int32_t)weights[channel];
    }
}

/* Horizontal dot product over the 64 channels. */
static int32_t dot_i16_i8x64_scalar(const int16_t *values, const int8_t *weights) {
    int32_t total = 0;
    int channel;
    for (channel = 0; channel < F26_CHANNELS_FIXED; ++channel) {
        total += (int32_t)values[channel] * (int32_t)weights[channel];
    }
    return total;
}

#if defined(F26_HAVE_NEON)
static void add_i8x64_neon(int32_t destination[64], const int8_t source[64]) {
    int channel;
    for (channel = 0; channel < F26_CHANNELS_FIXED; channel += 16) {
        int8x16_t packed = vld1q_s8(source + channel);
        int16x8_t low16 = vmovl_s8(vget_low_s8(packed));
        int16x8_t high16 = vmovl_s8(vget_high_s8(packed));
        int32x4_t values[4] = {
            vmovl_s16(vget_low_s16(low16)),
            vmovl_s16(vget_high_s16(low16)),
            vmovl_s16(vget_low_s16(high16)),
            vmovl_s16(vget_high_s16(high16)),
        };
        int lane;
        for (lane = 0; lane < 4; ++lane) {
            int32_t *target = destination + channel + lane * 4;
            vst1q_s32(target, vaddq_s32(vld1q_s32(target), values[lane]));
        }
    }
}

static void add_i16x64_neon(int32_t destination[64], const int16_t source[64]) {
    int channel;
    for (channel = 0; channel < F26_CHANNELS_FIXED; channel += 8) {
        int16x8_t packed = vld1q_s16(source + channel);
        int32x4_t low32 = vmovl_s16(vget_low_s16(packed));
        int32x4_t high32 = vmovl_s16(vget_high_s16(packed));
        vst1q_s32(destination + channel, vaddq_s32(vld1q_s32(destination + channel), low32));
        vst1q_s32(
            destination + channel + 4, vaddq_s32(vld1q_s32(destination + channel + 4), high32)
        );
    }
}

static void mac_i16_i8x64_neon(
    int32_t sums[64], const int16_t *values, const int8_t *weights
) {
    int channel;
    for (channel = 0; channel < F26_CHANNELS_FIXED; channel += 16) {
        int8x16_t packed = vld1q_s8(weights + channel);
        int16x8_t weight_low = vmovl_s8(vget_low_s8(packed));
        int16x8_t weight_high = vmovl_s8(vget_high_s8(packed));
        int16x8_t value_low = vld1q_s16(values + channel);
        int16x8_t value_high = vld1q_s16(values + channel + 8);
        int32x4_t acc0 = vld1q_s32(sums + channel);
        int32x4_t acc1 = vld1q_s32(sums + channel + 4);
        int32x4_t acc2 = vld1q_s32(sums + channel + 8);
        int32x4_t acc3 = vld1q_s32(sums + channel + 12);
        acc0 = vmlal_s16(acc0, vget_low_s16(value_low), vget_low_s16(weight_low));
        acc1 = vmlal_s16(acc1, vget_high_s16(value_low), vget_high_s16(weight_low));
        acc2 = vmlal_s16(acc2, vget_low_s16(value_high), vget_low_s16(weight_high));
        acc3 = vmlal_s16(acc3, vget_high_s16(value_high), vget_high_s16(weight_high));
        vst1q_s32(sums + channel, acc0);
        vst1q_s32(sums + channel + 4, acc1);
        vst1q_s32(sums + channel + 8, acc2);
        vst1q_s32(sums + channel + 12, acc3);
    }
}

static int32_t dot_i16_i8x64_neon(const int16_t *values, const int8_t *weights) {
    int32x4_t acc = vdupq_n_s32(0);
    int channel;
    for (channel = 0; channel < F26_CHANNELS_FIXED; channel += 16) {
        int8x16_t packed = vld1q_s8(weights + channel);
        int16x8_t weight_low = vmovl_s8(vget_low_s8(packed));
        int16x8_t weight_high = vmovl_s8(vget_high_s8(packed));
        int16x8_t value_low = vld1q_s16(values + channel);
        int16x8_t value_high = vld1q_s16(values + channel + 8);
        acc = vmlal_s16(acc, vget_low_s16(value_low), vget_low_s16(weight_low));
        acc = vmlal_s16(acc, vget_high_s16(value_low), vget_high_s16(weight_low));
        acc = vmlal_s16(acc, vget_low_s16(value_high), vget_low_s16(weight_high));
        acc = vmlal_s16(acc, vget_high_s16(value_high), vget_high_s16(weight_high));
    }
    return vaddvq_s32(acc);
}
#endif /* F26_HAVE_NEON */

#if defined(F26_HAVE_X86)
__attribute__((target("avx2")))
static void add_i8x64_avx2(int32_t destination[64], const int8_t source[64]) {
    int channel;
    for (channel = 0; channel < F26_CHANNELS_FIXED; channel += 8) {
        __m128i packed = _mm_loadl_epi64((const __m128i *)(source + channel));
        __m256i values = _mm256_cvtepi8_epi32(packed);
        __m256i prior = _mm256_loadu_si256((const __m256i *)(destination + channel));
        _mm256_storeu_si256((__m256i *)(destination + channel), _mm256_add_epi32(prior, values));
    }
}

__attribute__((target("avx2")))
static void add_i16x64_avx2(int32_t destination[64], const int16_t source[64]) {
    int channel;
    for (channel = 0; channel < F26_CHANNELS_FIXED; channel += 8) {
        __m128i packed = _mm_loadu_si128((const __m128i *)(source + channel));
        __m256i values = _mm256_cvtepi16_epi32(packed);
        __m256i prior = _mm256_loadu_si256((const __m256i *)(destination + channel));
        _mm256_storeu_si256((__m256i *)(destination + channel), _mm256_add_epi32(prior, values));
    }
}

__attribute__((target("avx2")))
static void mac_i16_i8x64_avx2(
    int32_t sums[64], const int16_t *values, const int8_t *weights
) {
    int channel;
    for (channel = 0; channel < F26_CHANNELS_FIXED; channel += 16) {
        /* Widen both operands to int16 lanes, then madd into int32 pairs.
         * _mm256_madd_epi16 sums adjacent pairs, so multiply against a
         * zero-interleaved partner would fold two channels together.  Instead
         * widen to int32 and use mullo, which keeps one channel per lane. */
        __m128i weight_packed = _mm_loadu_si128((const __m128i *)(weights + channel));
        __m256i weight_low = _mm256_cvtepi8_epi32(weight_packed);
        __m256i weight_high =
            _mm256_cvtepi8_epi32(_mm_srli_si128(weight_packed, 8));
        __m256i value_low =
            _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)(values + channel)));
        __m256i value_high =
            _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)(values + channel + 8)));
        __m256i acc_low = _mm256_loadu_si256((const __m256i *)(sums + channel));
        __m256i acc_high = _mm256_loadu_si256((const __m256i *)(sums + channel + 8));
        acc_low = _mm256_add_epi32(acc_low, _mm256_mullo_epi32(value_low, weight_low));
        acc_high = _mm256_add_epi32(acc_high, _mm256_mullo_epi32(value_high, weight_high));
        _mm256_storeu_si256((__m256i *)(sums + channel), acc_low);
        _mm256_storeu_si256((__m256i *)(sums + channel + 8), acc_high);
    }
}

__attribute__((target("avx2")))
static int32_t dot_i16_i8x64_avx2(const int16_t *values, const int8_t *weights) {
    __m256i acc = _mm256_setzero_si256();
    __m128i low128;
    int32_t lanes[8];
    int channel;
    int lane;
    int32_t total = 0;
    for (channel = 0; channel < F26_CHANNELS_FIXED; channel += 16) {
        __m128i weight_packed = _mm_loadu_si128((const __m128i *)(weights + channel));
        __m256i weight_low = _mm256_cvtepi8_epi32(weight_packed);
        __m256i weight_high =
            _mm256_cvtepi8_epi32(_mm_srli_si128(weight_packed, 8));
        __m256i value_low =
            _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)(values + channel)));
        __m256i value_high =
            _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)(values + channel + 8)));
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(value_low, weight_low));
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(value_high, weight_high));
    }
    low128 = _mm_add_epi32(
        _mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1)
    );
    _mm_storeu_si128((__m128i *)lanes, low128);
    for (lane = 0; lane < 4; ++lane) total += lanes[lane];
    return total;
}
#endif /* F26_HAVE_X86 */

/*
 * Runtime dispatch.  On x86 the kernel is chosen by cpuid via
 * ``__builtin_cpu_supports``; a prebuilt library shipped to unknown silicon
 * therefore still selects a legal path instead of faulting.  On arm64 NEON is
 * architectural.  The scalar row is the fail-closed floor and is also what
 * ``-DF26_FORCE_SCALAR=1`` pins for the identity twin.
 */
static void (*f26_add_i8x64)(int32_t[64], const int8_t[64]) = add_i8x64_scalar;
static void (*f26_add_i16x64)(int32_t[64], const int16_t[64]) = add_i16x64_scalar;
static void (*f26_mac_i16_i8x64)(int32_t[64], const int16_t *, const int8_t *) =
    mac_i16_i8x64_scalar;
static int32_t (*f26_dot_i16_i8x64)(const int16_t *, const int8_t *) = dot_i16_i8x64_scalar;
static const char *f26_dispatch_name = "scalar";
static int f26_dispatch_ready = 0;

static void f26_select_dispatch(void) {
    if (f26_dispatch_ready) return;
#if defined(F26_HAVE_NEON)
    f26_add_i8x64 = add_i8x64_neon;
    f26_add_i16x64 = add_i16x64_neon;
    f26_mac_i16_i8x64 = mac_i16_i8x64_neon;
    f26_dot_i16_i8x64 = dot_i16_i8x64_neon;
    f26_dispatch_name = "neon";
#elif defined(F26_HAVE_X86)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2")) {
        f26_add_i8x64 = add_i8x64_avx2;
        f26_add_i16x64 = add_i16x64_avx2;
        f26_mac_i16_i8x64 = mac_i16_i8x64_avx2;
        f26_dot_i16_i8x64 = dot_i16_i8x64_avx2;
        f26_dispatch_name = "avx2";
    } else {
        f26_dispatch_name = "x86-scalar";
    }
#endif
    f26_dispatch_ready = 1;
}

const char *f26_hpac_dispatch_path(void) {
    f26_select_dispatch();
    return f26_dispatch_name;
}

static void add_i8x64_to_i32(int32_t destination[64], const int8_t source[64]) {
    f26_add_i8x64(destination, source);
}

typedef struct {
    const f26_hpac_model *model;
    const uint8_t *previous;
    int16_t *past;
    int16_t *spm;
    int16_t *spm_dw;
    int32_t *pooled;
} f26_context_job;

static void f26_context_past_range(void *context, int32_t patch_begin, int32_t patch_end);
static void f26_context_spm_range(void *context, int32_t patch_begin, int32_t patch_end);

static int f26_prepare_frame_context(
    f26_rc64_decoder *decoder,
    const f26_hpac_model *model,
    int32_t frame_index,
    const uint8_t *previous
) {
    int16_t *shift = decoder->shift_workspace;
    int16_t *past = decoder->past_workspace;
    int16_t *scale = decoder->scale_workspace;
    int16_t *spm = decoder->spm_workspace;
    int16_t *spm_dw = decoder->spm_dw_workspace;
    int32_t *pooled = decoder->pool_workspace;
    int32_t patch_index;
    f26_context_job context_job;

    if (
        frame_index < 0 || frame_index >= model->num_frames || model->frame_dim <= 0 ||
        model->patch != 64 || model->channels != 64 || !previous || !model->frame_codes ||
        !model->frame_shift_weight || !model->frame_shift_bias ||
        !model->frame_shift_exponent || !model->conv_past_weight ||
        !model->conv_past_bias || !model->conv_past_exponent
    ) return -1;
    if (
        model->has_frame_scale &&
        (!model->frame_scale_weight || !model->frame_scale_bias || !model->frame_scale_exponent)
    ) return -2;
    if (
        model->has_spm &&
        (!model->spm_dw_weight || !model->spm_dw_bias || !model->spm_dw_exponent ||
         !model->spm_pw_weight || !model->spm_pw_bias || !model->spm_pw_exponent)
    ) return -3;

    {
        const int8_t *embedding = model->frame_codes +
            (size_t)frame_index * (size_t)model->frame_dim;
        int16_t frame_shift[64];
        int16_t frame_scale[64];
        int32_t channel;
        for (channel = 0; channel < model->channels; ++channel) {
            int64_t shift_sum = 0;
            int64_t scale_sum = 0;
            int32_t feature;
            for (feature = 0; feature < model->frame_dim; ++feature) {
                shift_sum += (int64_t)embedding[feature] *
                    model->frame_shift_weight[channel * model->frame_dim + feature];
                if (model->has_frame_scale) {
                    scale_sum += (int64_t)embedding[feature] *
                        model->frame_scale_weight[channel * model->frame_dim + feature];
                }
            }
            frame_shift[channel] = (int16_t)requantize_affine(
                shift_sum, model->frame_shift_bias[channel],
                model->frame_shift_exponent[channel], 1u, -127, 127
            );
            frame_scale[channel] = model->has_frame_scale
                ? (int16_t)requantize_affine(
                    scale_sum, model->frame_scale_bias[channel],
                    model->frame_scale_exponent[channel], 4u, -8, 8
                )
                : 0;
        }
        for (patch_index = 0; patch_index < model->patch_count; ++patch_index) {
            memcpy(
                shift + (size_t)patch_index * (size_t)model->channels,
                frame_shift,
                (size_t)model->channels * sizeof(int16_t)
            );
            memcpy(
                scale + (size_t)patch_index * (size_t)model->channels,
                frame_scale,
                (size_t)model->channels * sizeof(int16_t)
            );
        }
    }

    context_job.model = model;
    context_job.previous = previous;
    context_job.past = past;
    context_job.spm = spm;
    context_job.spm_dw = spm_dw;
    context_job.pooled = pooled;
    f26_parallel_for(model->patch_count, f26_context_past_range, &context_job);

    if (!model->has_spm) {
        memset(spm, 0, (size_t)model->patch_count * (size_t)model->channels * sizeof(int16_t));
        return 0;
    }
    f26_parallel_for(model->patch_count, f26_context_spm_range, &context_job);
    return 0;
}

static void f26_context_past_range(void *context, int32_t patch_begin, int32_t patch_end) {
    f26_context_job *job = (f26_context_job *)context;
    const f26_hpac_model *model = job->model;
    const uint8_t *previous = job->previous;
    int16_t *past = job->past;
    int32_t *pooled = job->pooled;
    int32_t patch_index;
    for (patch_index = patch_begin; patch_index < patch_end; ++patch_index) {
        int32_t patch_row = patch_index / model->patch_cols;
        int32_t patch_col = patch_index % model->patch_cols;
        int32_t pool_sum[64] = {0};
        int32_t local_row;
        for (local_row = 0; local_row < model->patch; ++local_row) {
            int32_t local_col;
            int32_t global_row = patch_row * model->patch + local_row;
            for (local_col = 0; local_col < model->patch; ++local_col) {
                int32_t accum[64] = {0};
                int32_t global_col = patch_col * model->patch + local_col;
                int32_t kernel_row;
                size_t destination_offset =
                    (((size_t)patch_index * (size_t)model->patch + (size_t)local_row) *
                     (size_t)model->patch + (size_t)local_col) * (size_t)model->channels;
                for (kernel_row = 0; kernel_row < 3; ++kernel_row) {
                    int32_t source_row = global_row + kernel_row - 1;
                    int32_t kernel_col;
                    if (source_row < 0 || source_row >= model->height) continue;
                    for (kernel_col = 0; kernel_col < 3; ++kernel_col) {
                        int32_t source_col = global_col + kernel_col - 1;
                        uint8_t class_id;
                        const int8_t *weights;
                        if (source_col < 0 || source_col >= model->width) continue;
                        class_id = previous[
                            (size_t)source_row * (size_t)model->width + (size_t)source_col
                        ];
                        if (class_id >= F26_ALPHABET) continue;
                        weights = model->conv_past_weight +
                            ((((size_t)kernel_row * 3u + (size_t)kernel_col) * F26_ALPHABET +
                              (size_t)class_id) * (size_t)model->channels);
                        add_i8x64_to_i32(accum, weights);
                    }
                }
                {
                    int32_t channel;
                    for (channel = 0; channel < model->channels; ++channel) {
                        int32_t value = requantize_affine(
                            accum[channel], model->conv_past_bias[channel],
                            model->conv_past_exponent[channel], 0u, -127, 127
                        );
                        past[destination_offset + (size_t)channel] = (int16_t)value;
                        pool_sum[channel] += value;
                    }
                }
            }
        }
        {
            int32_t channel;
            for (channel = 0; channel < model->channels; ++channel) {
                pooled[(size_t)patch_index * (size_t)model->channels + (size_t)channel] =
                    (int32_t)round_div_pow2_even(pool_sum[channel], 12u);
            }
        }
    }
}

static void f26_context_spm_range(void *context, int32_t patch_begin, int32_t patch_end) {
    f26_context_job *job = (f26_context_job *)context;
    const f26_hpac_model *model = job->model;
    int16_t *spm = job->spm;
    int16_t *spm_dw = job->spm_dw;
    const int32_t *pooled = job->pooled;
    int32_t patch_index;
    for (patch_index = patch_begin; patch_index < patch_end; ++patch_index) {
        int32_t patch_row = patch_index / model->patch_cols;
        int32_t patch_col = patch_index % model->patch_cols;
        int32_t channel;
        for (channel = 0; channel < model->channels; ++channel) {
            int64_t sum = 0;
            int32_t kernel_row;
            for (kernel_row = 0; kernel_row < 3; ++kernel_row) {
                int32_t source_row = patch_row + kernel_row - 1;
                int32_t kernel_col;
                if (source_row < 0 || source_row >= model->patch_rows) continue;
                for (kernel_col = 0; kernel_col < 3; ++kernel_col) {
                    int32_t source_col = patch_col + kernel_col - 1;
                    int32_t source_patch;
                    if (source_col < 0 || source_col >= model->patch_cols) continue;
                    source_patch = source_row * model->patch_cols + source_col;
                    sum += (int64_t)pooled[
                        (size_t)source_patch * (size_t)model->channels + (size_t)channel
                    ] * model->spm_dw_weight[
                        ((size_t)kernel_row * 3u + (size_t)kernel_col) *
                        (size_t)model->channels + (size_t)channel
                    ];
                }
            }
            {
                int32_t value = requantize_affine(
                    sum, model->spm_dw_bias[channel], model->spm_dw_exponent[channel],
                    3u, -127, 127
                );
                if (value < 0) value = 0;
                spm_dw[(size_t)patch_index * (size_t)model->channels + (size_t)channel] =
                    (int16_t)value;
            }
        }
        for (channel = 0; channel < model->channels; ++channel) {
            int64_t sum = 0;
            int32_t input_channel;
            const int8_t *weights = model->spm_pw_weight +
                (size_t)channel * (size_t)model->channels;
            for (input_channel = 0; input_channel < model->channels; ++input_channel) {
                sum += (int64_t)spm_dw[
                    (size_t)patch_index * (size_t)model->channels + (size_t)input_channel
                ] * weights[input_channel];
            }
            spm[(size_t)patch_index * (size_t)model->channels + (size_t)channel] =
                (int16_t)requantize_affine(
                    sum, model->spm_pw_bias[channel], model->spm_pw_exponent[channel],
                    4u, -127, 127
                );
        }
    }
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

/*
 * ---------------------------------------------------------------------------
 * ddm_wc2c: shared frame/group kernels.
 *
 * The fused ``f26_hpac_decode_frame`` path and the split
 * ``f26_hpac_frame_begin`` / ``f26_hpac_group_logits`` / ``f26_hpac_group_commit``
 * path both call the statics below.  There is ONE implementation of the
 * arithmetic, not two that happen to agree, so identity between the two paths is
 * structural rather than coincidental.  This is deliberate: the split path
 * exists so the float64 probability corrector can stay in its audited numpy
 * form while the integer model work is lowered, and a second copy of the integer
 * kernels would reintroduce exactly the drift the split is meant to avoid.
 * ---------------------------------------------------------------------------
 */

static int f26_check_model(const f26_hpac_model *model) {
    if (
        model->channels != 64 || model->patch_count != model->patch_rows * model->patch_cols ||
        model->height != model->patch_rows * model->patch ||
        model->width != model->patch_cols * model->patch || model->groups <= 0 ||
        !model->conv_a_delta
    ) return -2;
    return 0;
}

/* Cache the per-group maxima once; every entry point needs them for capacity. */
static void f26_cache_geometry(f26_rc64_decoder *decoder, const f26_hpac_model *model) {
    int32_t group;
    int32_t max_h = 0;
    int32_t max_b1 = 0;
    int32_t max_targets = 0;
    /* Always recomputed, never cached across calls: a stale maximum would size
     * decoded_workspace for the wrong model and let group_commit write past its
     * end.  190 integer comparisons per frame is not worth that risk. */
    for (group = 0; group < model->groups; ++group) {
        int32_t h_count = model->group_h_offsets[group + 1] - model->group_h_offsets[group];
        int32_t b1_count = model->group_b1_offsets[group + 1] - model->group_b1_offsets[group];
        int32_t target_count =
            model->group_target_offsets[group + 1] - model->group_target_offsets[group];
        if (h_count > max_h) max_h = h_count;
        if (b1_count > max_b1) max_b1 = b1_count;
        if (target_count > max_targets) max_targets = target_count;
    }
    decoder->max_h = max_h;
    decoder->max_b1 = max_b1;
    decoder->max_targets = max_targets;
}

static int f26_ensure_workspaces(f26_rc64_decoder *decoder, const f26_hpac_model *model) {
    size_t hidden_capacity;
    size_t b1_capacity;
    size_t logits_capacity;
    size_t compact_context_capacity;
    size_t past_capacity;
    size_t conv_state_capacity;
    size_t decoded_capacity;

    f26_cache_geometry(decoder, model);
    hidden_capacity =
        (size_t)model->patch_count * (size_t)(decoder->max_h + 1) * (size_t)model->channels;
    b1_capacity =
        (size_t)model->patch_count * (size_t)(decoder->max_b1 + 1) * (size_t)model->channels;
    logits_capacity =
        (size_t)model->patch_count * (size_t)decoder->max_targets * F26_ALPHABET;
    compact_context_capacity = (size_t)model->patch_count * (size_t)model->channels;
    past_capacity = (size_t)model->patch_count * (size_t)model->patch *
        (size_t)model->patch * (size_t)model->channels;
    conv_state_capacity = (size_t)model->patch_count * (size_t)model->patch *
        (size_t)model->patch * (size_t)model->channels;
    decoded_capacity = (size_t)model->patch_count * (size_t)decoder->max_targets;

    if (!decoder->hidden_workspace) {
        decoder->hidden_workspace = (int16_t *)malloc(hidden_capacity * sizeof(int16_t));
        decoder->b1_workspace = (int16_t *)malloc(b1_capacity * sizeof(int16_t));
        decoder->logits_workspace = (int16_t *)malloc(logits_capacity * sizeof(int16_t));
        decoder->shift_workspace = (int16_t *)malloc(compact_context_capacity * sizeof(int16_t));
        decoder->past_workspace = (int16_t *)malloc(past_capacity * sizeof(int16_t));
        decoder->scale_workspace = (int16_t *)malloc(compact_context_capacity * sizeof(int16_t));
        decoder->spm_workspace = (int16_t *)malloc(compact_context_capacity * sizeof(int16_t));
        decoder->spm_dw_workspace = (int16_t *)malloc(
            compact_context_capacity * sizeof(int16_t)
        );
        decoder->pool_workspace = (int32_t *)malloc(compact_context_capacity * sizeof(int32_t));
        decoder->conv_state_workspace = (int32_t *)malloc(conv_state_capacity * sizeof(int32_t));
        decoder->decoded_workspace = (uint8_t *)malloc(decoded_capacity);
        decoder->patch_status_workspace = (int8_t *)malloc((size_t)model->patch_count);
        decoder->patch_status_capacity = (size_t)model->patch_count;
        decoder->hidden_capacity = hidden_capacity;
        decoder->b1_capacity = b1_capacity;
        decoder->logits_capacity = logits_capacity;
        decoder->shift_capacity = compact_context_capacity;
        decoder->past_capacity = past_capacity;
        decoder->scale_capacity = compact_context_capacity;
        decoder->spm_capacity = compact_context_capacity;
        decoder->spm_dw_capacity = compact_context_capacity;
        decoder->pool_capacity = compact_context_capacity;
        decoder->conv_state_capacity = conv_state_capacity;
        decoder->decoded_capacity = decoded_capacity;
    }
    if (
        !decoder->hidden_workspace || !decoder->b1_workspace || !decoder->logits_workspace ||
        !decoder->shift_workspace || !decoder->past_workspace || !decoder->scale_workspace ||
        !decoder->spm_workspace || !decoder->spm_dw_workspace || !decoder->pool_workspace ||
        !decoder->conv_state_workspace || !decoder->decoded_workspace ||
        !decoder->patch_status_workspace ||
        decoder->patch_status_capacity < (size_t)model->patch_count ||
        decoder->hidden_capacity < hidden_capacity || decoder->b1_capacity < b1_capacity ||
        decoder->logits_capacity < logits_capacity ||
        decoder->shift_capacity < compact_context_capacity ||
        decoder->past_capacity < past_capacity ||
        decoder->scale_capacity < compact_context_capacity ||
        decoder->spm_capacity < compact_context_capacity ||
        decoder->spm_dw_capacity < compact_context_capacity ||
        decoder->pool_capacity < compact_context_capacity ||
        decoder->conv_state_capacity < conv_state_capacity ||
        decoder->decoded_capacity < decoded_capacity
    ) return -4;
    return 0;
}

/* Reset the incremental conv-A accumulator to its class-zero + coordinate base. */
static void f26_reset_conv_state(f26_rc64_decoder *decoder, const f26_hpac_model *model) {
    int32_t patch_index;
    size_t stride = (size_t)model->patch * (size_t)model->patch * (size_t)model->channels;
    for (patch_index = 0; patch_index < model->patch_count; ++patch_index) {
        memcpy(
            decoder->conv_state_workspace + (size_t)patch_index * stride,
            model->conv_a_initial,
            stride * sizeof(int32_t)
        );
    }
}

/* One group's integer model evaluation: conv-A state -> hidden -> b1 -> b2 -> logits. */
typedef struct {
    f26_rc64_decoder *decoder;
    const f26_hpac_model *model;
    int32_t h_start;
    int32_t h_count;
    int32_t b1_start;
    int32_t b1_count;
    int32_t target_start;
    int32_t target_count;
} f26_model_job;

static void f26_group_model_range(void *context, int32_t patch_begin, int32_t patch_end) {
    f26_model_job *job = (f26_model_job *)context;
    f26_rc64_decoder *decoder = job->decoder;
    const f26_hpac_model *model = job->model;
    int16_t *hidden = decoder->hidden_workspace;
    int16_t *b1 = decoder->b1_workspace;
    int16_t *logits = decoder->logits_workspace;
    int16_t *shift = decoder->shift_workspace;
    int16_t *past = decoder->past_workspace;
    int16_t *scale = decoder->scale_workspace;
    int16_t *spm = decoder->spm_workspace;
    const int32_t *conv_state = decoder->conv_state_workspace;
    int32_t h_start = job->h_start;
    int32_t h_count = job->h_count;
    int32_t b1_start = job->b1_start;
    int32_t b1_count = job->b1_count;
    int32_t target_start = job->target_start;
    int32_t target_count = job->target_count;
    int32_t patch_index;
    int32_t local_index;

    for (patch_index = patch_begin; patch_index < patch_end; ++patch_index) {
        for (local_index = 0; local_index < h_count; ++local_index) {
            int32_t local_row = model->h_positions[(h_start + local_index) * 2];
            int32_t local_col = model->h_positions[(h_start + local_index) * 2 + 1];
            int32_t channel;
            const int32_t *accum = conv_state +
                (((size_t)patch_index * (size_t)model->patch * (size_t)model->patch +
                  (size_t)local_row * (size_t)model->patch + (size_t)local_col) *
                 (size_t)model->channels);
            for (channel = 0; channel < model->channels; ++channel) {
                int32_t value = requantize_affine(
                    accum[channel], model->conv_a_bias[channel],
                    model->conv_a_exponent[channel], 1u, -127, 127
                );
                int32_t context_offset =
                    (((patch_index * model->patch + local_row) * model->patch + local_col) *
                     model->channels + channel);
                int32_t film_offset = patch_index * model->channels + channel;
                int64_t scaled = (int64_t)value * (16 + (int32_t)scale[film_offset]);
                value = clamp_i32(round_div_pow2_even(scaled, 4u), -127, 127);
                {
                    int32_t past_value = (int32_t)past[context_offset];
                    past_value += (int32_t)spm[film_offset];
                    /* Match the reference's distinct past+SPM requantization. */
                    past_value = clamp_i32(past_value, -127, 127);
                    value += (int32_t)shift[film_offset];
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
            int32_t sums[64];
            int32_t channel;
            int32_t tap;
            memset(sums, 0, sizeof(sums));
            for (tap = 0; tap < model->b1_taps; ++tap) {
                int32_t gather = model->b1_gather[
                    ((size_t)b1_start + (size_t)local_index) * (size_t)model->b1_taps +
                    (size_t)tap
                ];
                const int16_t *values = hidden +
                    ((size_t)patch_index * (size_t)(h_count + 1) + (size_t)gather) *
                    (size_t)model->channels;
                const int8_t *weights = model->b1_weight +
                    (size_t)tap * (size_t)model->channels;
                f26_mac_i16_i8x64(sums, values, weights);
            }
            for (channel = 0; channel < model->channels; ++channel) {
                {
                    int32_t value = requantize_affine(
                        sums[channel], model->b1_bias[channel], model->b1_exponent[channel],
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
            int32_t sums[64];
            int32_t channel;
            int32_t symbol;
            int32_t tap;
            memset(sums, 0, sizeof(sums));
            for (tap = 0; tap < model->b2_taps; ++tap) {
                int32_t gather = model->b2_gather[
                    ((size_t)target_start + (size_t)local_index) * (size_t)model->b2_taps +
                    (size_t)tap
                ];
                const int16_t *values = b1 +
                    ((size_t)patch_index * (size_t)(b1_count + 1) + (size_t)gather) *
                    (size_t)model->channels;
                const int8_t *weights = model->b2_weight +
                    (size_t)tap * (size_t)model->channels;
                f26_mac_i16_i8x64(sums, values, weights);
            }
            for (channel = 0; channel < model->channels; ++channel) {
                {
                    int32_t value = requantize_affine(
                        sums[channel], model->b2_bias[channel], model->b2_exponent[channel],
                        3u, -127, 127
                    );
                    if (value < 0) value = 0;
                    b2_hidden[channel] = (int16_t)value;
                }
            }
            for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
                int64_t sum = (int64_t)f26_dot_i16_i8x64(
                    b2_hidden, model->head_weight + symbol * model->channels
                );
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
}

static int f26_group_model(
    f26_rc64_decoder *decoder,
    const f26_hpac_model *model,
    int32_t group
) {
    f26_model_job job;
    job.decoder = decoder;
    job.model = model;
    job.h_start = model->group_h_offsets[group];
    job.h_count = model->group_h_offsets[group + 1] - job.h_start;
    job.b1_start = model->group_b1_offsets[group];
    job.b1_count = model->group_b1_offsets[group + 1] - job.b1_start;
    job.target_start = model->group_target_offsets[group];
    job.target_count = model->group_target_offsets[group + 1] - job.target_start;
    f26_parallel_for(model->patch_count, f26_group_model_range, &job);
    return 0;
}

/*
 * One output row: argmax of the int16 logits, then the RCF1 boundary correction.
 *
 * The argmax keeps the FIRST maximum, matching numpy's ``argmax`` on the float32
 * view of the same values -- the division by 8 is a strictly monotone scaling, so
 * the winner and the tie-break are identical in both representations.
 */
static int f26_row_corrected(
    const f26_hpac_model *model,
    const int16_t *row,
    const uint8_t *boundary,
    int32_t flat_position,
    float corrected[F26_ALPHABET],
    int32_t *predicted_out
) {
    int32_t predicted = 0;
    int32_t symbol;
    int32_t feature;
    for (symbol = 1; symbol < F26_ALPHABET; ++symbol) {
        if (row[symbol] > row[predicted]) predicted = symbol;
    }
    feature = (int32_t)boundary[flat_position] * F26_ALPHABET + predicted;
    if (feature < 0 || feature >= 25) return -7;
    for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
        corrected[symbol] = (float)row[symbol] / 8.0f +
            model->residual_table[feature * F26_ALPHABET + symbol];
    }
    if (predicted_out) *predicted_out = predicted;
    return 0;
}

/* Fold one group's decoded symbols into the incremental conv-A state. */
typedef struct {
    f26_rc64_decoder *decoder;
    const f26_hpac_model *model;
    int32_t target_start;
    int32_t target_count;
    /* One status cell per patch, so an illegal class is reported without any
     * cross-thread write.  A shared int written by several threads is a data
     * race even when every writer stores the same value. */
    int8_t *patch_status;
} f26_apply_job;

static void f26_group_apply_range(void *context, int32_t patch_begin, int32_t patch_end) {
    f26_apply_job *job = (f26_apply_job *)context;
    const f26_hpac_model *model = job->model;
    const uint8_t *decoded_patch_major = job->decoder->decoded_workspace;
    int32_t *conv_state = job->decoder->conv_state_workspace;
    int32_t target_start = job->target_start;
    int32_t target_count = job->target_count;
    int32_t patch_index;
    int32_t local_index;

    for (patch_index = patch_begin; patch_index < patch_end; ++patch_index) {
        for (local_index = 0; local_index < target_count; ++local_index) {
            uint8_t class_id = decoded_patch_major[patch_index * target_count + local_index];
            int32_t source_row;
            int32_t source_col;
            int32_t tap;
            if (!class_id) continue;
            if (class_id >= F26_ALPHABET) {
                job->patch_status[patch_index] = -8;
                continue;
            }
            source_row = model->target_positions[(target_start + local_index) * 2];
            source_col = model->target_positions[(target_start + local_index) * 2 + 1];
            for (tap = 0; tap < model->a_taps; ++tap) {
                int32_t hidden_row = source_row - model->a_offsets[tap * 2];
                int32_t hidden_col = source_col - model->a_offsets[tap * 2 + 1];
                int32_t *destination;
                const int16_t *class_delta;
                if (
                    hidden_row < 0 || hidden_row >= model->patch ||
                    hidden_col < 0 || hidden_col >= model->patch
                ) continue;
                destination = conv_state +
                    (((size_t)patch_index * (size_t)model->patch * (size_t)model->patch +
                      (size_t)hidden_row * (size_t)model->patch + (size_t)hidden_col) *
                     (size_t)model->channels);
                class_delta = model->conv_a_delta +
                    ((size_t)class_id * (size_t)model->a_taps + (size_t)tap) *
                    (size_t)model->channels;
                f26_add_i16x64(destination, class_delta);
            }
        }
    }
}

static int f26_group_apply(
    f26_rc64_decoder *decoder,
    const f26_hpac_model *model,
    int32_t group
) {
    f26_apply_job job;
    int32_t patch_index;
    int status = 0;
    if (!decoder->patch_status_workspace) return -4;
    memset(decoder->patch_status_workspace, 0, (size_t)model->patch_count);
    job.decoder = decoder;
    job.model = model;
    job.target_start = model->group_target_offsets[group];
    job.target_count = model->group_target_offsets[group + 1] - job.target_start;
    job.patch_status = decoder->patch_status_workspace;
    f26_parallel_for(model->patch_count, f26_group_apply_range, &job);
    for (patch_index = 0; patch_index < model->patch_count; ++patch_index) {
        if (decoder->patch_status_workspace[patch_index]) {
            status = decoder->patch_status_workspace[patch_index];
            break;
        }
    }
    return status;
}

int f26_hpac_decode_frame(
    void *opaque_decoder,
    const f26_hpac_model *model,
    const uint8_t *boundary,
    const uint8_t *previous,
    int32_t frame_index,
    uint8_t *current,
    float *corrected_trace,
    float *probability_trace
) {
    f26_rc64_decoder *decoder = (f26_rc64_decoder *)opaque_decoder;
    int16_t *logits = NULL;
    uint8_t *decoded_patch_major = NULL;
    int32_t group;
    int status = 0;
    double timing_started;

    if (
        !decoder || !model || !boundary || !previous || !current || !corrected_trace ||
        !probability_trace
    ) return -1;
    status = f26_check_model(model);
    if (status) return status;
    if (fesetround(FE_TONEAREST)) return -3;

    status = f26_ensure_workspaces(decoder, model);
    if (status) return status;
    logits = decoder->logits_workspace;
    decoded_patch_major = decoder->decoded_workspace;

    memset(current, 0, (size_t)model->height * (size_t)model->width);
    memset(f26_last_timing, 0, sizeof(f26_last_timing));
    timing_started = f26_now_seconds();
    status = f26_prepare_frame_context(decoder, model, frame_index, previous);
    if (status) return status - 40;
    f26_last_timing[0] = f26_now_seconds() - timing_started;

    timing_started = f26_now_seconds();
    f26_reset_conv_state(decoder, model);
    f26_last_timing[1] = f26_now_seconds() - timing_started;

    for (group = 0; group < model->groups; ++group) {
        int32_t target_start = model->group_target_offsets[group];
        int32_t target_count = model->group_target_offsets[group + 1] - target_start;
        int32_t output_start = model->group_output_offsets[group];
        int32_t output_count = model->group_output_offsets[group + 1] - output_start;
        int32_t local_index;

        if (output_count != model->patch_count * target_count) return -5;

        timing_started = f26_now_seconds();
        status = f26_group_model(decoder, model, group);
        if (status) return status;
        f26_last_timing[2] += f26_now_seconds() - timing_started;

        timing_started = f26_now_seconds();
        for (local_index = 0; local_index < output_count; ++local_index) {
            int32_t patch_major = model->output_order[output_start + local_index];
            int32_t flat_position = model->flat_positions[output_start + local_index];
            const int16_t *row = logits + (size_t)patch_major * F26_ALPHABET;
            float corrected[F26_ALPHABET];
            float probability[F26_ALPHABET];
            uint32_t frequency[F26_ALPHABET];
            int32_t symbol;
            uint8_t decoded;

            status = f26_row_corrected(model, row, boundary, flat_position, corrected, NULL);
            if (status) return status;
            for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
                corrected_trace[
                    ((size_t)output_start + (size_t)local_index) * F26_ALPHABET + symbol
                ] = corrected[symbol];
            }
            status = probability_and_frequencies(
                corrected, model->logit_precision, probability, frequency
            );
            if (status) return -20 + status;
            for (symbol = 0; symbol < F26_ALPHABET; ++symbol) {
                probability_trace[
                    ((size_t)output_start + (size_t)local_index) * F26_ALPHABET + symbol
                ] = probability[symbol];
            }
            status = f26_decode_row(decoder, frequency, &decoded);
            if (status) return status - 30;
            current[flat_position] = decoded;
            decoded_patch_major[patch_major] = decoded;
        }
        f26_last_timing[3] += f26_now_seconds() - timing_started;

        timing_started = f26_now_seconds();
        status = f26_group_apply(decoder, model, group);
        if (status) return status;
        f26_last_timing[4] += f26_now_seconds() - timing_started;
    }
    return 0;
}

/*
 * ---------------------------------------------------------------------------
 * ddm_wc2c split entry points.
 *
 * WHY THE SPLIT EXISTS.  ``f26_hpac_decode_frame`` fuses the integer model, the
 * probability table, and the RC64 coder into one call, which leaves no seam for
 * the shipped decode-time probability corrector.  The corrector is float64 on
 * its decision path and its identity rests on IEEE ``+ - * /`` in a hand-fixed
 * summation order (``free_corrector.py:266-279``); reimplementing it in C would
 * put a 2,121-line adaptive stack under a reduction-order hazard whose measured
 * failure mode is a catastrophically desynchronised decoder, not a rounding
 * wobble.  Splitting instead lets the audited numpy corrector stay exactly where
 * it is while the integer work -- which carries no such hazard -- is lowered.
 *
 * CONTRACT.  Per frame:  frame_begin, then for each group in order
 * group_logits -> (caller: probability, corrector, RC64) -> group_commit.  The
 * caller must not skip or reorder groups: ``conv_state`` is autoregressive and
 * ``group_commit`` is the only thing that advances it.
 * ---------------------------------------------------------------------------
 */

int32_t f26_hpac_group_output_count(const f26_hpac_model *model, int32_t group) {
    if (!model || group < 0 || group >= model->groups) return -1;
    return model->group_output_offsets[group + 1] - model->group_output_offsets[group];
}

void f26_hpac_timing_reset(void) {
    memset(f26_last_timing, 0, sizeof(f26_last_timing));
}

int f26_hpac_frame_begin(
    void *opaque_decoder,
    const f26_hpac_model *model,
    const uint8_t *previous,
    int32_t frame_index,
    uint8_t *current
) {
    f26_rc64_decoder *decoder = (f26_rc64_decoder *)opaque_decoder;
    int status;
    double timing_started;

    if (!decoder || !model || !previous || !current) return -1;
    decoder->frame_ready = 0;
    status = f26_check_model(model);
    if (status) return status;
    if (fesetround(FE_TONEAREST)) return -3;
    status = f26_ensure_workspaces(decoder, model);
    if (status) return status;

    memset(current, 0, (size_t)model->height * (size_t)model->width);
    timing_started = f26_now_seconds();
    status = f26_prepare_frame_context(decoder, model, frame_index, previous);
    if (status) return status - 40;
    f26_last_timing[0] += f26_now_seconds() - timing_started;

    timing_started = f26_now_seconds();
    f26_reset_conv_state(decoder, model);
    f26_last_timing[1] += f26_now_seconds() - timing_started;
    decoder->frame_ready = 1;
    return 0;
}

int f26_hpac_group_logits(
    void *opaque_decoder,
    const f26_hpac_model *model,
    const uint8_t *boundary,
    int32_t group,
    float *corrected_out,
    int32_t *predicted_out,
    int32_t *flat_out
) {
    f26_rc64_decoder *decoder = (f26_rc64_decoder *)opaque_decoder;
    const int16_t *logits;
    int32_t target_start;
    int32_t target_count;
    int32_t output_start;
    int32_t output_count;
    int32_t local_index;
    int status;
    double timing_started;

    if (!decoder || !model || !boundary || !corrected_out || !predicted_out || !flat_out) {
        return -1;
    }
    if (!decoder->frame_ready) return -9;
    if (group < 0 || group >= model->groups) return -6;
    target_start = model->group_target_offsets[group];
    target_count = model->group_target_offsets[group + 1] - target_start;
    output_start = model->group_output_offsets[group];
    output_count = model->group_output_offsets[group + 1] - output_start;
    if (output_count != model->patch_count * target_count) return -5;

    timing_started = f26_now_seconds();
    status = f26_group_model(decoder, model, group);
    if (status) return status;
    f26_last_timing[2] += f26_now_seconds() - timing_started;

    timing_started = f26_now_seconds();
    logits = decoder->logits_workspace;
    for (local_index = 0; local_index < output_count; ++local_index) {
        int32_t patch_major = model->output_order[output_start + local_index];
        int32_t flat_position = model->flat_positions[output_start + local_index];
        const int16_t *row = logits + (size_t)patch_major * F26_ALPHABET;
        int32_t predicted = 0;
        status = f26_row_corrected(
            model, row, boundary, flat_position,
            corrected_out + (size_t)local_index * F26_ALPHABET, &predicted
        );
        if (status) return status;
        predicted_out[local_index] = predicted;
        flat_out[local_index] = flat_position;
    }
    f26_last_timing[3] += f26_now_seconds() - timing_started;
    return 0;
}

int f26_hpac_group_commit(
    void *opaque_decoder,
    const f26_hpac_model *model,
    int32_t group,
    const uint8_t *symbols,
    uint8_t *current
) {
    f26_rc64_decoder *decoder = (f26_rc64_decoder *)opaque_decoder;
    uint8_t *decoded_patch_major;
    int32_t output_start;
    int32_t output_count;
    int32_t local_index;
    int status;
    double timing_started;

    if (!decoder || !model || !symbols || !current) return -1;
    if (!decoder->frame_ready) return -9;
    if (group < 0 || group >= model->groups) return -6;
    output_start = model->group_output_offsets[group];
    output_count = model->group_output_offsets[group + 1] - output_start;

    timing_started = f26_now_seconds();
    decoded_patch_major = decoder->decoded_workspace;
    for (local_index = 0; local_index < output_count; ++local_index) {
        int32_t patch_major = model->output_order[output_start + local_index];
        int32_t flat_position = model->flat_positions[output_start + local_index];
        uint8_t decoded = symbols[local_index];
        if (decoded >= F26_ALPHABET) return -8;
        current[flat_position] = decoded;
        decoded_patch_major[patch_major] = decoded;
    }
    status = f26_group_apply(decoder, model, group);
    if (status) return status;
    f26_last_timing[4] += f26_now_seconds() - timing_started;
    return 0;
}

uint64_t f26_total_frequency(void) {
    return F26_TOTAL;
}

void f26_hpac_last_timing(double output[5]) {
    if (output) memcpy(output, f26_last_timing, sizeof(f26_last_timing));
}
