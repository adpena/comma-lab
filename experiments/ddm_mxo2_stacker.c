/* MXO2 fixed-point low-rank quadratic stacker.
 *
 * Generic code only: the two projection sign matrices are derived from the
 * declared seed.  The only mutable model state is the cold-started online
 * output weight matrix.  process_group predicts every member of a shipped
 * decode group before applying the group's aggregate update.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MXO2_INPUTS 23
#define MXO2_CLASSES 5
#define MXO2_MAX_RANK 8
#define MXO2_TOTAL ((int64_t)1 << 31)
#define MXO2_WEIGHT_LIMIT 32768
#define MXO2_SCORE_LIMIT 2048
#define MXO2_UPDATE_SHIFT 34

typedef struct {
    int rank;
    int8_t left[MXO2_MAX_RANK][MXO2_INPUTS];
    int8_t right[MXO2_MAX_RANK][MXO2_INPUTS];
    int32_t weights[MXO2_CLASSES][MXO2_MAX_RANK];
} Mxo2Stacker;

static uint64_t splitmix64(uint64_t *state)
{
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static int64_t floor_shift(int64_t value, int shift)
{
    if (value >= 0) {
        return value >> shift;
    }
    return -(((-value) + (((int64_t)1 << shift) - 1)) >> shift);
}

static int64_t round_div(int64_t numerator, int64_t denominator)
{
    if (numerator >= 0) {
        return (numerator + denominator / 2) / denominator;
    }
    return -((-numerator + denominator / 2) / denominator);
}

static int64_t clip64(int64_t value, int64_t low, int64_t high)
{
    return value < low ? low : value > high ? high : value;
}

void *mxo2_create(int rank, uint64_t seed)
{
    if (rank != 2 && rank != 4 && rank != 8) {
        return NULL;
    }
    Mxo2Stacker *self = (Mxo2Stacker *)calloc(1, sizeof(*self));
    if (!self) {
        return NULL;
    }
    self->rank = rank;
    uint64_t state = seed;
    for (int h = 0; h < MXO2_MAX_RANK; ++h) {
        for (int j = 0; j < MXO2_INPUTS; ++j) {
            self->left[h][j] = (splitmix64(&state) & 1) ? 1 : -1;
            self->right[h][j] = (splitmix64(&state) & 1) ? 1 : -1;
        }
    }
    return self;
}

void mxo2_destroy(void *handle)
{
    free(handle);
}

int mxo2_get_weights(void *handle, int32_t *output, int count)
{
    Mxo2Stacker *self = (Mxo2Stacker *)handle;
    if (!self || !output || count != MXO2_CLASSES * self->rank) {
        return -1;
    }
    for (int c = 0; c < MXO2_CLASSES; ++c) {
        memcpy(output + c * self->rank, self->weights[c],
               (size_t)self->rank * sizeof(int32_t));
    }
    return 0;
}

int mxo2_set_weights(void *handle, const int32_t *input, int count)
{
    Mxo2Stacker *self = (Mxo2Stacker *)handle;
    if (!self || !input || count != MXO2_CLASSES * self->rank) {
        return -1;
    }
    for (int c = 0; c < MXO2_CLASSES; ++c) {
        for (int h = 0; h < self->rank; ++h) {
            if (input[c * self->rank + h] < -MXO2_WEIGHT_LIMIT ||
                input[c * self->rank + h] > MXO2_WEIGHT_LIMIT) {
                return -2;
            }
            self->weights[c][h] = input[c * self->rank + h];
        }
    }
    return 0;
}

static int16_t quadratic_feature(const Mxo2Stacker *self, int h,
                                 const uint16_t *family, uint16_t mixed)
{
    int64_t left = 0;
    int64_t right = 0;
    for (int j = 0; j < MXO2_INPUTS; ++j) {
        int64_t centered = (int64_t)family[j] - (int64_t)mixed;
        left += self->left[h][j] * centered;
        right += self->right[h][j] * centered;
    }
    left = clip64(floor_shift(left, 3), -32768, 32767);
    right = clip64(floor_shift(right, 3), -32768, 32767);
    return (int16_t)clip64(floor_shift(left * right, 15), -32768, 32767);
}

static void rescale_row(const uint32_t *source, int hit_class, int64_t hit,
                        uint32_t *output)
{
    int64_t old_hit = source[hit_class];
    int64_t old_miss = MXO2_TOTAL - old_hit;
    int64_t new_miss = MXO2_TOTAL - hit;
    int final_miss = -1;
    int64_t used = 0;
    for (int c = 0; c < MXO2_CLASSES; ++c) {
        if (c == hit_class) {
            output[c] = (uint32_t)hit;
            continue;
        }
        final_miss = c;
        int64_t value = round_div((int64_t)source[c] * new_miss, old_miss);
        value = clip64(value, 1, new_miss - 3);
        output[c] = (uint32_t)value;
        used += value;
    }
    int64_t excess = used - new_miss;
    if (excess >= 0) {
        for (int c = MXO2_CLASSES - 1; c >= 0 && excess >= 0; --c) {
            if (c == hit_class) {
                continue;
            }
            int64_t removable = (int64_t)output[c] - 1;
            int64_t take = removable < excess + 1 ? removable : excess + 1;
            output[c] -= (uint32_t)take;
            used -= take;
            excess = used - new_miss;
        }
    }
    output[final_miss] += (uint32_t)(new_miss - used);
}

int mxo2_process_group(void *handle, const uint16_t *family_q15,
                       const uint16_t *mixer_q15, const uint32_t *frequencies,
                       const uint8_t *hit_class, const uint8_t *symbols,
                       int64_t count, uint32_t *output)
{
    Mxo2Stacker *self = (Mxo2Stacker *)handle;
    if (!self || !family_q15 || !mixer_q15 || !frequencies || !hit_class ||
        !symbols || !output || count <= 0) {
        return -1;
    }
    int64_t gradient[MXO2_CLASSES][MXO2_MAX_RANK] = {{0}};
    int64_t class_count[MXO2_CLASSES] = {0};

    for (int64_t i = 0; i < count; ++i) {
        int cls = hit_class[i];
        int symbol = symbols[i];
        if (cls < 0 || cls >= MXO2_CLASSES || symbol < 0 || symbol >= MXO2_CLASSES) {
            return -2;
        }
        int16_t phi[MXO2_MAX_RANK];
        int64_t score_acc = 0;
        const uint16_t *family = family_q15 + i * MXO2_INPUTS;
        for (int h = 0; h < self->rank; ++h) {
            phi[h] = quadratic_feature(self, h, family, mixer_q15[i]);
            score_acc += (int64_t)self->weights[cls][h] * phi[h];
        }
        int64_t score = clip64(floor_shift(score_acc, 15),
                               -MXO2_SCORE_LIMIT, MXO2_SCORE_LIMIT);
        const uint32_t *row = frequencies + i * MXO2_CLASSES;
        int64_t base_hit = row[cls];
        int64_t variance = floor_shift(base_hit * (MXO2_TOTAL - base_hit), 31);
        int64_t hit = clip64(base_hit + floor_shift(variance * score, 12),
                             4, MXO2_TOTAL - 4);
        rescale_row(row, cls, hit, output + i * MXO2_CLASSES);

        int64_t actual_hit = output[i * MXO2_CLASSES + cls];
        int64_t residual = (symbol == cls ? MXO2_TOTAL : 0) - actual_hit;
        class_count[cls] += 1;
        for (int h = 0; h < self->rank; ++h) {
            gradient[cls][h] += residual * (int64_t)phi[h];
        }
    }

    for (int c = 0; c < MXO2_CLASSES; ++c) {
        if (!class_count[c]) {
            continue;
        }
        for (int h = 0; h < self->rank; ++h) {
            int64_t mean = gradient[c][h] / class_count[c];
            int64_t step = floor_shift(mean, MXO2_UPDATE_SHIFT);
            self->weights[c][h] = (int32_t)clip64(
                (int64_t)self->weights[c][h] + step,
                -MXO2_WEIGHT_LIMIT, MXO2_WEIGHT_LIMIT);
        }
    }
    return 0;
}

