/* Ordered float64 adaptive probability correction; no contracted operations. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define NUM_CLASSES 5
#define U_BINS 64
#define RUN_LEVELS 8
#define RUN_CAP 255
#define BOUNDARY_LEVELS 5
#define KT_ALPHA 0.5
#define MIN_COUNT 32
#define ODDS_LOW 0.0625
#define ODDS_HIGH 16.0
#define PROB_EPS 1e-9
#define PHAT_SCALE 1073741824.0
#define HEIGHT 384
#define WIDTH 512
#define SPATIAL_LEVELS 5
#define WEIGHT_STORE_BITS 20
#define WEIGHT_STORE_ONE ((int64_t)1 << 20)
#define POWER_BITS 6
#define INT_POWER_BITS 4
#define ERR_SCALE 1048576.0
#define STRETCH_SCALE 1048576.0
#define STRETCH_CLAMP 33554432.0
#define COUNT_HALVING_PASSES 40
#define WEIGHT_LOW (-4 * WEIGHT_STORE_ONE)
#define WEIGHT_HIGH (8 * WEIGHT_STORE_ONE)
#define STORE_SHIFT (WEIGHT_STORE_BITS - POWER_BITS)
#define LR_SHIFT 24
#define SPATIAL4_LEVELS 6
#define HOMOGENEITY_LEVELS 5
#define N_CAUSAL 4
#define GROUP_BINS 8
#define UNKNOWN NUM_CLASSES
#define MISS_KT_ALPHA 0.5
#define MISS_MIN_COUNT 1
#define MISS_CLAMP_HIGH 16.0
#define MISS_CLAMP_LOW 0.0625
#define MISS_BASE (NUM_CLASSES + 1)
#define N_MISS_CELLS (MISS_BASE * MISS_BASE * MISS_BASE * MISS_BASE)
#define N_FAMILIES 23
#define N_MIXER_CONTEXTS (NUM_CLASSES * BOUNDARY_LEVELS * 4 * HOMOGENEITY_LEVELS * 8)
#define N_WEIGHT_SETS N_MIXER_CONTEXTS
#define JOINT_SIZE (NUM_CLASSES * U_BINS * 2 * 2 * RUN_LEVELS * BOUNDARY_LEVELS)
#define SLOT_LEFT 0
#define SLOT_UP 1
#define SLOT_UPRIGHT 2
/* The four causal neighbours, in slot order: left, up, up-right, up-left. */
static const int CAUSAL_DX[N_CAUSAL] = {-1, 0, 1, -1};
static const int CAUSAL_DY[N_CAUSAL] = {0, -1, -1, -1};
enum {
    RULE_JOINT = 0, RULE_TEMPORAL_SPATIAL, RULE_SURPRISE_ONLY,
    RULE_SPATIAL_SURPRISE, RULE_SPATIAL_BOUNDARY, RULE_RUN_SURPRISE,
    RULE_BOUNDARY_SURPRISE, RULE_TEMPORAL_SURPRISE, RULE_SPATIAL4_SURPRISE,
    RULE_HOMOG_SURPRISE, RULE_HOMOG_BOUNDARY_SURPRISE, RULE_SPATIAL4_BOUNDARY,
    RULE_HOMOG_SPATIAL4, RULE_SPATIAL4_TEMPORAL, RULE_GROUPBIN8_SURPRISE,
    RULE_CLS_GROUPBIN8, RULE_PATCH192_ONLY, RULE_TILE48_GROUPBIN8
};
/* Four parallel tables, one entry per context family, read together by index.
 *
 *   FAMILY_RULE              which context the family keys on; the switch in
 *                            family_rule_index turns that rule plus the pixel's
 *                            features into one table offset.
 *   FAMILY_SIZE              how many cells that family's table has, so it is the
 *                            product of the level counts the rule combines.
 *   FAMILY_COUNT_LIMIT       cap on a cell's observation count, 0 meaning no cap.
 *                            A cap makes the family forget: whenever a cell goes
 *                            over its cap, family_halve halves that cell's counts
 *                            until it is back under, so it tracks recent pixels
 *                            rather than all of them. That is why families 8, 9,
 *                            10, 17 and 18 repeat an earlier rule -- the same
 *                            context, with a shorter memory.
 *   FAMILY_MIXER_STARTS_AT_ONE  the one family whose mixer weight begins at 1.0
 *                            instead of 0; every other family has to earn its
 *                            weight from the first pixels it sees.
 */
static const int FAMILY_RULE[N_FAMILIES] = {
    RULE_JOINT, RULE_TEMPORAL_SPATIAL, RULE_SURPRISE_ONLY, RULE_SPATIAL_SURPRISE, RULE_SPATIAL_BOUNDARY,
    RULE_RUN_SURPRISE, RULE_BOUNDARY_SURPRISE, RULE_TEMPORAL_SURPRISE, RULE_JOINT, RULE_JOINT,
    RULE_SURPRISE_ONLY, RULE_SPATIAL4_SURPRISE, RULE_HOMOG_SURPRISE, RULE_HOMOG_BOUNDARY_SURPRISE,
    RULE_SPATIAL4_BOUNDARY, RULE_HOMOG_SPATIAL4, RULE_SPATIAL4_TEMPORAL, RULE_HOMOG_SURPRISE,
    RULE_SPATIAL4_SURPRISE, RULE_GROUPBIN8_SURPRISE, RULE_CLS_GROUPBIN8, RULE_PATCH192_ONLY,
    RULE_TILE48_GROUPBIN8
};
static const int64_t FAMILY_SIZE[N_FAMILIES] = {
    JOINT_SIZE, NUM_CLASSES * 2 * 2 * SPATIAL_LEVELS, NUM_CLASSES * U_BINS,
    NUM_CLASSES * SPATIAL_LEVELS * U_BINS, NUM_CLASSES * SPATIAL_LEVELS * BOUNDARY_LEVELS,
    NUM_CLASSES * RUN_LEVELS * U_BINS, NUM_CLASSES * BOUNDARY_LEVELS * U_BINS, NUM_CLASSES * 2 * 2 * U_BINS,
    JOINT_SIZE, JOINT_SIZE, NUM_CLASSES * U_BINS, NUM_CLASSES * SPATIAL4_LEVELS * U_BINS,
    NUM_CLASSES * HOMOGENEITY_LEVELS * U_BINS, NUM_CLASSES * HOMOGENEITY_LEVELS * BOUNDARY_LEVELS * U_BINS,
    NUM_CLASSES * SPATIAL4_LEVELS * BOUNDARY_LEVELS, NUM_CLASSES * HOMOGENEITY_LEVELS * SPATIAL4_LEVELS,
    NUM_CLASSES * 2 * 2 * SPATIAL4_LEVELS, NUM_CLASSES * HOMOGENEITY_LEVELS * U_BINS,
    NUM_CLASSES * SPATIAL4_LEVELS * U_BINS, NUM_CLASSES * GROUP_BINS * U_BINS, NUM_CLASSES * GROUP_BINS, 192,
    48 * GROUP_BINS
};
static const int64_t FAMILY_COUNT_LIMIT[N_FAMILIES] = {
    0, 0, 0, 0, 0, 0, 0, 0, 256, 4096, 256, 0, 0, 0, 0, 0, 0, 256, 256, 0, 0, 0, 0
};
static const int FAMILY_MIXER_STARTS_AT_ONE[N_FAMILIES] = {
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
static inline int64_t floor_div_i64(int64_t numerator, int64_t denominator) {
    int64_t quotient = numerator / denominator;
    int64_t remainder = numerator % denominator;
    if (remainder != 0 && ((remainder < 0) != (denominator < 0))) quotient -= 1;
    return quotient;
}
static inline int64_t round_shift(int64_t value, int bits) {
    if (bits <= 0) return value;
    int64_t half = (int64_t)1 << (bits - 1);
    return floor_div_i64(value + half, (int64_t)1 << bits);
}
static inline double clamp_double(double value, double low, double high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}
static inline int64_t clamp_i64(int64_t value, int64_t low, int64_t high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}
static inline int64_t min_i64(int64_t a, int64_t b) { return a < b ? a : b; }
static double SURPRISE_ASC[U_BINS - 1];
static int SURPRISE_READY = 0;
static const uint64_t INV_SQRT2_BITS = 0x3FE6A09E667F3BCDull;
static int build_surprise_table(void) {
    double inv_sqrt2 = sqrt(0.5);
    uint64_t bits;
    memcpy(&bits, &inv_sqrt2, sizeof(bits));
    if (bits != INV_SQRT2_BITS) return -1;
    double descending[U_BINS - 1];
    for (int k = 1; k < U_BINS; ++k) {
        double value = ldexp(1.0, -(k / 2));
        if (k % 2) value *= inv_sqrt2;
        descending[k - 1] = value;
    }
    for (int i = 0; i < U_BINS - 1; ++i) SURPRISE_ASC[i] = descending[U_BINS - 2 - i];
    SURPRISE_READY = 1;
    return 0;
}
static inline int64_t searchsorted_left(const double *ascending, int64_t size, double value) {
    int64_t low = 0, high = size;
    while (low < high) {
        int64_t mid = low + ((high - low) >> 1);
        if (ascending[mid] < value) low = mid + 1; else high = mid;
    }
    return low;
}
static int sqrt_is_correctly_rounded(void) {
    static const double roots[8] = {1.0, 1.5, 2.0, 3.0, 4.0, 1.0625, 65536.0, 1e8};
    for (int i = 0; i < 8; ++i) {
        double squared = roots[i] * roots[i];
        if (sqrt(squared) != roots[i]) return 0;
    }
    double value = 1.0;
    for (int i = 0; i < 8; ++i) value = value / 4.0;
    for (int i = 0; i < 16; ++i) {
        if (sqrt(value * value) != value) return 0;
        value = value * 4.0;
    }
    return 1;
}
typedef struct {
    int64_t *counts, *hits, *phat_q;
    int64_t size, count_limit;
    int rule;
} Family;
typedef struct {
    int64_t plane;
    uint8_t *prev1, *prev2;
    int64_t *run, *boundary;
    int have_prev;
    uint8_t *current, *known;
    Family families[N_FAMILIES];
    int64_t *weights, *miss_counts, *miss_expect, *miss_seen;
    int64_t capacity, n;
    int group_open;
    double *row64;
    int64_t *arg;
    double *p_max, *one_minus;
    int64_t *p_max_q, *flat, *fam_index, *mixer, *miss_cell;
    double *stretch, *q, *blended, *out64;
    int64_t *ws_counts, *ws_gradient, *residual, *hit;
} Corrector;
static void family_free(Family *family) {
    free(family->counts); free(family->hits); free(family->phat_q);
    family->counts = family->hits = family->phat_q = NULL;
}
static int family_init(Family *family, int rule, int64_t size, int64_t count_limit) {
    family->rule = rule;
    family->size = size;
    family->count_limit = count_limit;
    family->counts = (int64_t *)calloc((size_t)size, sizeof(int64_t));
    family->hits = (int64_t *)calloc((size_t)size, sizeof(int64_t));
    family->phat_q = (int64_t *)calloc((size_t)size, sizeof(int64_t));
    if (!family->counts || !family->hits || !family->phat_q) {
        family_free(family);
        return -1;
    }
    return 0;
}
void corrector_destroy(void *handle) {
    Corrector *self = (Corrector *)handle;
    if (!self) return;
    for (int i = 0; i < N_FAMILIES; ++i) family_free(&self->families[i]);
    free(self->prev1); free(self->prev2); free(self->run);
    free(self->boundary); free(self->current); free(self->known);
    free(self->weights); free(self->miss_counts); free(self->miss_expect);
    free(self->miss_seen); free(self->row64); free(self->arg);
    free(self->p_max); free(self->one_minus); free(self->p_max_q);
    free(self->flat); free(self->fam_index); free(self->mixer);
    free(self->miss_cell); free(self->stretch); free(self->q);
    free(self->blended); free(self->out64); free(self->ws_counts);
    free(self->ws_gradient); free(self->residual); free(self->hit);
    free(self);
}
/* A failed growth keeps every owned pointer valid for destruction or retry. */
static int ensure_capacity(Corrector *self, int64_t n) {
    if (n <= self->capacity) return 0;
    int64_t capacity = self->capacity ? self->capacity : 1024;
    while (capacity < n) capacity *= 2;
#define GROW(name, type, count) do { \
    type *next = realloc(self->name, (size_t)capacity * (count) * sizeof(type)); \
    if (!next) return -1; \
    self->name = next; \
} while (0)
    GROW(row64, double, NUM_CLASSES);
    GROW(arg, int64_t, 1);
    GROW(p_max, double, 1);
    GROW(one_minus, double, 1);
    GROW(p_max_q, int64_t, 1);
    GROW(flat, int64_t, 1);
    GROW(fam_index, int64_t, N_FAMILIES);
    GROW(mixer, int64_t, 1);
    GROW(miss_cell, int64_t, 1);
    GROW(stretch, double, N_FAMILIES);
    GROW(q, double, 1);
    GROW(blended, double, 1);
    GROW(out64, double, NUM_CLASSES);
    GROW(residual, int64_t, 1);
    GROW(hit, int64_t, 1);
#undef GROW
    self->capacity = capacity;
    return 0;
}
void *corrector_create(int64_t plane) {
    if (plane != (int64_t)HEIGHT * WIDTH) return NULL;
    if (!SURPRISE_READY && build_surprise_table() != 0) return NULL;
    if (!sqrt_is_correctly_rounded()) return NULL;
    Corrector *self = (Corrector *)calloc(1, sizeof(Corrector));
    if (!self) return NULL;
    self->plane = plane;
    self->prev1 = (uint8_t *)calloc((size_t)plane, sizeof(uint8_t));
    self->prev2 = (uint8_t *)calloc((size_t)plane, sizeof(uint8_t));
    self->run = (int64_t *)calloc((size_t)plane, sizeof(int64_t));
    self->boundary = (int64_t *)malloc((size_t)plane * sizeof(int64_t));
    self->current = (uint8_t *)calloc((size_t)plane, sizeof(uint8_t));
    self->known = (uint8_t *)calloc((size_t)plane, sizeof(uint8_t));
    self->weights = (int64_t *)calloc((size_t)N_WEIGHT_SETS * N_FAMILIES, sizeof(int64_t));
    self->miss_counts = (int64_t *)calloc((size_t)N_MISS_CELLS * NUM_CLASSES, sizeof(int64_t));
    self->miss_expect = (int64_t *)calloc((size_t)N_MISS_CELLS * NUM_CLASSES, sizeof(int64_t));
    self->miss_seen = (int64_t *)calloc((size_t)N_MISS_CELLS, sizeof(int64_t));
    self->ws_counts = (int64_t *)calloc((size_t)N_WEIGHT_SETS, sizeof(int64_t));
    self->ws_gradient = (int64_t *)calloc((size_t)N_WEIGHT_SETS, sizeof(int64_t));
    if (!self->prev1 || !self->prev2 || !self->run || !self->boundary || !self->current ||
        !self->known || !self->weights || !self->miss_counts || !self->miss_expect ||
        !self->miss_seen || !self->ws_counts || !self->ws_gradient) {
        corrector_destroy(self);
        return NULL;
    }
    for (int i = 0; i < N_FAMILIES; ++i) {
        if (family_init(&self->families[i], FAMILY_RULE[i], FAMILY_SIZE[i],
                        FAMILY_COUNT_LIMIT[i]) != 0) {
            corrector_destroy(self);
            return NULL;
        }
    }
    for (int64_t i = 0; i < plane; ++i) self->boundary[i] = BOUNDARY_LEVELS - 1;
    for (int64_t ws = 0; ws < N_WEIGHT_SETS; ++ws) {
        for (int pos = 0; pos < N_FAMILIES; ++pos) {
            self->weights[ws * N_FAMILIES + pos] =
                FAMILY_MIXER_STARTS_AT_ONE[pos] ? WEIGHT_STORE_ONE : 0;
        }
    }
    if (ensure_capacity(self, 2048) != 0) {
        corrector_destroy(self);
        return NULL;
    }
    return self;
}
int corrector_begin_frame(void *handle, const int64_t *boundary, int64_t size) {
    Corrector *self = (Corrector *)handle;
    if (!self || size != self->plane) return -1;
    memcpy(self->boundary, boundary, (size_t)size * sizeof(int64_t));
    memset(self->known, 0, (size_t)self->plane * sizeof(uint8_t));
    memset(self->current, 0, (size_t)self->plane * sizeof(uint8_t));
    self->group_open = 0;
    return 0;
}
static inline double family_multiplier(const Family *family, int64_t index) {
    int64_t raw_count = family->counts[index];
    double count = (double)raw_count;
    double denominator = count + 2.0 * KT_ALPHA;
    double hit_numerator = (double)family->hits[index] + KT_ALPHA;
    double hit_denominator = denominator - hit_numerator;
    double expected = (double)family->phat_q[index] / PHAT_SCALE;
    double exp_numerator = expected + KT_ALPHA;
    double exp_denominator = denominator - exp_numerator;
    double multiplier = 1.0;
    if (raw_count >= MIN_COUNT && hit_numerator > 0.0 && hit_denominator > 0.0 &&
        exp_numerator > 0.0 && exp_denominator > 0.0) {
        multiplier = (hit_numerator * exp_denominator) / (hit_denominator * exp_numerator);
    }
    return clamp_double(multiplier, ODDS_LOW, ODDS_HIGH);
}
static inline double dyadic_power(double value, const double *radicals, int64_t weight) {
    int negative = weight < 0;
    int64_t magnitude = negative ? -weight : weight;
    int64_t integer_part = magnitude >> POWER_BITS;
    int64_t fraction = magnitude & (((int64_t)1 << POWER_BITS) - 1);
    double accumulator = 1.0;
    double base = value;
    int64_t remaining = integer_part;
    for (int i = 0; i < INT_POWER_BITS; ++i) {
        if ((remaining & 1) == 1) accumulator = accumulator * base;
        base = base * base;
        remaining >>= 1;
    }
    for (int index = 0; index < POWER_BITS; ++index) {
        int64_t bit = (fraction >> (POWER_BITS - 1 - index)) & 1;
        if (bit == 1) accumulator = accumulator * radicals[index];
    }
    return negative ? 1.0 / accumulator : accumulator;
}
static inline int64_t family_rule_index(int rule, int64_t cls, int64_t ubin, int64_t agree1,
                                        int64_t agree2, int64_t run, int64_t boundary,
                                        int64_t spatial, int64_t spatial4, int64_t homog,
                                        int64_t groupbin8, int64_t patch192,
                                        int64_t tile48_groupbin8) {
    switch (rule) {
    case RULE_JOINT: {
        int64_t head = ((cls * U_BINS + ubin) * 2 + agree1) * 2 + agree2;
        return (head * RUN_LEVELS + run) * BOUNDARY_LEVELS + boundary;
    }
    case RULE_TEMPORAL_SPATIAL: {
        int64_t head = (cls * 2 + agree1) * 2 + agree2;
        return head * SPATIAL_LEVELS + spatial;
    }
    case RULE_SURPRISE_ONLY:
        return cls * U_BINS + ubin;
    case RULE_SPATIAL_SURPRISE:
        return (cls * SPATIAL_LEVELS + spatial) * U_BINS + ubin;
    case RULE_SPATIAL_BOUNDARY:
        return (cls * SPATIAL_LEVELS + spatial) * BOUNDARY_LEVELS + boundary;
    case RULE_RUN_SURPRISE:
        return (cls * RUN_LEVELS + run) * U_BINS + ubin;
    case RULE_BOUNDARY_SURPRISE:
        return (cls * BOUNDARY_LEVELS + boundary) * U_BINS + ubin;
    case RULE_TEMPORAL_SURPRISE: {
        int64_t head = (cls * 2 + agree1) * 2 + agree2;
        return head * U_BINS + ubin;
    }
    case RULE_SPATIAL4_SURPRISE:
        return (cls * SPATIAL4_LEVELS + spatial4) * U_BINS + ubin;
    case RULE_HOMOG_SURPRISE:
        return (cls * HOMOGENEITY_LEVELS + homog) * U_BINS + ubin;
    case RULE_HOMOG_BOUNDARY_SURPRISE: {
        int64_t head = (cls * HOMOGENEITY_LEVELS + homog) * BOUNDARY_LEVELS + boundary;
        return head * U_BINS + ubin;
    }
    case RULE_SPATIAL4_BOUNDARY:
        return (cls * SPATIAL4_LEVELS + spatial4) * BOUNDARY_LEVELS + boundary;
    case RULE_HOMOG_SPATIAL4:
        return (cls * HOMOGENEITY_LEVELS + homog) * SPATIAL4_LEVELS + spatial4;
    case RULE_SPATIAL4_TEMPORAL: {
        int64_t head = (cls * 2 + agree1) * 2 + agree2;
        return head * SPATIAL4_LEVELS + spatial4;
    }
    case RULE_GROUPBIN8_SURPRISE:
        return (cls * GROUP_BINS + groupbin8) * U_BINS + ubin;
    case RULE_CLS_GROUPBIN8:
        return cls * GROUP_BINS + groupbin8;
    case RULE_PATCH192_ONLY:
        return patch192;
    case RULE_TILE48_GROUPBIN8:
        return tile48_groupbin8;
    default:
        return 0;
    }
}
int64_t tile_context(int64_t x, int64_t y) {
    if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) return -1;
    int64_t tile48 = (y / 64) * (WIDTH / 64) + (x / 64);
    int64_t groupbin8 = (((x % 64) + 2 * (y % 64)) * GROUP_BINS) / 190;
    return tile48 * GROUP_BINS + groupbin8;
}
int corrector_group_state(void *handle, const float *probability,
                              const int64_t *predicted, const int64_t *positions, int64_t n) {
    Corrector *self = (Corrector *)handle;
    if (!self || n <= 0 || n > self->plane || self->group_open) return -1;
    for (int64_t i = 0; i < n; ++i) if (positions[i] < 0 || positions[i] >= self->plane || predicted[i] < 0 || predicted[i] >= 5) return -1;
    if (ensure_capacity(self, n) != 0) return -1;
    self->n = n;
    for (int64_t i = 0; i < n; ++i) {
        double *row = &self->row64[i * NUM_CLASSES];
        for (int c = 0; c < NUM_CLASSES; ++c) row[c] = (double)probability[i * NUM_CLASSES + c];
        int64_t arg = 0;
        for (int c = 1; c < NUM_CLASSES; ++c) if (row[c] > row[arg]) arg = c;
        double p_max = row[arg];
        double one_minus = 1.0 - p_max;
        if (!(one_minus > PROB_EPS)) one_minus = PROB_EPS;
        int64_t below = searchsorted_left(SURPRISE_ASC, U_BINS - 1, one_minus);
        int64_t ubin = clamp_i64((U_BINS - 1) - below, 0, U_BINS - 1);
        int64_t base_class = predicted[i];
        int64_t flat = positions[i];
        int64_t agree1 = 0, agree2 = 0;
        if (self->have_prev) {
            agree1 = ((int64_t)self->prev1[flat] == base_class) ? 1 : 0;
            agree2 = ((int64_t)self->prev2[flat] == base_class) ? 1 : 0;
        }
        int64_t run = min_i64(self->run[flat], RUN_LEVELS - 1);
        int64_t head = ((base_class * U_BINS + ubin) * 2 + agree1) * 2 + agree2;
        int64_t context = (head * RUN_LEVELS + run) * BOUNDARY_LEVELS + self->boundary[flat];
        self->arg[i] = arg;
        self->p_max[i] = p_max;
        self->one_minus[i] = one_minus;
        self->p_max_q[i] = (int64_t)rint(p_max * PHAT_SCALE);
        self->flat[i] = flat;
        int64_t packed = context;
        int64_t boundary_f = packed % BOUNDARY_LEVELS;
        int64_t rest = packed / BOUNDARY_LEVELS;
        int64_t run_f = rest % RUN_LEVELS;
        rest /= RUN_LEVELS;
        int64_t agree2_f = rest % 2;
        rest /= 2;
        int64_t agree1_f = rest % 2;
        rest /= 2;
        int64_t ubin_f = rest % U_BINS;
        int64_t cls = rest / U_BINS;
        int64_t x = flat % WIDTH;
        int64_t y = flat / WIDTH;
        int64_t classes[N_CAUSAL];
        int available[N_CAUSAL];
        for (int slot = 0; slot < N_CAUSAL; ++slot) {
            int64_t nx = x + CAUSAL_DX[slot];
            int64_t ny = y + CAUSAL_DY[slot];
            int inside = (nx >= 0) && (nx < WIDTH) && (ny >= 0) && (ny < HEIGHT);
            int64_t cy = clamp_i64(ny, 0, HEIGHT - 1);
            int64_t cx = clamp_i64(nx, 0, WIDTH - 1);
            int64_t neighbour = cy * WIDTH + cx;
            available[slot] = inside && self->known[neighbour];
            classes[slot] = available[slot] ? (int64_t)self->current[neighbour] : -1;
        }
        int64_t agreeing = 0;
        int any_available = 0;
        for (int slot = 0; slot < N_CAUSAL; ++slot) {
            if (available[slot]) {
                any_available = 1;
                if (classes[slot] == base_class) agreeing += 1;
            }
        }
        int64_t spatial4 =     any_available ? min_i64(agreeing + 1, SPATIAL4_LEVELS - 1) : 0;
        int present[NUM_CLASSES] = {0, 0, 0, 0, 0};
        for (int slot = 0; slot < N_CAUSAL; ++slot) {
            if (!available[slot]) continue;
            for (int value = 0; value < NUM_CLASSES; ++value) if (classes[slot] == value) present[value] = 1;
        }
        int64_t distinct = 0;
        for (int value = 0; value < NUM_CLASSES; ++value) distinct += present[value];
        int64_t homog = min_i64(distinct, HOMOGENEITY_LEVELS - 1);
        int64_t left = flat - 1 > 0 ? flat - 1 : 0;
        int64_t up = flat - WIDTH > 0 ? flat - WIDTH : 0;
        int has_left = (x > 0) && self->known[left];
        int has_up = (y > 0) && self->known[up];
        int agree_left = has_left && ((int64_t)self->current[left] == base_class);
        int agree_up = has_up && ((int64_t)self->current[up] == base_class);
        int64_t available2 = (int64_t)has_left + (int64_t)has_up;
        int64_t agreeing2 = (int64_t)agree_left + (int64_t)agree_up;
        int64_t spatial = (available2 == 0) ? 0 : agreeing2 + 1;
        int64_t groupbin8 = (((x % 64) + 2 * (y % 64)) * 8) / 190;
        int64_t patch192 = (y / 32) * (WIDTH / 32) + (x / 32);
        int64_t tile48_groupbin8 = tile_context(x, y);
        for (int pos = 0; pos < N_FAMILIES; ++pos) {
            self->fam_index[(int64_t)pos * self->capacity + i] =
                family_rule_index(self->families[pos].rule, cls, ubin_f, agree1_f, agree2_f,
                                  run_f, boundary_f, spatial, spatial4, homog, groupbin8,
                                  patch192, tile48_groupbin8);
        }
        int64_t ubin8 = min_i64(ubin_f >> 3, 7);
        int64_t mixer_head =     ((cls * BOUNDARY_LEVELS + boundary_f) * 4 + agree1_f * 2 + agree2_f);
        self->mixer[i] = (mixer_head * HOMOGENEITY_LEVELS + homog) * 8 + ubin8;
        int64_t nb[N_CAUSAL];
        for (int slot = 0; slot < N_CAUSAL; ++slot) nb[slot] = available[slot] ? classes[slot] : UNKNOWN;
        int64_t prev1_value = self->have_prev ? (int64_t)self->prev1[flat] : UNKNOWN;
        int64_t miss_head =     (nb[SLOT_UP] * MISS_BASE + nb[SLOT_UPRIGHT]) * MISS_BASE + nb[SLOT_LEFT];
        self->miss_cell[i] = miss_head * MISS_BASE + prev1_value;
    }
    self->group_open = 1;
    return 0;
}
static inline void miss_multiplier(const Corrector *self, int64_t cell, double *out) {
    if (self->miss_seen[cell] >= MISS_MIN_COUNT) {
        const int64_t *counts = &self->miss_counts[cell * NUM_CLASSES];
        const int64_t *expect = &self->miss_expect[cell * NUM_CLASSES];
        for (int k = 0; k < NUM_CLASSES; ++k) {
            double ratio = ((double)counts[k] + MISS_KT_ALPHA) /
                           ((double)expect[k] / PHAT_SCALE + MISS_KT_ALPHA);
            out[k] = clamp_double(ratio, MISS_CLAMP_LOW, MISS_CLAMP_HIGH);
        }
        } else for (int k = 0; k < NUM_CLASSES; ++k) out[k] = 1.0;
}
int corrector_coding_row(void *handle, float *output, int64_t n) {
    Corrector *self = (Corrector *)handle;
    if (!self || !self->group_open || n != self->n) return -1;
    for (int64_t i = 0; i < n; ++i) {
        double blended = 1.0;
        int64_t mixer_index = self->mixer[i];
        int64_t weight_index = mixer_index;
        const int64_t *weight_row = &self->weights[weight_index * N_FAMILIES];
        for (int pos = 0; pos < N_FAMILIES; ++pos) {
            const Family *family = &self->families[pos];
            int64_t index = self->fam_index[(int64_t)pos * self->capacity + i];
            double multiplier = family_multiplier(family, index);
            int64_t grid_weight = round_shift(weight_row[pos], STORE_SHIFT);
            double radicals[POWER_BITS];
            double root = multiplier;
            for (int r = 0; r < POWER_BITS; ++r) {
                root = sqrt(root);
                radicals[r] = root;
            }
            self->stretch[(int64_t)pos * self->capacity + i] =
                (radicals[POWER_BITS - 1] - 1.0) * (double)(1 << POWER_BITS);
            blended = blended * dyadic_power(multiplier, radicals, grid_weight);
        }
        blended = clamp_double(blended, ODDS_LOW, ODDS_HIGH);
        self->blended[i] = blended;
        double p_max = self->p_max[i];
        double one_minus = self->one_minus[i];
        double shifted = p_max * blended;
        double q = clamp_double(shifted / (shifted + one_minus), PROB_EPS, 1.0 - PROB_EPS);
        self->q[i] = q;
        const double *row = &self->row64[i * NUM_CLASSES];
        double *out = &self->out64[i * NUM_CLASSES];
        int64_t arg = self->arg[i];
        if (blended != 1.0) {
            double scale = (1.0 - q) / one_minus;
            for (int c = 0; c < NUM_CLASSES; ++c) out[c] = row[c] * scale;
            out[arg] = q;
        } else for (int c = 0; c < NUM_CLASSES; ++c) out[c] = row[c];
        float narrowed[NUM_CLASSES];
        for (int c = 0; c < NUM_CLASSES; ++c) narrowed[c] = (float)out[c];
        double m[NUM_CLASSES];
        miss_multiplier(self, self->miss_cell[i], m);
        m[arg] = 1.0;
        int active = 0;
        for (int c = 0; c < NUM_CLASSES; ++c) if (m[c] != 1.0) active = 1;
        if (!active) {
            for (int c = 0; c < NUM_CLASSES; ++c) output[i * NUM_CLASSES + c] = narrowed[c];
            continue;
        }
        double row64b[NUM_CLASSES], weighted[NUM_CLASSES], base[NUM_CLASSES];
        for (int c = 0; c < NUM_CLASSES; ++c) {
            row64b[c] = (double)narrowed[c];
            weighted[c] = row64b[c] * m[c];
            base[c] = row64b[c];
        }
        weighted[arg] = 0.0;
        base[arg] = 0.0;
        double big_w = 0.0, big_s = 0.0;
        for (int lane = 0; lane < NUM_CLASSES; ++lane) {
            big_w += weighted[lane];
            big_s += base[lane];
        }
        if (!(big_w > 0.0) || !(big_s > 0.0)) {
            for (int c = 0; c < NUM_CLASSES; ++c) output[i * NUM_CLASSES + c] = narrowed[c];
            continue;
        }
        double scale2 = big_s / big_w;
        for (int c = 0; c < NUM_CLASSES; ++c) output[i * NUM_CLASSES + c] = (float)(weighted[c] * scale2);
        output[i * NUM_CLASSES + arg] = (float)row64b[arg];
    }
    return 0;
}
static inline void family_halve(Family *family, int64_t index) {
    if (!family->count_limit) return;
    for (int pass = 0; pass < COUNT_HALVING_PASSES; ++pass) {
        if (family->counts[index] <= family->count_limit) break;
        family->counts[index] >>= 1;
        family->hits[index] >>= 1;
        family->phat_q[index] >>= 1;
    }
}
int corrector_observe(void *handle, const int64_t *symbols, int64_t n) {
    Corrector *self = (Corrector *)handle;
    if (!self || !self->group_open || n != self->n) return -1;
    for (int64_t i = 0; i < n; ++i) {
        int64_t arg = self->arg[i];
        int64_t decoded = symbols[i];
        if (decoded == arg) continue;
        int64_t cell = self->miss_cell[i];
        const double *row = &self->row64[i * NUM_CLASSES];
        double one_minus = self->one_minus[i];
        int64_t *expect = &self->miss_expect[cell * NUM_CLASSES];
        for (int k = 0; k < NUM_CLASSES; ++k) {
            double relative = (k == arg) ? 0.0 : row[k] / one_minus;
            expect[k] += (int64_t)rint(relative * PHAT_SCALE);
        }
        self->miss_counts[cell * NUM_CLASSES + decoded] += 1;
        self->miss_seen[cell] += 1;
    }
    memset(self->ws_counts, 0, (size_t)N_WEIGHT_SETS * sizeof(int64_t));
    for (int64_t i = 0; i < n; ++i) {
        self->ws_counts[self->mixer[i]] += 1;
        int64_t hit = (symbols[i] == self->arg[i]) ? 1 : 0;
        self->hit[i] = hit;
        self->residual[i] = (int64_t)rint(((double)hit - self->q[i]) * ERR_SCALE);
    }
    for (int pos = 0; pos < N_FAMILIES; ++pos) {
        memset(self->ws_gradient, 0, (size_t)N_WEIGHT_SETS * sizeof(int64_t));
        const double *stretch = &self->stretch[(int64_t)pos * self->capacity];
        for (int64_t i = 0; i < n; ++i) {
            double quantised_d =         clamp_double(rint(stretch[i] * STRETCH_SCALE), -STRETCH_CLAMP, STRETCH_CLAMP);
            self->ws_gradient[self->mixer[i]] += self->residual[i] * (int64_t)quantised_d;
        }
        for (int64_t ws = 0; ws < N_WEIGHT_SETS; ++ws) {
            int64_t gradient = self->ws_gradient[ws];
            int64_t count = self->ws_counts[ws];
            gradient = (count > 0) ? floor_div_i64(gradient, count) : 0;
            int64_t step = round_shift(gradient, LR_SHIFT);
            int64_t *slot = &self->weights[ws * N_FAMILIES + pos];
            *slot = clamp_i64(*slot + step, WEIGHT_LOW, WEIGHT_HIGH);
        }
    }
    for (int pos = 0; pos < N_FAMILIES; ++pos) {
        Family *family = &self->families[pos];
        const int64_t *indices = &self->fam_index[(int64_t)pos * self->capacity];
        for (int64_t i = 0; i < n; ++i) {
            int64_t index = indices[i];
            family->counts[index] += 1;
            family->hits[index] += self->hit[i];
            family->phat_q[index] += self->p_max_q[i];
        }
        if (family->count_limit) for (int64_t i = 0; i < n; ++i) family_halve(family, indices[i]);
    }
    for (int64_t i = 0; i < n; ++i) {
        int64_t flat = self->flat[i];
        self->current[flat] = (uint8_t)symbols[i];
        self->known[flat] = 1;
    }
    self->group_open = 0;
    return 0;
}
int corrector_end_frame(void *handle, const uint8_t *tokens, int64_t size) {
    Corrector *self = (Corrector *)handle;
    if (!self || size != self->plane) return -1;
    if (self->have_prev) {
        for (int64_t i = 0; i < self->plane; ++i) {
            if (tokens[i] == self->prev1[i]) {
                int64_t next = self->run[i] + 1;
                self->run[i] = next < RUN_CAP ? next : RUN_CAP;
        } else self->run[i] = 0;
        }
        memcpy(self->prev2, self->prev1, (size_t)self->plane * sizeof(uint8_t));
    }
    memcpy(self->prev1, tokens, (size_t)self->plane * sizeof(uint8_t));
    self->have_prev = 1;
    return 0;
}
/* Fixed-point log2 for the mixer's inputs: frexp splits off the exponent, then eleven
 * squarings read the mantissa's log2 one bit at a time. Same ordered float64 operations
 * as everything above, so the result does not depend on the compiler's scheduling.
 */
void mixer_log2(const double *values, int16_t *out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        int exponent;
        double work = frexp(values[i], &exponent) * 2.0;
        int64_t fraction = 0;
        for (int bit = 0; bit < 11; ++bit) {
            work = work * work;
            int high = work >= 2.0;
            if (high) work = work * 0.5;
            fraction = fraction * 2 + high;
        }
        out[i] = (int16_t)floor_div_i64((exponent - 1) * 2048 + fraction + 1, 2);
    }
}
void mixer_probability(const int64_t *freq, const int16_t *phi, const int8_t *weights,
                       const double *power, float *out, int64_t n, int features) {
    for (int64_t i = 0; i < n; ++i) {
        const int64_t *f = freq + i * 5;
        int winner = 0;
        for (int k = 1; k < 5; ++k) if (f[k] > f[winner]) winner = k;
        int64_t exponents[5], maximum = INT64_MIN;
        for (int k = 0; k < 5; ++k) {
            int64_t value = 0;
            for (int j = 0; j < features; ++j)
                value += (int64_t)phi[(i * 5 + k) * features + j] * weights[winner * features + j];
            exponents[k] = value;
            if (value > maximum) maximum = value;
        }
        double raw[5], sum = 0.0;
        for (int k = 0; k < 5; ++k) {
            int64_t code = exponents[k] - maximum, integer = floor_div_i64(code, 32768);
            raw[k] = (double)f[k] / 2147483648.0 * ldexp(power[code - integer * 32768], (int)integer);
            sum += raw[k];
        }
        for (int k = 0; k < 5; ++k) {
            double value = raw[k] / sum;
            out[i * 5 + k] = (float)(value < 0x1p-126 ? 0x1p-126 : value);
        }
    }
}
