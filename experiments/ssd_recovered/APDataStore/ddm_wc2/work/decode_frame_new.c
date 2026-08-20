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
    if (decoder->geometry_ready) return;
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
    decoder->geometry_ready = 1;
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
static int f26_group_model(
    f26_rc64_decoder *decoder,
    const f26_hpac_model *model,
    int32_t group
) {
    int16_t *hidden = decoder->hidden_workspace;
    int16_t *b1 = decoder->b1_workspace;
    int16_t *logits = decoder->logits_workspace;
    int16_t *shift = decoder->shift_workspace;
    int16_t *past = decoder->past_workspace;
    int16_t *scale = decoder->scale_workspace;
    int16_t *spm = decoder->spm_workspace;
    const int32_t *conv_state = decoder->conv_state_workspace;
    int32_t h_start = model->group_h_offsets[group];
    int32_t h_count = model->group_h_offsets[group + 1] - h_start;
    int32_t b1_start = model->group_b1_offsets[group];
    int32_t b1_count = model->group_b1_offsets[group + 1] - b1_start;
    int32_t target_start = model->group_target_offsets[group];
    int32_t target_count = model->group_target_offsets[group + 1] - target_start;
    int32_t patch_index;
    int32_t local_index;
    int status = 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4) private(local_index)
#endif
    for (patch_index = 0; patch_index < model->patch_count; ++patch_index) {
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
            int64_t sums[64];
            int32_t channel;
            int32_t tap;
            for (channel = 0; channel < model->channels; ++channel) sums[channel] = 0;
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
                for (channel = 0; channel < model->channels; ++channel) {
                    sums[channel] += (int64_t)values[channel] * weights[channel];
                }
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
            int64_t sums[64];
            int32_t channel;
            int32_t symbol;
            int32_t tap;
            for (channel = 0; channel < model->channels; ++channel) sums[channel] = 0;
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
                for (channel = 0; channel < model->channels; ++channel) {
                    sums[channel] += (int64_t)values[channel] * weights[channel];
                }
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
    return status;
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
static int f26_group_apply(
    f26_rc64_decoder *decoder,
    const f26_hpac_model *model,
    int32_t group
) {
    const uint8_t *decoded_patch_major = decoder->decoded_workspace;
    int32_t *conv_state = decoder->conv_state_workspace;
    int32_t target_start = model->group_target_offsets[group];
    int32_t target_count = model->group_target_offsets[group + 1] - target_start;
    int32_t patch_index;
    int32_t local_index;
    int status = 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4) private(local_index)
#endif
    for (patch_index = 0; patch_index < model->patch_count; ++patch_index) {
        for (local_index = 0; local_index < target_count; ++local_index) {
            uint8_t class_id = decoded_patch_major[patch_index * target_count + local_index];
            int32_t source_row;
            int32_t source_col;
            int32_t tap;
            if (!class_id) continue;
            if (class_id >= F26_ALPHABET) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
                status = -8;
                continue;
            }
            source_row = model->target_positions[(target_start + local_index) * 2];
            source_col = model->target_positions[(target_start + local_index) * 2 + 1];
            for (tap = 0; tap < model->a_taps; ++tap) {
                int32_t hidden_row = source_row - model->a_offsets[tap * 2];
                int32_t hidden_col = source_col - model->a_offsets[tap * 2 + 1];
                int32_t channel;
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
                for (channel = 0; channel < model->channels; ++channel) {
                    destination[channel] += (int32_t)class_delta[channel];
                }
            }
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

