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
    int16_t *hidden = NULL;
    int16_t *b1 = NULL;
    int16_t *logits = NULL;
    int16_t *shift = NULL;
    int16_t *past = NULL;
    int16_t *scale = NULL;
    int16_t *spm = NULL;
    int32_t *conv_state = NULL;
    uint8_t *decoded_patch_major = NULL;
    size_t hidden_capacity;
    size_t b1_capacity;
    size_t logits_capacity;
    size_t compact_context_capacity;
    size_t past_capacity;
    size_t conv_state_capacity;
    int32_t max_h = 0;
    int32_t max_b1 = 0;
    int32_t max_targets = 0;
    int32_t group;
    int status = 0;
    double timing_started;

    if (
        !decoder || !model || !boundary || !previous || !current || !corrected_trace ||
        !probability_trace
    ) return -1;
    if (
        model->channels != 64 || model->patch_count != model->patch_rows * model->patch_cols ||
        model->height != model->patch_rows * model->patch ||
        model->width != model->patch_cols * model->patch || model->groups <= 0 ||
        !model->conv_a_delta
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
    compact_context_capacity = (size_t)model->patch_count * (size_t)model->channels;
    past_capacity = (size_t)model->patch_count * (size_t)model->patch *
        (size_t)model->patch * (size_t)model->channels;
    conv_state_capacity = (size_t)model->patch_count * (size_t)model->patch *
        (size_t)model->patch * (size_t)model->channels;
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
        decoder->decoded_workspace = (uint8_t *)malloc(
            (size_t)model->patch_count * (size_t)max_targets
        );
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
        decoder->decoded_capacity = (size_t)model->patch_count * (size_t)max_targets;
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
        decoder->decoded_capacity < (size_t)model->patch_count * (size_t)max_targets
    ) {
        status = -4;
        goto cleanup;
    }
    hidden = decoder->hidden_workspace;
    b1 = decoder->b1_workspace;
    logits = decoder->logits_workspace;
    shift = decoder->shift_workspace;
    past = decoder->past_workspace;
    scale = decoder->scale_workspace;
    spm = decoder->spm_workspace;
    conv_state = decoder->conv_state_workspace;
    decoded_patch_major = decoder->decoded_workspace;

    memset(current, 0, (size_t)model->height * (size_t)model->width);
    memset(f26_last_timing, 0, sizeof(f26_last_timing));
    timing_started = f26_now_seconds();
    status = f26_prepare_frame_context(decoder, model, frame_index, previous);
    if (status) {
        status -= 40;
        goto cleanup;
    }
    f26_last_timing[0] = f26_now_seconds() - timing_started;
    timing_started = f26_now_seconds();
    for (group = 0; group < model->patch_count; ++group) {
        memcpy(
            conv_state + (size_t)group * (size_t)model->patch * (size_t)model->patch *
                (size_t)model->channels,
            model->conv_a_initial,
            (size_t)model->patch * (size_t)model->patch * (size_t)model->channels *
            sizeof(int32_t)
        );
    }
    f26_last_timing[1] = f26_now_seconds() - timing_started;
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
        timing_started = f26_now_seconds();
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
        if (status) goto cleanup;
        f26_last_timing[2] += f26_now_seconds() - timing_started;

        timing_started = f26_now_seconds();
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
            decoded_patch_major[patch_major] = decoded;
        }
        f26_last_timing[3] += f26_now_seconds() - timing_started;
        /* Apply this group's decoded symbols to the incremental conv-A state. */
        timing_started = f26_now_seconds();
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
        if (status) goto cleanup;
        f26_last_timing[4] += f26_now_seconds() - timing_started;
    }

cleanup:
    return status;
}

