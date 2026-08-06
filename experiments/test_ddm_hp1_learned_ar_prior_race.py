# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from experiments import ddm_hp1_learned_ar_prior_race as hp1


def _temporal_copy_lattice() -> np.ndarray:
    rng = np.random.default_rng(20260806)
    base = rng.integers(0, hp1.LEVELS, size=(1, 4, 5, 2), dtype=np.uint8)
    codes = np.repeat(base, 12, axis=0)
    noise = rng.integers(0, hp1.LEVELS, size=codes.shape, dtype=np.uint8)
    mask = rng.random(codes.shape) < 0.07
    return np.where(mask, noise, codes).astype(np.uint8)


def test_hp1_learned_prior_frame_roundtrips_small_lattice() -> None:
    codes = _temporal_copy_lattice()
    model, info = hp1.train_context_table(codes, context_rows=41, patch=2, mode="hash_prev_spatial")
    stream = hp1.encode_tokens_with_model(codes, model, context_rows=41, patch=2, mode="hash_prev_spatial")
    frame = hp1.build_hp1_frame(codes, model, stream, context_rows=41, patch=2, mode="hash_prev_spatial")

    restored = hp1.decode_hp1_frame(frame, verify_canonical=True)

    assert info["model_raw_bytes"] == 41 * hp1.LEVELS
    assert np.array_equal(restored, codes)


def test_structured_context_mode_row_count_and_roundtrip() -> None:
    codes = _temporal_copy_lattice()
    rows = hp1.context_rows_for_mode("prev_up", channels=codes.shape[3])
    model, info = hp1.train_context_table(codes, context_rows=rows, patch=2, mode="prev_up")
    stream = hp1.encode_tokens_with_model(codes, model, context_rows=rows, patch=2, mode="prev_up")
    frame = hp1.build_hp1_frame(codes, model, stream, context_rows=rows, patch=2, mode="prev_up")

    restored = hp1.decode_hp1_frame(frame)

    assert rows == 17 * 17
    assert info["model_raw_bytes"] == rows * hp1.LEVELS
    assert np.array_equal(restored, codes)


def test_entropy_baselines_show_prev_pair_conditioning_on_temporal_copy() -> None:
    codes = _temporal_copy_lattice()

    entropy = hp1.entropy_baselines(codes)

    assert entropy["prev_pair_conditioned_same_cell"]["bits_per_symbol"] < entropy["order0"]["bits_per_symbol"]
    assert entropy["prev_pair_plus_spatial_context"]["bits_per_symbol"] <= entropy["spatial_context_left_up_channelprev"]["bits_per_symbol"]


def test_forced_lzma_ix2_frame_decodes_to_input() -> None:
    codes = _temporal_copy_lattice()

    frame = hp1.forced_lzma_ix2_token_frame(codes)
    restored = hp1.ix2.decode_token_frame(frame)

    assert np.array_equal(restored, codes)
