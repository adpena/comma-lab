# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import ddm_tk1_semantic_stream_race as tk1


def _labels() -> np.ndarray:
    rng = np.random.default_rng(20260806)
    base = rng.integers(0, tk1.LEVELS, size=(1, 8, 8), dtype=np.uint8)
    labels = np.repeat(base, 6, axis=0)
    noise = rng.integers(0, tk1.LEVELS, size=labels.shape, dtype=np.uint8)
    return np.where(rng.random(labels.shape) < 0.10, noise, labels).astype(np.uint8)


def test_patch_group_order_is_causal_for_left_up_ul() -> None:
    receipt = tk1.patch_group_causality_receipt(height=8, width=8, patch=4, delta=2)

    assert receipt["causal_for_left_up_ul"] is True
    assert receipt["steps_per_patch"] == 10
    assert receipt["violations"] == []


def test_tk1_learned_prior_frame_roundtrips_small_labels() -> None:
    labels = _labels()
    rows = tk1.context_rows_for_mode("prev_left_up_ul")
    model, info = tk1.train_label_context_table(
        labels, mode="prev_left_up_ul", context_rows=rows, patch=4
    )
    stream = tk1.encode_labels_with_model(
        labels, model, mode="prev_left_up_ul", context_rows=rows, patch=4
    )
    frame = tk1.build_tk1_frame(
        labels, model, stream, mode="prev_left_up_ul", context_rows=rows, patch=4
    )

    restored = tk1.decode_tk1_frame(frame, verify_canonical=True)

    assert info["model_raw_bytes"] == rows * tk1.LEVELS
    assert np.array_equal(restored, labels)


def test_context_training_estimate_prefers_temporal_context_on_copy() -> None:
    labels = _labels()
    prev_rows = tk1.context_rows_for_mode("prev")
    model_prev, _ = tk1.train_label_context_table(labels, mode="prev", context_rows=prev_rows, patch=4)
    prev_bits = tk1.estimated_static_model_bits(
        labels, model_prev, mode="prev", context_rows=prev_rows, patch=4
    )

    rich_rows = tk1.context_rows_for_mode("prev_left_up")
    model_rich, _ = tk1.train_label_context_table(
        labels, mode="prev_left_up", context_rows=rich_rows, patch=4
    )
    rich_bits = tk1.estimated_static_model_bits(
        labels, model_rich, mode="prev_left_up", context_rows=rich_rows, patch=4
    )

    assert rich_bits <= prev_bits


def test_verify_batch_digests_accepts_matching_digest(tmp_path: Path) -> None:
    labels = _labels()
    digest = tk1.sha256_bytes(np.ascontiguousarray(labels[0:2]).tobytes())
    (tmp_path / "batch_0000_0002.json").write_text(
        json.dumps({"pair_range": [0, 2], "cells_sha256": digest})
    )

    receipt = tk1.verify_batch_digests(labels, tmp_path)

    assert receipt["all_pass"] is True
    assert receipt["batch_count"] == 1
