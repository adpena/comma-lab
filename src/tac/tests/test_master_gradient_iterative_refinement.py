# SPDX-License-Identifier: MIT
"""Tests for tac.master_gradient_iterative_refinement.

Sister of `tac.master_gradient_post_brotli_decompress` (item #3) and
`tac.master_gradient_pr101_mps_axis_probe` (item #4). This module covers the
"multiple passes and per byte deterministic corrections" methodology
(codex op7 iteration framework — operator standing directive 2026-05-19).
"""
from __future__ import annotations

from typing import Mapping

import pytest

from tac.master_gradient_iterative_refinement import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_PERTURBATION_DELTA,
    DEFAULT_TOP_K,
    IterativeRefinementError,
    IterativeRefinementPass,
    compute_prediction_vs_measurement_residuals,
    deterministic_top_k_byte_selection,
    summarize_pass_for_next_recommendation,
)


def _valid_pass_kwargs(**overrides):
    base = {
        "pass_index": 1,
        "master_gradient_anchor_archive_sha256": "a" * 64,
        "master_gradient_anchor_path": ".omx/state/master_gradient_anchors.jsonl",
        "mutation_grain": "post_brotli_decompress_decoder_weight_bytes",
        "perturbed_byte_indices": (10, 100, 1000),
        "perturbation_deltas": (1, 1, 1),
        "measured_score_deltas": {"contest_cpu": 0.001, "contest_cuda": 0.0009},
        "predicted_score_deltas": {"contest_cpu": 0.0005, "contest_cuda": 0.0005},
        "prediction_vs_measurement_residual": {
            "contest_cpu": 0.0005,
            "contest_cuda": 0.0004,
        },
        "measurement_axis": "[contest-CPU]",
        "measurement_hardware": "linux_x86_64_modal_cpu",
        "measurement_utc": "2026-05-19T13:00:00+00:00",
    }
    base.update(overrides)
    return base


def test_default_constants() -> None:
    assert DEFAULT_TOP_K == 10
    assert DEFAULT_PERTURBATION_DELTA == 1
    assert DEFAULT_LEARNING_RATE == 0.1


def test_pass_record_minimal_valid() -> None:
    rec = IterativeRefinementPass(**_valid_pass_kwargs())
    assert rec.pass_index == 1
    assert rec.perturbed_byte_indices == (10, 100, 1000)
    assert rec.perturbation_deltas == (1, 1, 1)


def test_pass_record_as_dict_roundtrip() -> None:
    rec = IterativeRefinementPass(**_valid_pass_kwargs())
    d = rec.as_dict()
    assert d["pass_index"] == 1
    assert d["measured_score_deltas"]["contest_cpu"] == pytest.approx(0.001)
    assert d["predicted_score_deltas"]["contest_cpu"] == pytest.approx(0.0005)
    assert d["mutation_grain"] == "post_brotli_decompress_decoder_weight_bytes"


def test_pass_rejects_zero_pass_index() -> None:
    with pytest.raises(IterativeRefinementError):
        IterativeRefinementPass(**_valid_pass_kwargs(pass_index=0))


def test_pass_rejects_negative_pass_index() -> None:
    with pytest.raises(IterativeRefinementError):
        IterativeRefinementPass(**_valid_pass_kwargs(pass_index=-1))


def test_pass_rejects_short_anchor_sha() -> None:
    with pytest.raises(IterativeRefinementError):
        IterativeRefinementPass(
            **_valid_pass_kwargs(master_gradient_anchor_archive_sha256="abc")
        )


def test_pass_rejects_empty_mutation_grain() -> None:
    with pytest.raises(IterativeRefinementError):
        IterativeRefinementPass(**_valid_pass_kwargs(mutation_grain=""))


def test_pass_rejects_mismatched_indices_and_deltas() -> None:
    with pytest.raises(IterativeRefinementError):
        IterativeRefinementPass(
            **_valid_pass_kwargs(
                perturbed_byte_indices=(10, 100, 1000), perturbation_deltas=(1, 1)
            )
        )


def test_pass_rejects_empty_perturbations() -> None:
    with pytest.raises(IterativeRefinementError):
        IterativeRefinementPass(
            **_valid_pass_kwargs(perturbed_byte_indices=(), perturbation_deltas=())
        )


def test_pass_rejects_out_of_order_byte_indices() -> None:
    # Deterministic invariant: indices must be strictly ascending
    with pytest.raises(IterativeRefinementError):
        IterativeRefinementPass(
            **_valid_pass_kwargs(
                perturbed_byte_indices=(100, 10, 1000), perturbation_deltas=(1, 1, 1)
            )
        )


def test_pass_rejects_duplicate_byte_indices() -> None:
    with pytest.raises(IterativeRefinementError):
        IterativeRefinementPass(
            **_valid_pass_kwargs(
                perturbed_byte_indices=(10, 10, 1000), perturbation_deltas=(1, 1, 1)
            )
        )


def test_pass_rejects_negative_byte_index() -> None:
    with pytest.raises(IterativeRefinementError):
        IterativeRefinementPass(
            **_valid_pass_kwargs(
                perturbed_byte_indices=(-1, 10), perturbation_deltas=(1, 1)
            )
        )


def test_pass_rejects_unlabeled_measurement_axis() -> None:
    with pytest.raises(IterativeRefinementError):
        IterativeRefinementPass(
            **_valid_pass_kwargs(measurement_axis="contest-CPU")  # missing []
        )


def test_pass_accepts_predicted_axis() -> None:
    rec = IterativeRefinementPass(
        **_valid_pass_kwargs(measurement_axis="[predicted]")
    )
    assert rec.measurement_axis == "[predicted]"


def test_pass_accepts_mps_axis() -> None:
    rec = IterativeRefinementPass(
        **_valid_pass_kwargs(measurement_axis="[MPS-research-signal]")
    )
    assert rec.measurement_axis == "[MPS-research-signal]"


def test_pass_rejects_non_numeric_delta() -> None:
    with pytest.raises(IterativeRefinementError):
        IterativeRefinementPass(
            **_valid_pass_kwargs(
                measured_score_deltas={"contest_cpu": "0.001"}  # type: ignore[dict-item]
            )
        )


def test_compute_residuals_simple() -> None:
    measured = {"contest_cpu": 0.005, "contest_cuda": 0.003}
    predicted = {"contest_cpu": 0.002, "contest_cuda": 0.001}
    r = compute_prediction_vs_measurement_residuals(measured, predicted)
    assert r["contest_cpu"] == pytest.approx(0.003)
    assert r["contest_cuda"] == pytest.approx(0.002)


def test_compute_residuals_missing_axis_treated_as_zero() -> None:
    measured = {"contest_cpu": 0.005}
    predicted = {"contest_cuda": 0.003}
    r = compute_prediction_vs_measurement_residuals(measured, predicted)
    assert r["contest_cpu"] == pytest.approx(0.005)  # measured - 0
    assert r["contest_cuda"] == pytest.approx(-0.003)  # 0 - predicted


def test_deterministic_top_k_byte_selection_basic() -> None:
    import numpy as np

    arr = np.array(
        [
            [0.1, 0.05, 0.0],
            [0.5, 0.2, 0.0],
            [0.01, 0.001, 0.0],
            [-0.3, 0.1, 0.0],
            [0.02, 0.02, 0.0],
        ],
        dtype=np.float32,
    )
    top3 = deterministic_top_k_byte_selection(arr, top_k=3)
    # Combined |seg|+|pose|: idx 0=0.15, 1=0.7, 2=0.011, 3=0.4, 4=0.04
    # Top 3 = {1, 3, 0}; sorted ascending = (0, 1, 3)
    assert top3 == (0, 1, 3)


def test_deterministic_top_k_byte_selection_seg_only() -> None:
    import numpy as np

    arr = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    top2 = deterministic_top_k_byte_selection(arr, top_k=2, rank_by="seg_abs")
    assert top2 == (1, 2)


def test_deterministic_top_k_byte_selection_pose_only() -> None:
    import numpy as np

    arr = np.array(
        [
            [0.0, 0.1, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.3, 0.0],
        ],
        dtype=np.float32,
    )
    top2 = deterministic_top_k_byte_selection(arr, top_k=2, rank_by="pose_abs")
    assert top2 == (1, 2)


def test_deterministic_top_k_repeatable() -> None:
    """Per the operator's "deterministic" requirement, two invocations
    on the same tensor with the same args MUST return the same indices."""
    import numpy as np

    arr = np.random.RandomState(42).randn(100, 3).astype(np.float32)
    a = deterministic_top_k_byte_selection(arr, top_k=10)
    b = deterministic_top_k_byte_selection(arr, top_k=10)
    assert a == b


def test_deterministic_top_k_rejects_invalid_rank_by() -> None:
    import numpy as np

    arr = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(IterativeRefinementError):
        deterministic_top_k_byte_selection(arr, top_k=3, rank_by="nonexistent")


def test_deterministic_top_k_rejects_wrong_shape() -> None:
    import numpy as np

    arr = np.zeros((5, 4), dtype=np.float32)
    with pytest.raises(IterativeRefinementError):
        deterministic_top_k_byte_selection(arr, top_k=3)


def test_deterministic_top_k_zero_k_rejected() -> None:
    import numpy as np

    arr = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(IterativeRefinementError):
        deterministic_top_k_byte_selection(arr, top_k=0)


def test_deterministic_top_k_empty_input() -> None:
    import numpy as np

    arr = np.zeros((0, 3), dtype=np.float32)
    result = deterministic_top_k_byte_selection(arr, top_k=10)
    assert result == ()


def test_summarize_large_residual_triggers_refinement() -> None:
    rec = IterativeRefinementPass(
        **_valid_pass_kwargs(
            measured_score_deltas={"contest_cpu": 0.005},
            predicted_score_deltas={"contest_cpu": 0.001},
            prediction_vs_measurement_residual={"contest_cpu": 0.004},
            mutation_grain="raw_archive_byte",
        )
    )
    recs = summarize_pass_for_next_recommendation(rec)
    assert any("refine master-gradient" in r.lower() for r in recs)
    assert any("under-estimated" in r.lower() for r in recs)
    # Raw-byte grain warning expected
    assert any("post-brotli" in r.lower() for r in recs)


def test_summarize_small_residual_validates_model() -> None:
    rec = IterativeRefinementPass(
        **_valid_pass_kwargs(
            measured_score_deltas={"contest_cpu": 0.0001},
            predicted_score_deltas={"contest_cpu": 0.0001},
            prediction_vs_measurement_residual={"contest_cpu": 0.0},
        )
    )
    recs = summarize_pass_for_next_recommendation(rec)
    assert any("validated" in r.lower() for r in recs)


def test_summarize_post_brotli_grain_no_cascade_warning() -> None:
    rec = IterativeRefinementPass(
        **_valid_pass_kwargs(
            mutation_grain="post_brotli_decompress_decoder_weight_bytes"
        )
    )
    recs = summarize_pass_for_next_recommendation(rec)
    # Should mention post-brotli grain validation
    assert any("post-brotli" in r.lower() and "byte-locality" in r.lower() for r in recs)


def test_summarize_negative_residual_marks_overestimate() -> None:
    rec = IterativeRefinementPass(
        **_valid_pass_kwargs(
            measured_score_deltas={"contest_cpu": 0.001},
            predicted_score_deltas={"contest_cpu": 0.005},
            prediction_vs_measurement_residual={"contest_cpu": -0.004},
        )
    )
    recs = summarize_pass_for_next_recommendation(rec)
    assert any("over-estimated" in r.lower() for r in recs)
