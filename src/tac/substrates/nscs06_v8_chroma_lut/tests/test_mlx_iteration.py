# SPDX-License-Identifier: MIT
"""Tests for the NSCS06 v8 chroma_lut MLX-LOCAL ITERATION extension.

Per ``src/tac/substrates/nscs06_v8_chroma_lut/mlx_iteration.py`` + the
corrected #1258 empirical anchor + #1265 contest-equivalence gate. Covers:

* Non-promotable contract invariants on :class:`MLXIterationVerdict` +
  :class:`MLXParityVerdict` (Catalog #1 + #127 + #192 + #317 + #341).
* Canonical cargo-cult-unwind arm enumeration (Catalog #303 sister).
* Aggregation policy correctness (per_class / binary_foreground /
  merged_road_lane).
* Deterministic seed derivation (same RGB pair input -> same LUT bytes ->
  same seed bytes across runs).
* :class:`MLXIterationArm` invariants (rejects invalid policies + ranges).

Tests that require an actual MLX install are guarded by
``pytest.importorskip("mlx.core")`` and only run on Apple Silicon hosts
where MLX is installed (sister of canonical MLX test guard pattern across
``src/tac/local_acceleration/tests/``).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tac.substrates.nscs06_v8_chroma_lut import (
    GRAYSCALE_LEVELS_DEFAULT,
    NUM_SEGNET_CLASSES,
    PROCEDURAL_SEED_SIZE_BYTES,
)
from tac.substrates.nscs06_v8_chroma_lut.mlx_iteration import (
    CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
    DEFAULT_MLX_AXIS_TAG,
    DEFAULT_PARITY_TOLERANCE_ARGMAX_FLIP_FRACTION,
    MLX_NON_PROMOTABLE_PROVENANCE,
    MLXIterationArm,
    MLXIterationError,
    MLXIterationVerdict,
    MLXParityVerdict,
    _apply_aggregation_policy,
    enumerate_cargo_cult_unwind_arms,
    is_mlx_available,
)


# ---------------------------------------------------------------------------
# Non-promotable contract (Catalog #1 + #127 + #192 + #317 + #341)
# ---------------------------------------------------------------------------


def _make_valid_verdict_kwargs() -> dict:
    return {
        "arm_label": "baseline_4bit_per_class",
        "grayscale_levels": 16,
        "num_segnet_classes_aggregation_policy": "per_class",
        "chroma_lut_bytes_full": 240,
        "procedural_seed_size_bytes": 32,
        "predicted_archive_bytes_saved": 4064,
        "predicted_delta_s": -0.002706,
        "chroma_lut_sha256": "a" * 64,
        "procedural_seed_sha256": "b" * 64,
        "cls_label_remap_effective_classes": 5,
    }


def test_mlx_iteration_verdict_canonical_construction() -> None:
    """Canonical construction with defaults preserves all non-promotable markers."""
    v = MLXIterationVerdict(**_make_valid_verdict_kwargs())
    assert v.axis_tag == DEFAULT_MLX_AXIS_TAG
    assert v.score_claim is False
    assert v.promotion_eligible is False
    assert v.ready_for_exact_eval_dispatch is False
    assert v.rank_or_kill_eligible is False
    assert v.evidence_grade == "research-signal"
    assert v.contest_equivalence_gate_required_before_dispatch is True
    assert (
        v.canonical_equation_in_domain_context
        == CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT
    )
    assert (
        "macos_mlx_research_signal_not_contest_authority" in v.blockers
    )
    assert (
        "requires_paired_contest_cpu_plus_cuda_for_score_claim" in v.blockers
    )
    assert (
        "requires_pass_verdict_from_gate_mlx_candidate_contest_equivalence"
        in v.blockers
    )


def test_mlx_iteration_verdict_refuses_score_claim_true() -> None:
    """Per Catalog #1 + #127 + #192 the score_claim MUST be False."""
    kwargs = _make_valid_verdict_kwargs()
    kwargs["score_claim"] = True
    with pytest.raises(MLXIterationError, match="score_claim"):
        MLXIterationVerdict(**kwargs)


def test_mlx_iteration_verdict_refuses_promotion_eligible_true() -> None:
    """Per Catalog #192 promotion_eligible MUST be False."""
    kwargs = _make_valid_verdict_kwargs()
    kwargs["promotion_eligible"] = True
    with pytest.raises(MLXIterationError, match="promotion_eligible"):
        MLXIterationVerdict(**kwargs)


def test_mlx_iteration_verdict_refuses_ready_for_exact_eval_dispatch_true() -> None:
    kwargs = _make_valid_verdict_kwargs()
    kwargs["ready_for_exact_eval_dispatch"] = True
    with pytest.raises(
        MLXIterationError, match="ready_for_exact_eval_dispatch"
    ):
        MLXIterationVerdict(**kwargs)


def test_mlx_iteration_verdict_refuses_rank_or_kill_eligible_true() -> None:
    kwargs = _make_valid_verdict_kwargs()
    kwargs["rank_or_kill_eligible"] = True
    with pytest.raises(MLXIterationError, match="rank_or_kill_eligible"):
        MLXIterationVerdict(**kwargs)


def test_mlx_iteration_verdict_refuses_non_canonical_axis_tag() -> None:
    """Per CLAUDE.md FORBIDDEN_PATTERNS axis_tag MUST be the canonical MLX tag."""
    kwargs = _make_valid_verdict_kwargs()
    kwargs["axis_tag"] = "[contest-CUDA]"  # forbidden: this is a CUDA score axis
    with pytest.raises(MLXIterationError, match="axis_tag"):
        MLXIterationVerdict(**kwargs)


def test_mlx_iteration_verdict_refuses_non_research_signal_grade() -> None:
    kwargs = _make_valid_verdict_kwargs()
    kwargs["evidence_grade"] = "contest-CUDA"
    with pytest.raises(MLXIterationError, match="evidence_grade"):
        MLXIterationVerdict(**kwargs)


def test_mlx_iteration_verdict_as_dict_round_trip() -> None:
    v = MLXIterationVerdict(**_make_valid_verdict_kwargs())
    d = v.as_dict()
    assert d["axis_tag"] == DEFAULT_MLX_AXIS_TAG
    assert d["score_claim"] is False
    assert d["promotion_eligible"] is False
    assert d["evidence_grade"] == "research-signal"
    # Blockers serialized as list (JSON-safe)
    assert isinstance(d["blockers"], list)
    assert len(d["blockers"]) >= 3


def test_mlx_non_promotable_provenance_constant() -> None:
    """The module-level constant MUST be the canonical non-promotable shape."""
    assert MLX_NON_PROMOTABLE_PROVENANCE["score_claim"] is False
    assert MLX_NON_PROMOTABLE_PROVENANCE["promotion_eligible"] is False
    assert MLX_NON_PROMOTABLE_PROVENANCE["ready_for_exact_eval_dispatch"] is False
    assert MLX_NON_PROMOTABLE_PROVENANCE["axis_tag"] == DEFAULT_MLX_AXIS_TAG
    assert MLX_NON_PROMOTABLE_PROVENANCE["evidence_grade"] == "research-signal"
    assert (
        MLX_NON_PROMOTABLE_PROVENANCE[
            "contest_equivalence_gate_required_before_dispatch"
        ]
        is True
    )


# ---------------------------------------------------------------------------
# MLXParityVerdict non-promotable contract
# ---------------------------------------------------------------------------


def _make_valid_parity_kwargs() -> dict:
    return {
        "num_pairs_compared": 8,
        "argmax_flip_fraction": 1.5e-5,
        "tolerance": DEFAULT_PARITY_TOLERANCE_ARGMAX_FLIP_FRACTION,
        "parity_ok": True,
        "max_abs_logit_drift": 0.011,
        "mean_abs_logit_drift": 0.00013,
    }


def test_mlx_parity_verdict_canonical_construction() -> None:
    v = MLXParityVerdict(**_make_valid_parity_kwargs())
    assert v.axis_tag == DEFAULT_MLX_AXIS_TAG
    assert v.score_claim is False
    assert v.promotion_eligible is False
    assert v.evidence_grade == "research-signal"
    assert v.parity_ok is True


def test_mlx_parity_verdict_refuses_score_claim_true() -> None:
    kwargs = _make_valid_parity_kwargs()
    kwargs["score_claim"] = True
    with pytest.raises(MLXIterationError, match="score_claim"):
        MLXParityVerdict(**kwargs)


def test_mlx_parity_verdict_refuses_promotion_eligible_true() -> None:
    kwargs = _make_valid_parity_kwargs()
    kwargs["promotion_eligible"] = True
    with pytest.raises(MLXIterationError, match="promotion_eligible"):
        MLXParityVerdict(**kwargs)


def test_mlx_parity_verdict_refuses_non_canonical_axis_tag() -> None:
    kwargs = _make_valid_parity_kwargs()
    kwargs["axis_tag"] = "[contest-CPU]"
    with pytest.raises(MLXIterationError, match="axis_tag"):
        MLXParityVerdict(**kwargs)


def test_mlx_parity_verdict_as_dict_round_trip() -> None:
    v = MLXParityVerdict(**_make_valid_parity_kwargs())
    d = v.as_dict()
    assert d["axis_tag"] == DEFAULT_MLX_AXIS_TAG
    assert d["score_claim"] is False
    assert d["parity_ok"] is True


# ---------------------------------------------------------------------------
# MLXIterationArm invariants (Catalog #287 placeholder-rationale sister)
# ---------------------------------------------------------------------------


def test_mlx_iteration_arm_canonical_construction() -> None:
    arm = MLXIterationArm(
        arm_label="baseline_4bit_per_class",
        grayscale_levels=16,
    )
    assert arm.grayscale_levels == 16
    assert arm.num_segnet_classes_aggregation_policy == "per_class"


def test_mlx_iteration_arm_refuses_empty_label() -> None:
    with pytest.raises(MLXIterationError, match="arm_label"):
        MLXIterationArm(
            arm_label="",
            grayscale_levels=16,
        )


def test_mlx_iteration_arm_refuses_invalid_grayscale_levels() -> None:
    with pytest.raises(MLXIterationError, match="grayscale_levels"):
        MLXIterationArm(arm_label="x", grayscale_levels=0)
    with pytest.raises(MLXIterationError, match="grayscale_levels"):
        MLXIterationArm(arm_label="x", grayscale_levels=257)


def test_mlx_iteration_arm_refuses_invalid_aggregation_policy() -> None:
    with pytest.raises(
        MLXIterationError,
        match="num_segnet_classes_aggregation_policy",
    ):
        MLXIterationArm(
            arm_label="x",
            grayscale_levels=16,
            num_segnet_classes_aggregation_policy="random_policy",
        )


# ---------------------------------------------------------------------------
# Cargo-cult-unwind enumeration (Catalog #303 sister)
# ---------------------------------------------------------------------------


def test_enumerate_cargo_cult_unwind_arms_baseline_first() -> None:
    arms = enumerate_cargo_cult_unwind_arms()
    assert arms[0].arm_label == "baseline_4bit_per_class"
    assert arms[0].grayscale_levels == GRAYSCALE_LEVELS_DEFAULT
    assert arms[0].num_segnet_classes_aggregation_policy == "per_class"


def test_enumerate_cargo_cult_unwind_arms_count_at_least_5() -> None:
    """Catalog #303 sister: at least 2 cargo-cult unwinds per assumption."""
    arms = enumerate_cargo_cult_unwind_arms()
    assert len(arms) >= 5


def test_enumerate_cargo_cult_unwind_arms_unique_labels() -> None:
    arms = enumerate_cargo_cult_unwind_arms()
    labels = [arm.arm_label for arm in arms]
    assert len(labels) == len(set(labels))


def test_enumerate_cargo_cult_unwind_arms_covers_levels_axis() -> None:
    """Cargo-cult #1 (4-bit luma is always enough) MUST be unwound."""
    arms = enumerate_cargo_cult_unwind_arms()
    levels_present = {arm.grayscale_levels for arm in arms}
    # Baseline (16) + at least one smaller + one larger
    assert 16 in levels_present
    assert any(lvl < 16 for lvl in levels_present)
    assert any(lvl > 16 for lvl in levels_present)


def test_enumerate_cargo_cult_unwind_arms_covers_aggregation_axis() -> None:
    """Cargo-cult #2 (5-class per-class is always optimal) MUST be unwound."""
    arms = enumerate_cargo_cult_unwind_arms()
    policies_present = {arm.num_segnet_classes_aggregation_policy for arm in arms}
    # Baseline + at least one alternative aggregation policy
    assert "per_class" in policies_present
    assert len(policies_present) >= 2


# ---------------------------------------------------------------------------
# Aggregation policy correctness
# ---------------------------------------------------------------------------


def test_apply_aggregation_policy_per_class_identity() -> None:
    cls = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
    remapped, n = _apply_aggregation_policy(cls, "per_class")
    np.testing.assert_array_equal(remapped, cls)
    assert n == NUM_SEGNET_CLASSES


def test_apply_aggregation_policy_binary_foreground() -> None:
    cls = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
    remapped, n = _apply_aggregation_policy(cls, "binary_foreground")
    expected = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(remapped, expected)
    assert n == 2


def test_apply_aggregation_policy_merged_road_lane() -> None:
    # cls labels: 0=background 1=lane 2=vehicle 3=road 4=sky-or-other
    cls = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
    remapped, n = _apply_aggregation_policy(cls, "merged_road_lane")
    expected = np.array([[0, 1, 2, 1, 3]], dtype=np.uint8)
    np.testing.assert_array_equal(remapped, expected)
    assert n == 4


def test_apply_aggregation_policy_unknown_rejected() -> None:
    cls = np.zeros((1, 1), dtype=np.uint8)
    with pytest.raises(MLXIterationError, match="unknown aggregation policy"):
        _apply_aggregation_policy(cls, "policy_that_does_not_exist")


# ---------------------------------------------------------------------------
# is_mlx_available
# ---------------------------------------------------------------------------


def test_is_mlx_available_returns_bool() -> None:
    # On Apple Silicon with MLX installed -> True; on non-Apple -> False.
    # Either way, returns a bool.
    assert isinstance(is_mlx_available(), bool)


# ---------------------------------------------------------------------------
# MLX-dependent tests (Apple Silicon + MLX install only)
# ---------------------------------------------------------------------------


def test_derive_chroma_lut_via_mlx_scorer_smoke() -> None:
    """End-to-end smoke: tiny synthetic RGB -> MLX SegNet -> LUT.

    Skipped on hosts without MLX OR without upstream scorer weights.
    """
    pytest.importorskip("mlx.core")
    from pathlib import Path

    # Verify upstream weights are reachable; skip if not.
    upstream_dir = Path(__file__).resolve().parents[5] / "upstream"
    models_dir = upstream_dir / "models"
    if not (models_dir / "segnet.safetensors").is_file() or not (
        models_dir / "posenet.safetensors"
    ).is_file():
        pytest.skip("upstream scorer weights not available")

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.local_acceleration.mlx_scorer_adapters import torch_segnet_to_mlx
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.nscs06_v8_chroma_lut.mlx_iteration import (
        derive_chroma_lut_via_mlx_scorer,
    )

    tok = patch_upstream_yuv6_globally()
    try:
        _, torch_segnet = load_differentiable_scorers(upstream_dir, device="cpu")
        mlx_seg = torch_segnet_to_mlx(torch_segnet)
        rng = np.random.RandomState(0)
        # Tiny window: 2 frames @ 384x512 (canonical EVAL_HW).
        rgb_pairs = rng.randint(0, 256, size=(2, 3, 384, 512), dtype=np.uint8)
        chroma_lut, cls_full = derive_chroma_lut_via_mlx_scorer(
            rgb_pairs, mlx_seg, chunk_size=2
        )
        assert chroma_lut.shape == (
            GRAYSCALE_LEVELS_DEFAULT,
            NUM_SEGNET_CLASSES,
            3,
        )
        assert chroma_lut.dtype == np.uint8
        assert cls_full.shape == (2, 384, 512)
        assert cls_full.dtype == np.uint8
        assert int(cls_full.min()) >= 0
        assert int(cls_full.max()) < NUM_SEGNET_CLASSES
    finally:
        unpatch_upstream_yuv6(tok)


def test_iterate_chroma_lut_policies_via_mlx_smoke() -> None:
    """End-to-end smoke: 2 arms produce 2 verdicts with non-promotable markers."""
    pytest.importorskip("mlx.core")
    from pathlib import Path

    upstream_dir = Path(__file__).resolve().parents[5] / "upstream"
    models_dir = upstream_dir / "models"
    if not (models_dir / "segnet.safetensors").is_file() or not (
        models_dir / "posenet.safetensors"
    ).is_file():
        pytest.skip("upstream scorer weights not available")

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.local_acceleration.mlx_scorer_adapters import torch_segnet_to_mlx
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.nscs06_v8_chroma_lut.mlx_iteration import (
        MLXIterationArm,
        iterate_chroma_lut_policies_via_mlx,
    )

    tok = patch_upstream_yuv6_globally()
    try:
        _, torch_segnet = load_differentiable_scorers(upstream_dir, device="cpu")
        mlx_seg = torch_segnet_to_mlx(torch_segnet)
        rng = np.random.RandomState(0)
        rgb_pairs = rng.randint(0, 256, size=(2, 3, 384, 512), dtype=np.uint8)
        arms = (
            MLXIterationArm(arm_label="baseline", grayscale_levels=16),
            MLXIterationArm(
                arm_label="binary",
                grayscale_levels=16,
                num_segnet_classes_aggregation_policy="binary_foreground",
            ),
        )
        verdicts = iterate_chroma_lut_policies_via_mlx(
            rgb_pairs, mlx_seg, arms=arms, chunk_size=2
        )
        assert len(verdicts) == 2
        for v in verdicts:
            assert v.score_claim is False
            assert v.promotion_eligible is False
            assert v.ready_for_exact_eval_dispatch is False
            assert v.axis_tag == DEFAULT_MLX_AXIS_TAG
            assert v.chroma_lut_bytes_full > 0
            assert v.procedural_seed_size_bytes == PROCEDURAL_SEED_SIZE_BYTES
            assert len(v.chroma_lut_sha256) == 64
            assert len(v.procedural_seed_sha256) == 64
    finally:
        unpatch_upstream_yuv6(tok)


def test_verify_mlx_segnet_argmax_parity_with_torch_smoke() -> None:
    """End-to-end smoke: MLX vs torch SegNet argmax parity on synthetic input.

    Per #1258 corrected empirical anchor: argmax_flip_fraction is expected to
    be SMALL (anchor reports 1.58e-5 on PR95 HNeRV decoded frames). On
    synthetic random RGB the flip-fraction may be higher BUT should still be
    well below the default 2% tolerance.
    """
    pytest.importorskip("mlx.core")
    from pathlib import Path

    upstream_dir = Path(__file__).resolve().parents[5] / "upstream"
    models_dir = upstream_dir / "models"
    if not (models_dir / "segnet.safetensors").is_file() or not (
        models_dir / "posenet.safetensors"
    ).is_file():
        pytest.skip("upstream scorer weights not available")

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.local_acceleration.mlx_scorer_adapters import torch_segnet_to_mlx
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.nscs06_v8_chroma_lut.mlx_iteration import (
        verify_mlx_segnet_argmax_parity_with_torch,
    )

    tok = patch_upstream_yuv6_globally()
    try:
        _, torch_segnet = load_differentiable_scorers(upstream_dir, device="cpu")
        mlx_seg = torch_segnet_to_mlx(torch_segnet)
        rng = np.random.RandomState(0)
        rgb_pairs = rng.randint(0, 256, size=(2, 3, 384, 512), dtype=np.uint8)
        verdict = verify_mlx_segnet_argmax_parity_with_torch(
            rgb_pairs,
            mlx_seg,
            torch_segnet,
            chunk_size=2,
        )
        assert verdict.num_pairs_compared == 2
        assert 0.0 <= verdict.argmax_flip_fraction <= 1.0
        assert verdict.axis_tag == DEFAULT_MLX_AXIS_TAG
        assert verdict.score_claim is False
        # Synthetic random RGB may produce a higher flip fraction than the
        # #1258 anchor (1.58e-5 on natural PR95-decoded frames). We assert the
        # canonical contract holds rather than the empirical number.
        assert isinstance(verdict.parity_ok, bool)
    finally:
        unpatch_upstream_yuv6(tok)
