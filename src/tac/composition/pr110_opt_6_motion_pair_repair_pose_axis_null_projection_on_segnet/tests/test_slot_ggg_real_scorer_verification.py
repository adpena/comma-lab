# SPDX-License-Identifier: MIT
"""Slot GGG Part 3 — REAL scorer-verification dedicated tests.

Per Slot EEE 6-axis honesty audit 2026-05-29 op-routable #1:

    "Slot RR — rename apply_pose_axis_null_projection_to_pr110_archive ->
    build_pose_axis_null_projection_menu_for_pr110_archive OR implement
    actual frame perturbation. Add REAL behavioral test that exercises
    one menu mode against a sample frame."

Predecessor commits 30bf9029f + 32a70c051 landed the rename + REAL
pixel-level perturbation. Slot GGG closes the remaining gap (Axis F
cite-vs-impl): the function name asserts a SCORER-axis claim
("pose-axis null-projection ON SEGNET") which the predecessor's pixel-
level apply did not verify. THIS test file exercises the canonical
sister :func:`apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive`
against real upstream/videos/0.mkv frame pairs using real PoseNet +
SegNet via the canonical helpers, and asserts on the EMPIRICAL behavior
(not just constants).

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192: every assertion
in this file is at the ``[macOS-CPU advisory]`` axis and is NEVER
promotable to a contest-axis score claim. The tests verify Python
runtime correctness + per-mode CONFIRMED / FALSIFIED disambiguator
behavior. Paired Linux x86_64 + NVIDIA empirical anchor required per
Catalog #246 before any contest-axis score claim.

Tests use small num_pairs / max_modes_to_verify defaults so they stay
cheap (typical end-to-end is 5-15s on macOS-CPU). Heavy tests are
marked ``@pytest.mark.slow`` and skipped by default.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Per CLAUDE.md upstream/scorer load discipline: the upstream models must
# exist locally for these tests to run. If they don't, skip cleanly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
_UPSTREAM_POSENET = os.path.join(_REPO_ROOT, "upstream", "models", "posenet.safetensors")
_UPSTREAM_SEGNET = os.path.join(_REPO_ROOT, "upstream", "models", "segnet.safetensors")
_UPSTREAM_VIDEO = os.path.join(_REPO_ROOT, "upstream", "videos", "0.mkv")

pytestmark = pytest.mark.skipif(
    not (
        os.path.exists(_UPSTREAM_POSENET)
        and os.path.exists(_UPSTREAM_SEGNET)
        and os.path.exists(_UPSTREAM_VIDEO)
    ),
    reason=(
        "Upstream models or video not available locally; "
        "Slot GGG real-scorer verification requires upstream/models/*.safetensors "
        "and upstream/videos/0.mkv per Catalog #213"
    ),
)


from tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet import (
    POSENET_NULL_CARRIER_BAND_LOWER,
    POSENET_NULL_CARRIER_BAND_UPPER,
    SEGNET_ARGMAX_NULL_TOLERANCE,
    VERDICT_NULL_PROJECTION_CONFIRMED_PER_MODE,
    VERDICT_NULL_PROJECTION_FALSIFIED_PER_MODE,
    PoseAxisNullProjectionStrategy,
    apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive,
)


# Cheap smoke configuration: small pair count + resolution + mode cap so
# the test suite stays under ~20s wall-clock on macOS-CPU.
SMOKE_NUM_PAIRS = 2
SMOKE_RESOLUTION_HW = (48, 64)
SMOKE_MAX_MODES = 2


# ---------------------------------------------------------------------------
# Tier A canonical-routing markers per Catalog #341 (non-negotiable invariants)
# ---------------------------------------------------------------------------


def test_real_scorer_helper_promotable_false_per_catalog_192():
    """Real-scorer helper MUST return promotable=False per Catalog #192 NEVER-promotable."""
    res = apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_pairs=SMOKE_NUM_PAIRS,
        frame_resolution_hw=SMOKE_RESOLUTION_HW,
        max_modes_to_verify=1,
    )
    assert res["promotable"] is False
    assert res["score_claim"] is False
    assert res["predicted_delta_adjustment"] == 0.0


def test_real_scorer_helper_axis_tag_macos_cpu_advisory():
    """Real-scorer helper MUST tag output as [macOS-CPU advisory] per Catalog #192."""
    res = apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_pairs=SMOKE_NUM_PAIRS,
        frame_resolution_hw=SMOKE_RESOLUTION_HW,
        max_modes_to_verify=1,
    )
    assert res["axis_tag"] == "[macOS-CPU advisory]"
    assert res["canonical_routing_markers"]["axis_tag"] == "[macOS-CPU advisory]"
    assert res["canonical_routing_markers"]["evidence_grade"] == "predicted"


def test_real_scorer_helper_canonical_provenance_present():
    """Real-scorer helper MUST emit canonical Provenance per Catalog #323."""
    res = apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_pairs=SMOKE_NUM_PAIRS,
        frame_resolution_hw=SMOKE_RESOLUTION_HW,
        max_modes_to_verify=1,
    )
    prov = res["canonical_provenance"]
    assert prov is not None
    assert prov["measurement_axis"] == "[macOS-CPU advisory]"
    assert prov["hardware_substrate"] == "macos_arm64_cpu"
    # Provenance evidence_grade for predicted-from-model per Catalog #323.
    assert prov["evidence_grade"] == "predicted"
    # NEVER promotable per Catalog #192 propagates into Provenance.
    assert prov["promotion_eligible"] is False
    assert prov["score_claim_valid"] is False


def test_real_scorer_helper_axis_decomposition_per_catalog_356():
    """Real-scorer helper MUST emit AxisDecomposition per Catalog #356."""
    res = apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_pairs=SMOKE_NUM_PAIRS,
        frame_resolution_hw=SMOKE_RESOLUTION_HW,
        max_modes_to_verify=1,
    )
    decomp = res["predicted_axis_decomposition"]
    assert "predicted_d_seg_delta" in decomp
    assert "predicted_d_pose_delta" in decomp
    assert "predicted_archive_bytes_delta" in decomp
    assert decomp["predicted_archive_bytes_delta"] == 0
    assert decomp["axis_tag"] == "[predicted]"
    assert "canonical_provenance" in decomp


# ---------------------------------------------------------------------------
# REAL behavioral verification — the canonical Slot EEE op-routable #1 ask
# ---------------------------------------------------------------------------


def test_real_scorer_helper_actually_runs_segnet_and_posenet_on_real_frames():
    """Real-scorer helper MUST actually run scorers (per_mode_empirical_verification non-empty).

    This is the canonical disambiguator between FAKE-metadata-only and
    REAL-scorer-verified per Slot EEE Audit Axis F (cite-vs-impl).
    """
    res = apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_pairs=SMOKE_NUM_PAIRS,
        frame_resolution_hw=SMOKE_RESOLUTION_HW,
        max_modes_to_verify=SMOKE_MAX_MODES,
    )
    assert len(res["per_mode_empirical_verification"]) == SMOKE_MAX_MODES
    assert res["canonical_menu_size"] == SMOKE_MAX_MODES
    assert res["num_pairs_evaluated"] == SMOKE_NUM_PAIRS

    for mode in res["per_mode_empirical_verification"]:
        # Empirical fields MUST be present + finite numeric values.
        assert "empirical_d_seg_mean" in mode
        assert "empirical_abs_d_pose_mean" in mode
        assert "per_pixel_argmax_disagreement_rate_mean" in mode
        # Per-pair count matches request.
        assert mode["pair_count"] == SMOKE_NUM_PAIRS
        # Numeric assertions.
        assert np.isfinite(mode["empirical_d_seg_mean"])
        assert np.isfinite(mode["empirical_abs_d_pose_mean"])
        assert np.isfinite(mode["per_pixel_argmax_disagreement_rate_mean"])
        # abs_d_pose by definition non-negative.
        assert mode["empirical_abs_d_pose_mean"] >= 0.0
        # argmax disagreement is a rate in [0, 1].
        assert 0.0 <= mode["per_pixel_argmax_disagreement_rate_mean"] <= 1.0


def test_real_scorer_helper_pixel_rolls_actually_modify_posenet_output():
    """PER_PIXEL_ROLL perturbations MUST produce non-zero |d_pose| empirically.

    This is the canonical REAL-vs-FAKE disambiguator at the PoseNet axis:
    if the perturbation actually modifies bytes, PoseNet MUST see a
    non-zero pose delta (modulo floating-point exact-zero edge cases).
    The legacy FAKE returned zero perturbation, so PoseNet would have
    returned zero |d_pose|. Now we expect non-zero |d_pose|.
    """
    res = apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_pairs=SMOKE_NUM_PAIRS,
        frame_resolution_hw=SMOKE_RESOLUTION_HW,
        max_modes_to_verify=SMOKE_MAX_MODES,
    )
    # At least ONE mode must have |d_pose| > 0 (proves PoseNet actually
    # saw the perturbation).
    abs_d_poses = [
        m["empirical_abs_d_pose_mean"]
        for m in res["per_mode_empirical_verification"]
    ]
    assert max(abs_d_poses) > 0.0, (
        f"All |d_pose| values are zero ({abs_d_poses}); the perturbation "
        f"did not reach PoseNet — this is the FAKE-class symptom per Slot "
        f"EEE Axis F"
    )


def test_real_scorer_helper_per_mode_verdict_uses_canonical_strings():
    """Per-mode verdict MUST be one of the canonical constants per #287 + #307."""
    canonical_verdicts = {
        VERDICT_NULL_PROJECTION_CONFIRMED_PER_MODE,
        VERDICT_NULL_PROJECTION_FALSIFIED_PER_MODE,
    }
    res = apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_pairs=SMOKE_NUM_PAIRS,
        frame_resolution_hw=SMOKE_RESOLUTION_HW,
        max_modes_to_verify=SMOKE_MAX_MODES,
    )
    for mode in res["per_mode_empirical_verification"]:
        assert mode["verdict"] in canonical_verdicts, (
            f"Mode {mode['mode_id']} verdict {mode['verdict']!r} not in "
            f"canonical verdict set"
        )


def test_real_scorer_helper_confirmed_count_matches_per_mode_verdicts():
    """modes_confirmed_count MUST match the per-mode verdict CONFIRMED count.

    Internal consistency invariant: aggregate counts must match per-mode
    verdicts (no off-by-one errors in the helper's classification path).
    """
    res = apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_pairs=SMOKE_NUM_PAIRS,
        frame_resolution_hw=SMOKE_RESOLUTION_HW,
        max_modes_to_verify=SMOKE_MAX_MODES,
    )
    expected_confirmed = sum(
        1
        for m in res["per_mode_empirical_verification"]
        if m["verdict"] == VERDICT_NULL_PROJECTION_CONFIRMED_PER_MODE
    )
    expected_falsified = sum(
        1
        for m in res["per_mode_empirical_verification"]
        if m["verdict"] == VERDICT_NULL_PROJECTION_FALSIFIED_PER_MODE
    )
    assert res["modes_confirmed_count"] == expected_confirmed
    assert res["modes_falsified_count"] == expected_falsified
    # Lists agree with counts.
    assert len(res["confirmed_mode_ids"]) == expected_confirmed
    assert len(res["falsified_mode_ids"]) == expected_falsified


def test_real_scorer_helper_pixel_rolls_confirm_segnet_null_axis():
    """Single-pixel-roll perturbations MUST EMPIRICALLY confirm SegNet-null axis.

    This is the canonical paradigm validation: per the function name's
    claim ("pose-axis null-projection ON SEGNET"), single-pixel rolls
    should preserve SegNet argmax (because EfficientNet stride-2 stem +
    bilinear resize (512, 384) absorb sub-pixel shifts per CLAUDE.md
    "Exact scorer architectures: SegNet"). If this assertion ever fails,
    the canonical Fridrich-Yousfi inverse-steganalysis paradigm at the
    pose-axis null-projection axis is implementation-level FALSIFIED on
    this scorer architecture per Catalog #307.

    Note this assertion is the ACTUAL paradigm validation per Slot EEE
    audit; the FAKE-class legacy ``apply_*`` never tested this.
    """
    res = apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_pairs=SMOKE_NUM_PAIRS,
        frame_resolution_hw=SMOKE_RESOLUTION_HW,
        max_modes_to_verify=SMOKE_MAX_MODES,
    )
    # Per-mode argmax disagreement rate MUST be at or below the canonical
    # SegNet-null tolerance for at least one mode (canonical paradigm
    # validation; if EVERY mode fails the paradigm IS empirically
    # FALSIFIED on this scorer architecture).
    pass_rates = [
        m["per_pixel_argmax_disagreement_rate_mean"]
        for m in res["per_mode_empirical_verification"]
    ]
    assert min(pass_rates) <= SEGNET_ARGMAX_NULL_TOLERANCE, (
        f"No PER_PIXEL_ROLL mode preserves SegNet argmax within tolerance "
        f"{SEGNET_ARGMAX_NULL_TOLERANCE}; per-mode rates: {pass_rates}. "
        f"Per Catalog #307: this is IMPLEMENTATION-LEVEL falsification "
        f"of single-pixel-roll perturbations on this scorer architecture; "
        f"the canonical Fridrich-Yousfi inverse-steganalysis PARADIGM is "
        f"INTACT"
    )


# ---------------------------------------------------------------------------
# Slot GGG remediation anchor per Catalog #348 retroactive sweep
# ---------------------------------------------------------------------------


def test_real_scorer_helper_returns_slot_ggg_remediation_anchor():
    """Slot GGG remediation anchor MUST be returned per Catalog #348 retroactive sweep."""
    res = apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_pairs=SMOKE_NUM_PAIRS,
        frame_resolution_hw=SMOKE_RESOLUTION_HW,
        max_modes_to_verify=1,
    )
    anchor = res["slot_ggg_remediation_anchor"]
    assert "slot_eee_audit_finding_axis_f_cite_vs_impl" in anchor
    assert "slot_ggg_part_3_closure" in anchor
    assert "predecessor_anchor_slot_rr_part_2" in anchor
    assert "remediation_landed_at_utc" in anchor
    assert "remediation_canonical_helpers" in anchor
    assert "canonical_paradigm_per_catalog_307" in anchor
    assert "PARADIGM intact" in anchor["canonical_paradigm_per_catalog_307"]
    # Canonical helpers list cites the three real helpers used.
    helpers = anchor["remediation_canonical_helpers"]
    assert any("tac.scorer.load_default_scorers" in h for h in helpers)
    assert any(
        "tac.substrates.score_aware_common.score_pair_components" in h for h in helpers
    )
    assert any(
        "tac.inverse_steganalysis_real_video_mlx" in h for h in helpers
    )


# ---------------------------------------------------------------------------
# Argument validation per Catalog #287
# ---------------------------------------------------------------------------


def test_real_scorer_helper_max_modes_lt_1_rejected_per_catalog_287():
    """max_modes_to_verify < 1 MUST raise ValueError per Catalog #287 invariants."""
    with pytest.raises(ValueError, match="max_modes_to_verify"):
        apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive(
            strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
            num_pairs=SMOKE_NUM_PAIRS,
            frame_resolution_hw=SMOKE_RESOLUTION_HW,
            max_modes_to_verify=0,
        )


def test_real_scorer_helper_verdict_matches_confirmed_falsified_state():
    """Overall verdict MUST reflect ALL_CONFIRMED / ALL_FALSIFIED / PARTIAL state."""
    res = apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_pairs=SMOKE_NUM_PAIRS,
        frame_resolution_hw=SMOKE_RESOLUTION_HW,
        max_modes_to_verify=SMOKE_MAX_MODES,
    )
    verdict = res["verdict"]
    c = res["modes_confirmed_count"]
    f = res["modes_falsified_count"]
    if c > 0 and f == 0:
        assert "ALL_MODES_CONFIRMED" in verdict
    elif c == 0 and f > 0:
        assert "ALL_MODES_FALSIFIED" in verdict
    elif c > 0 and f > 0:
        assert "PARTIAL_CONFIRMED" in verdict
    else:
        assert "NO_MODES_VERIFIED" in verdict
    # Every verdict variant MUST reference paired-CUDA RATIFICATION (the
    # canonical Catalog #246 + #325 + #307 routing).
    assert (
        "PAIRED_CUDA" in verdict
        or "PAIRED_CUDA_RATIFICATION" in verdict
        or "DEFERRED" in verdict
    )


def test_real_scorer_helper_canonical_equation_anti_pattern_ids_propagated():
    """Canonical equation + anti-pattern candidate IDs MUST propagate per Catalog #344."""
    res = apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_pairs=SMOKE_NUM_PAIRS,
        frame_resolution_hw=SMOKE_RESOLUTION_HW,
        max_modes_to_verify=1,
    )
    # Canonical equation candidate ID per Catalog #344 — DEFERRED pending
    # first empirical paired-CUDA anchor per "iterate not force".
    assert "pose_axis_null_projection" in res["canonical_equation_candidate_id"]
    # Canonical anti-pattern candidate ID per Catalog #344 sister
    # discipline — registered ONLY if empirical smoke FALSIFIES.
    assert "implementation_falsified" in res["canonical_anti_pattern_candidate_id"]


# ---------------------------------------------------------------------------
# Canonical strategy switching — verifies all 4 strategies wire end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [
        PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        PoseAxisNullProjectionStrategy.DCT_CHROMA_BASIS,
        PoseAxisNullProjectionStrategy.HADAMARD_TILE,
        PoseAxisNullProjectionStrategy.GAUSSIAN_NOISE,
    ],
)
def test_real_scorer_helper_all_4_canonical_strategies_wire_through(strategy):
    """All 4 canonical strategies per Catalog #308 MUST run end-to-end without crashing.

    This is the canonical Catalog #308 alternative-reducer enumeration
    integration test: each strategy produces actual per-mode empirical
    verification output via the real-scorer path.
    """
    res = apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive(
        strategy=strategy,
        num_pairs=2,
        frame_resolution_hw=(48, 64),
        max_modes_to_verify=1,  # cheap: 1 mode per strategy
    )
    # Per-mode verification non-empty.
    assert len(res["per_mode_empirical_verification"]) == 1
    # Strategy field reflects request.
    assert res["strategy"] == strategy.value
    # All canonical Tier A markers honored.
    assert res["promotable"] is False
    assert res["score_claim"] is False
    assert res["axis_tag"] == "[macOS-CPU advisory]"
