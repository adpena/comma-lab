# SPDX-License-Identifier: MIT
"""Slot GGG SCALE-UP MATRIX dedicated tests (2026-05-30).

Per operator routing 2026-05-30 + Yousfi-cascade TIER-1 prerequisite for
Slot GGG x Cascade A FEC10 selector codec composition: scale-up from
2 modes x 2 pairs x 48x64 (Slot GGG Part 3) to N modes x M pairs x
contest resolution (384, 512).

Tests cover:

* Helper unit-tests for :func:`build_unified_canonical_scale_up_menu` +
  :func:`rank_confirmed_modes_by_capacity_per_cost` (numpy-only, fast)
* Tier A canonical-routing markers per Catalog #341 (NEVER promotable)
* Empirical artifact persistence to ``experiments/results/slot_ggg_scale_up_*``
* CONFIRMED_MODE_IDS ranking deterministic + DESCENDING by capacity_per_cost
* Per-axis decomposition emitted per Catalog #356
* Canonical Provenance per Catalog #323
* Catalog #287 placeholder rejection / Catalog #341 routing markers preserved
* End-to-end scale-up smoke at 4 modes x 2 pairs x 48x64 (cheap; ~25s)
* Larger scale-up at 16 modes x 4 pairs x (96, 128) marked ``@pytest.mark.slow``

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192: every assertion is
at the ``[macOS-CPU advisory]`` axis and is NEVER promotable to contest-axis
score claim. Paired Linux x86_64 + NVIDIA empirical anchor required per
Catalog #246 before any contest-axis claim.
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile

import numpy as np
import pytest


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
_UPSTREAM_POSENET = os.path.join(_REPO_ROOT, "upstream", "models", "posenet.safetensors")
_UPSTREAM_SEGNET = os.path.join(_REPO_ROOT, "upstream", "models", "segnet.safetensors")
_UPSTREAM_VIDEO = os.path.join(_REPO_ROOT, "upstream", "videos", "0.mkv")

_HAS_UPSTREAM_ASSETS = (
    os.path.exists(_UPSTREAM_POSENET)
    and os.path.exists(_UPSTREAM_SEGNET)
    and os.path.exists(_UPSTREAM_VIDEO)
)

# Tests that exercise the full end-to-end scale-up (decode + load scorers +
# scorer forward) require the upstream assets. Helper unit-tests do not.
requires_upstream_assets = pytest.mark.skipif(
    not _HAS_UPSTREAM_ASSETS,
    reason=(
        "Upstream models or video not available locally; "
        "Slot GGG scale-up end-to-end tests require upstream/models/*.safetensors "
        "and upstream/videos/0.mkv per Catalog #213"
    ),
)


from tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet import (
    CANONICAL_FRAME1_MENU_TOTAL,
    PR110_NUM_PAIRS,
    SCALE_UP_TIER_A_DEFAULT_N_MODES,
    SCALE_UP_TIER_A_DEFAULT_NUM_PAIRS,
    SCALE_UP_TIER_A_DEFAULT_RESOLUTION_HW,
    VERDICT_NULL_PROJECTION_CONFIRMED_PER_MODE,
    VERDICT_NULL_PROJECTION_FALSIFIED_PER_MODE,
    VERDICT_SCALE_UP_ALL_MODES_CONFIRMED,
    VERDICT_SCALE_UP_ALL_MODES_FALSIFIED,
    VERDICT_SCALE_UP_PARTIAL_CONFIRMED,
    apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution,
    build_unified_canonical_scale_up_menu,
    rank_confirmed_modes_by_capacity_per_cost,
)


# ============================================================================
# Helper unit tests (numpy-only; no upstream assets required)
# ============================================================================


class TestBuildUnifiedCanonicalScaleUpMenu:
    """Unit tests for the canonical multi-strategy unified menu builder."""

    def test_default_n_16_yields_8_pixel_roll_plus_8_dct_chroma(self):
        """Default n=16 fills first 8 PER_PIXEL_ROLL + first 8 DCT_CHROMA_BASIS."""
        menu = build_unified_canonical_scale_up_menu(16)
        assert len(menu) == 16
        family_counts = {}
        for m in menu:
            family_counts[m["family"]] = family_counts.get(m["family"], 0) + 1
        assert family_counts["frame1_pixel_roll"] == 8
        assert family_counts["frame1_dct_chroma"] == 8

    def test_n_8_yields_all_8_pixel_roll(self):
        """n=8 returns only PER_PIXEL_ROLL family (first in canonical fill order)."""
        menu = build_unified_canonical_scale_up_menu(8)
        assert len(menu) == 8
        assert all(m["family"] == "frame1_pixel_roll" for m in menu)

    def test_n_24_yields_pixel_roll_plus_full_dct_chroma(self):
        """n=24 = 8 PER_PIXEL_ROLL + 16 DCT_CHROMA_BASIS (all of both)."""
        menu = build_unified_canonical_scale_up_menu(24)
        assert len(menu) == 24
        family_counts = {}
        for m in menu:
            family_counts[m["family"]] = family_counts.get(m["family"], 0) + 1
        assert family_counts["frame1_pixel_roll"] == 8
        assert family_counts["frame1_dct_chroma"] == 16

    def test_n_27_pulls_3_hadamard_after_pixel_roll_and_dct(self):
        """n=27 = 8 PIXEL_ROLL + 16 DCT + 3 HADAMARD (canonical fill order)."""
        menu = build_unified_canonical_scale_up_menu(27)
        assert len(menu) == 27
        family_counts = {}
        for m in menu:
            family_counts[m["family"]] = family_counts.get(m["family"], 0) + 1
        assert family_counts["frame1_pixel_roll"] == 8
        assert family_counts["frame1_dct_chroma"] == 16
        assert family_counts["frame1_hadamard_tile"] == 3

    def test_n_43_yields_canonical_max_all_four_families(self):
        """n=43 = canonical CANONICAL_FRAME1_MENU_TOTAL across all 4 families."""
        menu = build_unified_canonical_scale_up_menu(43)
        assert len(menu) == 43
        family_counts = {}
        for m in menu:
            family_counts[m["family"]] = family_counts.get(m["family"], 0) + 1
        assert family_counts["frame1_pixel_roll"] == 8
        assert family_counts["frame1_dct_chroma"] == 16
        assert family_counts["frame1_hadamard_tile"] == 3
        assert family_counts["frame1_gaussian_noise"] == 16

    def test_n_99_clamps_to_canonical_max(self):
        """n > CANONICAL_FRAME1_MENU_TOTAL clamps to 43 (canonical max)."""
        menu = build_unified_canonical_scale_up_menu(99)
        assert len(menu) == CANONICAL_FRAME1_MENU_TOTAL == 43

    def test_n_lt_1_raises_per_catalog_287(self):
        """n < 1 raises ValueError per Catalog #287 invariants."""
        with pytest.raises(ValueError, match="n_modes_target must be >= 1"):
            build_unified_canonical_scale_up_menu(0)
        with pytest.raises(ValueError, match="n_modes_target must be >= 1"):
            build_unified_canonical_scale_up_menu(-1)

    def test_all_modes_have_canonical_shape(self):
        """Every mode dict carries canonical mode_id + family + params + description."""
        menu = build_unified_canonical_scale_up_menu(43)
        for m in menu:
            assert "mode_id" in m and isinstance(m["mode_id"], str) and m["mode_id"]
            assert "family" in m and m["family"].startswith("frame1_")
            assert "params" in m and isinstance(m["params"], dict)
            assert "description" in m

    def test_mode_ids_are_unique_within_unified_menu(self):
        """All mode_ids in the canonical menu are unique."""
        menu = build_unified_canonical_scale_up_menu(43)
        mode_ids = [m["mode_id"] for m in menu]
        assert len(mode_ids) == len(set(mode_ids))


class TestRankConfirmedModesByCapacityPerCost:
    """Unit tests for canonical capacity-per-cost ranker."""

    @staticmethod
    def _make_mode(
        mode_id: str,
        verdict: str,
        abs_d_pose: float,
        d_seg: float = 0.0,
        argmax_rate: float = 0.0,
    ) -> dict:
        return {
            "mode_id": mode_id,
            "family": "frame1_pixel_roll",
            "verdict": verdict,
            "empirical_abs_d_pose_mean": abs_d_pose,
            "empirical_d_seg_mean": d_seg,
            "per_pixel_argmax_disagreement_rate_mean": argmax_rate,
        }

    def test_empty_input_returns_empty(self):
        """No modes -> empty ranked list."""
        assert rank_confirmed_modes_by_capacity_per_cost([]) == []

    def test_falsified_modes_excluded(self):
        """FALSIFIED modes are excluded from ranking."""
        modes = [
            self._make_mode("a", VERDICT_NULL_PROJECTION_FALSIFIED_PER_MODE, 1e-7),
            self._make_mode("b", VERDICT_NULL_PROJECTION_CONFIRMED_PER_MODE, 1e-7),
        ]
        r = rank_confirmed_modes_by_capacity_per_cost(modes)
        assert len(r) == 1
        assert r[0]["mode_id"] == "b"

    def test_ranking_descending_by_capacity(self):
        """Modes with lower |d_pose| rank higher (capacity = 1/|d_pose|)."""
        modes = [
            self._make_mode("hi_cost", VERDICT_NULL_PROJECTION_CONFIRMED_PER_MODE, 5e-6),
            self._make_mode("lo_cost", VERDICT_NULL_PROJECTION_CONFIRMED_PER_MODE, 1e-7),
            self._make_mode("mid_cost", VERDICT_NULL_PROJECTION_CONFIRMED_PER_MODE, 1e-6),
        ]
        r = rank_confirmed_modes_by_capacity_per_cost(modes)
        assert [m["mode_id"] for m in r] == ["lo_cost", "mid_cost", "hi_cost"]
        # Capacity values strictly descending.
        capacities = [m["capacity_per_cost"] for m in r]
        assert capacities == sorted(capacities, reverse=True)

    def test_ranking_handles_exact_zero_d_pose_via_epsilon(self):
        """Mode with EXACTLY zero |d_pose| ranks first (capacity ~ 1/epsilon)."""
        modes = [
            self._make_mode("zero", VERDICT_NULL_PROJECTION_CONFIRMED_PER_MODE, 0.0),
            self._make_mode("tiny", VERDICT_NULL_PROJECTION_CONFIRMED_PER_MODE, 1e-7),
        ]
        r = rank_confirmed_modes_by_capacity_per_cost(modes)
        assert r[0]["mode_id"] == "zero"
        # Top capacity is finite (epsilon prevents div-by-zero).
        assert np.isfinite(r[0]["capacity_per_cost"])

    def test_ranking_propagates_canonical_fields(self):
        """Ranked entries preserve mode_id + family + d_seg + argmax_rate."""
        modes = [
            self._make_mode(
                "x",
                VERDICT_NULL_PROJECTION_CONFIRMED_PER_MODE,
                1e-7,
                d_seg=2e-9,
                argmax_rate=5e-4,
            )
        ]
        r = rank_confirmed_modes_by_capacity_per_cost(modes)
        assert r[0]["mode_id"] == "x"
        assert r[0]["family"] == "frame1_pixel_roll"
        assert r[0]["empirical_d_seg_mean"] == pytest.approx(2e-9)
        assert r[0]["per_pixel_argmax_disagreement_rate_mean"] == pytest.approx(5e-4)


# ============================================================================
# End-to-end scale-up entry point (requires upstream assets)
# ============================================================================


# Cheap smoke configuration: small N + M + low resolution so the suite stays
# under ~30s per parametrized test on macOS-CPU. The canonical Tier A
# operator-routing config is much larger; tests use smaller defaults so the
# suite stays cheap.
SCALE_UP_SMOKE_N_MODES = 2
SCALE_UP_SMOKE_NUM_PAIRS = 2
SCALE_UP_SMOKE_RESOLUTION_HW = (48, 64)


@requires_upstream_assets
class TestScaleUpEntryPoint:
    """End-to-end smoke tests for the canonical scale-up entry point."""

    def test_smoke_returns_canonical_tier_a_routing_markers(self, tmp_path):
        """Tier A non-promotable invariants per Catalog #192 + #341 + #357."""
        res = apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
            n_modes_target=SCALE_UP_SMOKE_N_MODES,
            num_pairs=SCALE_UP_SMOKE_NUM_PAIRS,
            frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
            artifact_output_dir=str(tmp_path),
        )
        assert res["promotable"] is False
        assert res["score_claim"] is False
        assert res["predicted_delta_adjustment"] == 0.0
        assert res["axis_tag"] == "[macOS-CPU advisory]"
        # Routing markers nested dict mirrors top-level Tier A invariants.
        markers = res["canonical_routing_markers"]
        assert markers["promotable"] is False
        assert markers["score_claim"] is False
        assert markers["axis_tag"] == "[macOS-CPU advisory]"
        assert markers["evidence_grade"] == "predicted"

    def test_smoke_emits_canonical_provenance_per_catalog_323(self, tmp_path):
        """Canonical Provenance per Catalog #323 with predicted-from-model grade."""
        res = apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
            n_modes_target=SCALE_UP_SMOKE_N_MODES,
            num_pairs=SCALE_UP_SMOKE_NUM_PAIRS,
            frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
            artifact_output_dir=str(tmp_path),
        )
        prov = res["canonical_provenance"]
        assert prov["measurement_axis"] == "[macOS-CPU advisory]"
        assert prov["hardware_substrate"] == "macos_arm64_cpu"
        assert prov["evidence_grade"] == "predicted"
        assert prov["promotion_eligible"] is False
        assert prov["score_claim_valid"] is False

    def test_smoke_emits_axis_decomposition_per_catalog_356(self, tmp_path):
        """Per-axis AxisDecomposition per Catalog #356 with EMPIRICAL d_seg/d_pose."""
        res = apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
            n_modes_target=SCALE_UP_SMOKE_N_MODES,
            num_pairs=SCALE_UP_SMOKE_NUM_PAIRS,
            frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
            artifact_output_dir=str(tmp_path),
        )
        decomp = res["predicted_axis_decomposition"]
        assert "predicted_d_seg_delta" in decomp
        assert "predicted_d_pose_delta" in decomp
        assert decomp["predicted_archive_bytes_delta"] == 0
        assert decomp["axis_tag"] == "[predicted]"
        assert "canonical_provenance" in decomp

    def test_smoke_per_mode_verification_includes_family_params(self, tmp_path):
        """Per-mode verification includes family + params (scale-up extension)."""
        res = apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
            n_modes_target=SCALE_UP_SMOKE_N_MODES,
            num_pairs=SCALE_UP_SMOKE_NUM_PAIRS,
            frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
            artifact_output_dir=str(tmp_path),
        )
        assert len(res["per_mode_empirical_verification"]) == SCALE_UP_SMOKE_N_MODES
        for mode in res["per_mode_empirical_verification"]:
            assert "mode_id" in mode
            assert "family" in mode and mode["family"].startswith("frame1_")
            assert "params" in mode and isinstance(mode["params"], dict)
            assert "empirical_d_seg_mean" in mode
            assert "empirical_abs_d_pose_mean" in mode
            assert "per_pixel_argmax_disagreement_rate_mean" in mode
            assert mode["verdict"] in {
                VERDICT_NULL_PROJECTION_CONFIRMED_PER_MODE,
                VERDICT_NULL_PROJECTION_FALSIFIED_PER_MODE,
            }

    def test_smoke_ranked_confirmed_modes_descending_by_capacity(self, tmp_path):
        """Ranked CONFIRMED_MODE_IDS sorted DESCENDING by capacity_per_cost."""
        res = apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
            n_modes_target=4,  # PER_PIXEL_ROLL family; reasonable for small smoke
            num_pairs=SCALE_UP_SMOKE_NUM_PAIRS,
            frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
            artifact_output_dir=str(tmp_path),
        )
        ranked = res["ranked_confirmed_modes_by_capacity_per_cost"]
        # Only confirmed modes appear; counts match top-level field.
        assert len(ranked) == res["modes_confirmed_count"]
        # Strictly descending by capacity (or equal at ties).
        capacities = [m["capacity_per_cost"] for m in ranked]
        assert capacities == sorted(capacities, reverse=True)
        # Every ranked entry corresponds to a CONFIRMED mode_id.
        ranked_ids = {m["mode_id"] for m in ranked}
        confirmed_ids = set(res["confirmed_mode_ids"])
        assert ranked_ids == confirmed_ids

    def test_smoke_baseline_cache_savings_above_zero(self, tmp_path):
        """Baseline cache reduces scorer calls vs naive 2*N*M."""
        res = apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
            n_modes_target=4,
            num_pairs=4,
            frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
            artifact_output_dir=str(tmp_path),
        )
        # At N=4, M=4: naive = 2*4*4 = 32; actual = 4 (baseline) + 4*4 (perturbed) = 20.
        # Savings = (32-20)/32 = 0.375.
        assert res["baseline_cache_savings_ratio"] > 0.3
        assert res["baseline_cache_calls_count"] == 4
        assert res["perturbed_scorer_calls_count"] == 16
        assert res["baseline_cache_actual_calls_count"] == 20
        assert res["baseline_cache_naive_calls_count_equivalent"] == 32

    def test_smoke_verdict_uses_canonical_scale_up_constants(self, tmp_path):
        """Verdict is one of canonical VERDICT_SCALE_UP_* constants."""
        res = apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
            n_modes_target=SCALE_UP_SMOKE_N_MODES,
            num_pairs=SCALE_UP_SMOKE_NUM_PAIRS,
            frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
            artifact_output_dir=str(tmp_path),
        )
        canonical_verdicts = {
            VERDICT_SCALE_UP_ALL_MODES_CONFIRMED,
            VERDICT_SCALE_UP_ALL_MODES_FALSIFIED,
            VERDICT_SCALE_UP_PARTIAL_CONFIRMED,
            "SCALE_UP_NO_MODES_VERIFIED_DEFERRED_PENDING_PAIRED_CUDA_RATIFICATION",
        }
        assert res["verdict"] in canonical_verdicts
        # Internal consistency: verdict reflects modes_confirmed_count.
        c = res["modes_confirmed_count"]
        f = res["modes_falsified_count"]
        if c > 0 and f == 0:
            assert res["verdict"] == VERDICT_SCALE_UP_ALL_MODES_CONFIRMED
        elif c == 0 and f > 0:
            assert res["verdict"] == VERDICT_SCALE_UP_ALL_MODES_FALSIFIED
        elif c > 0 and f > 0:
            assert res["verdict"] == VERDICT_SCALE_UP_PARTIAL_CONFIRMED

    def test_smoke_artifact_path_persisted_with_canonical_schema(self, tmp_path):
        """Empirical artifact JSON persisted to artifact_output_dir."""
        res = apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
            n_modes_target=SCALE_UP_SMOKE_N_MODES,
            num_pairs=SCALE_UP_SMOKE_NUM_PAIRS,
            frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
            artifact_output_dir=str(tmp_path),
        )
        assert res["artifact_path"] is not None
        assert pathlib.Path(res["artifact_path"]).exists()
        with open(res["artifact_path"]) as f:
            artifact = json.load(f)
        assert (
            artifact["schema"]
            == "slot_ggg_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution.v1"
        )
        # Non-promotable invariants survive persistence.
        assert artifact["score_claim"] is False
        assert artifact["promotion_eligible"] is False
        assert artifact["ready_for_exact_eval_dispatch"] is False
        assert artifact["rank_or_kill_eligible"] is False
        assert artifact["evidence_grade"] == "macOS-CPU-advisory"
        # Per-mode verification + ranked list survive JSON round-trip.
        assert (
            len(artifact["per_mode_empirical_verification"]) == SCALE_UP_SMOKE_N_MODES
        )
        # FEC10 handoff envelope present for downstream consumer.
        assert "fec10_composition_handoff" in artifact
        assert artifact["fec10_composition_handoff"]["do_not_touch_this_turn"] is True

    def test_smoke_skip_artifact_when_empty_string(self):
        """artifact_output_dir='' skips artifact write (test-only opt-out)."""
        res = apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
            n_modes_target=SCALE_UP_SMOKE_N_MODES,
            num_pairs=SCALE_UP_SMOKE_NUM_PAIRS,
            frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
            artifact_output_dir="",
        )
        assert res["artifact_path"] is None

    def test_smoke_slot_ggg_scale_up_anchor_returned(self, tmp_path):
        """Slot GGG scale-up anchor returned per Catalog #348 retroactive sweep."""
        res = apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
            n_modes_target=SCALE_UP_SMOKE_N_MODES,
            num_pairs=SCALE_UP_SMOKE_NUM_PAIRS,
            frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
            artifact_output_dir=str(tmp_path),
        )
        anchor = res["slot_ggg_scale_up_anchor"]
        assert "slot_ggg_part_3_predecessor_anchor" in anchor
        assert "slot_ggg_scale_up_closure" in anchor
        assert "yousfi_cascade_prediction" in anchor
        assert "next_turn_operator_routable" in anchor
        assert "canonical_paradigm_per_catalog_307" in anchor
        # Predecessor anchor cites canonical Slot GGG Part 3 commit.
        assert "f9d0f2465" in anchor["slot_ggg_part_3_predecessor_anchor"]
        # Closure cites canonical scale-up engineering invariants.
        assert "baseline cache" in anchor["slot_ggg_scale_up_closure"]

    def test_smoke_n_modes_clamped_to_canonical_max_via_menu_helper(self):
        """n_modes_target > 43 clamps to CANONICAL_FRAME1_MENU_TOTAL.

        We test via the canonical menu helper directly (numpy-only; fast) to
        avoid the macOS-CPU scorer-forward cost of 43 modes x 1 pair (~70s,
        over the repo-wide pytest timeout=60). The canonical entry-point's
        clamping behavior is structurally identical to the menu helper's
        because the entry point delegates clamping to
        :func:`build_unified_canonical_scale_up_menu`.
        """
        menu = build_unified_canonical_scale_up_menu(99)
        assert len(menu) == CANONICAL_FRAME1_MENU_TOTAL == 43

    def test_smoke_n_lt_1_rejected_per_catalog_287(self):
        """n_modes_target < 1 rejected per Catalog #287 invariants."""
        with pytest.raises(ValueError, match="n_modes_target must be >= 1"):
            apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
                n_modes_target=0,
                num_pairs=1,
                frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
                artifact_output_dir="",
            )

    def test_smoke_num_pairs_lt_1_rejected_per_catalog_287(self):
        """num_pairs < 1 rejected per Catalog #287 invariants."""
        with pytest.raises(ValueError, match="num_pairs must be >= 1"):
            apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
                n_modes_target=1,
                num_pairs=0,
                frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
                artifact_output_dir="",
            )

    def test_smoke_num_pairs_above_pr110_cap_rejected(self):
        """num_pairs > PR110_NUM_PAIRS rejected per canonical cap."""
        with pytest.raises(ValueError, match="num_pairs must be <="):
            apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
                n_modes_target=1,
                num_pairs=PR110_NUM_PAIRS + 1,
                frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
                artifact_output_dir="",
            )

    def test_smoke_canonical_equation_registration_deferred_below_threshold(
        self, tmp_path
    ):
        """Below-threshold canonical equation registration is DEFERRED per Catalog #344."""
        res = apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
            n_modes_target=SCALE_UP_SMOKE_N_MODES,
            num_pairs=SCALE_UP_SMOKE_NUM_PAIRS,
            frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
            artifact_output_dir=str(tmp_path),
            register_canonical_equation_on_confirmation=True,
            canonical_equation_confirmation_threshold=100,  # impossibly high
        )
        reg = res["canonical_equation_registration"]
        assert reg is not None
        assert reg["registered"] is False
        assert "DEFERRED" in reg["reason"]

    def test_smoke_canonical_equation_registration_skipped_when_opt_in_false(
        self, tmp_path
    ):
        """When opt-in flag is False, canonical_equation_registration is None."""
        res = apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
            n_modes_target=SCALE_UP_SMOKE_N_MODES,
            num_pairs=SCALE_UP_SMOKE_NUM_PAIRS,
            frame_resolution_hw=SCALE_UP_SMOKE_RESOLUTION_HW,
            artifact_output_dir=str(tmp_path),
            register_canonical_equation_on_confirmation=False,
        )
        assert res["canonical_equation_registration"] is None

    def test_canonical_default_constants_match_design_memo(self):
        """Canonical default constants match Slot GGG scale-up design memo Sec 4."""
        # Tier A defaults per design memo Sec 4: 16 modes, 60 pairs, contest resolution.
        assert SCALE_UP_TIER_A_DEFAULT_N_MODES == 16
        assert SCALE_UP_TIER_A_DEFAULT_NUM_PAIRS == 60
        assert SCALE_UP_TIER_A_DEFAULT_RESOLUTION_HW == (384, 512)


# ============================================================================
# Larger-scale tests marked slow (only run with --runslow / -m slow)
# ============================================================================


_RUN_SLOW_ENABLED = os.environ.get("RUN_SLOW", "").strip() in {"1", "true", "yes"}


@requires_upstream_assets
@pytest.mark.slow
@pytest.mark.skipif(
    not _RUN_SLOW_ENABLED,
    reason=(
        "Slow tests exceed repo-wide pytest timeout=60 in pyproject.toml. "
        "Run via RUN_SLOW=1 pytest ... -m slow to exercise full Tier A grid."
    ),
)
class TestScaleUpLargerConfigs:
    """Larger scale-up smokes for the canonical Tier A operator-routing config.

    These are NOT run by default (marked slow + skipped unless RUN_SLOW=1).
    Operator runs them via:
        RUN_SLOW=1 pytest -m slow src/tac/composition/pr110_opt_6_motion_pair_repair_*/tests/test_slot_ggg_scale_up_matrix.py
    """

    @pytest.mark.timeout(900)  # 15 minute budget for slow Tier A smoke
    def test_tier_a_smoke_16_modes_x_4_pairs_x_96_128(self, tmp_path):
        """16 modes x 4 pairs x (96, 128) ~ 8 minutes wall-clock on macOS-CPU."""
        res = apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution(
            n_modes_target=16,
            num_pairs=4,
            frame_resolution_hw=(96, 128),
            artifact_output_dir=str(tmp_path),
        )
        assert res["n_modes_probed"] == 16
        assert res["num_pairs_evaluated"] == 4
        # Per-mode verification non-empty.
        assert len(res["per_mode_empirical_verification"]) == 16
        # Ranked CONFIRMED list non-empty (ranker valid; depends on empirical
        # confirmation rate on macOS-CPU at this resolution).
        assert isinstance(res["ranked_confirmed_modes_by_capacity_per_cost"], list)
