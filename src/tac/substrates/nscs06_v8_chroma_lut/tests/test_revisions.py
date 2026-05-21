# SPDX-License-Identifier: MIT
"""Tests for the 4 binding revisions per the per-substrate symposium.

Per RATIFY-3 2026-05-21 + per-substrate symposium ``council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md``
PROCEED_WITH_REVISIONS verdict (Shannon LEAD + Daubechies + Mallat + Carmack +
Hotz + Assumption-Adversary 6-attendee sextet pact). Operator blanket
approval 2026-05-21 #3 of 8.

Tests:

- REVISION #1 (Assumption-Adversary): per-assumption ablation ladder
- REVISION #2 (Daubechies + Mallat): multi-scale Dykstra-feasibility verdict
- REVISION #3 (Carmack + Hotz): MVP-first 5-step pre-smoke verification
- REVISION #4 (Assumption-Adversary): machine-readable JSON ablation table

Sister of existing 49 tests in ``test_substrate.py``; tests are additive +
preserve the BUILD's canonical contract intact per CLAUDE.md "HNeRV /
leaderboard-implementation parity discipline" + Catalog #220 operational
mechanism declaration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.nscs06_v8_chroma_lut import (
    NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT,
    PROCEDURAL_SEED_SIZE_BYTES,
)
from tac.substrates.nscs06_v8_chroma_lut.revisions import (
    CANONICAL_GENERATOR_KIND_ABLATION_AXIS,
    CANONICAL_LUMA_QUANTIZATION_LEVEL_ABLATION_AXIS,
    CANONICAL_PER_LEVEL_CLASS_AGGREGATION_ABLATION_AXIS,
    PER_ASSUMPTION_ABLATION_DIR_NAME,
    PER_ASSUMPTION_ABLATION_TABLE_SCHEMA_VERSION,
    CarmackMvpFirstPreSmokeVerificationVerdict,
    CarmackMvpFirstStepResult,
    MultiScaleDykstraFeasibilityVerdict,
    PerAssumptionAblationArm,
    PerAssumptionAblationLadder,
    build_per_assumption_ablation_ladder,
    build_per_assumption_ablation_table_path,
    emit_per_assumption_ablation_table_json,
    run_carmack_mvp_first_pre_smoke_verification,
    verify_multi_scale_dykstra_feasibility,
)


# ---------------------------------------------------------------------------
# REVISION #1: per-assumption ablation ladder
# ---------------------------------------------------------------------------


class TestRevision1AblationLadder:
    """REVISION #1 (Assumption-Adversary) per-assumption ablation ladder tests."""

    def test_canonical_axes_have_three_values(self) -> None:
        assert len(CANONICAL_LUMA_QUANTIZATION_LEVEL_ABLATION_AXIS) == 3
        assert len(CANONICAL_PER_LEVEL_CLASS_AGGREGATION_ABLATION_AXIS) == 3
        assert len(CANONICAL_GENERATOR_KIND_ABLATION_AXIS) == 3

    def test_canonical_luma_axis_includes_canonical_default(self) -> None:
        assert 16 in CANONICAL_LUMA_QUANTIZATION_LEVEL_ABLATION_AXIS

    def test_canonical_aggregation_axis_includes_median(self) -> None:
        assert "median" in CANONICAL_PER_LEVEL_CLASS_AGGREGATION_ABLATION_AXIS

    def test_canonical_generator_axis_includes_pcg64(self) -> None:
        assert "pcg64" in CANONICAL_GENERATOR_KIND_ABLATION_AXIS

    def test_build_ladder_returns_seven_arms(self) -> None:
        ladder = build_per_assumption_ablation_ladder()
        assert isinstance(ladder, PerAssumptionAblationLadder)
        assert len(ladder.arms) == 7

    def test_build_ladder_canonical_arm_identified(self) -> None:
        ladder = build_per_assumption_ablation_ladder()
        canonical_arms = [a for a in ladder.arms if a.is_canonical_arm]
        assert len(canonical_arms) == 1
        assert canonical_arms[0].arm_id == ladder.canonical_default_arm_id

    def test_build_ladder_six_probe_arms(self) -> None:
        ladder = build_per_assumption_ablation_ladder()
        probe_arms = [a for a in ladder.arms if not a.is_canonical_arm]
        assert len(probe_arms) == 6

    def test_build_ladder_two_probes_per_axis(self) -> None:
        ladder = build_per_assumption_ablation_ladder()
        probe_arms = [a for a in ladder.arms if not a.is_canonical_arm]
        axis_counts = {1: 0, 2: 0, 3: 0}
        for arm in probe_arms:
            axis_counts[arm.assumption_index] += 1
        assert axis_counts == {1: 2, 2: 2, 3: 2}

    def test_build_ladder_total_cost_matches_symposium(self) -> None:
        """REVISION #1 verbatim: '3 ablation arms x $0.50 each = $1.50 incremental
        over base $0.50 smoke = $2.00 total'."""
        ladder = build_per_assumption_ablation_ladder()
        assert ladder.total_predicted_cost_usd == pytest.approx(2.0, abs=1e-9)

    def test_build_ladder_arm_ids_unique(self) -> None:
        ladder = build_per_assumption_ablation_ladder()
        ids = [a.arm_id for a in ladder.arms]
        assert len(set(ids)) == len(ids)

    def test_build_ladder_predicted_delta_s_invariant_across_arms(self) -> None:
        """Canonical equation #26 closed-form predicted ΔS is invariant across
        all arms because the rate-axis savings depend only on
        (lut_bytes - seed_bytes), not the ablation axis choice."""
        ladder = build_per_assumption_ablation_ladder()
        deltas = {a.predicted_delta_s_canonical_equation_26 for a in ladder.arms}
        assert len(deltas) == 1

    def test_build_ladder_archive_bytes_delta_zero_for_all(self) -> None:
        ladder = build_per_assumption_ablation_ladder()
        for arm in ladder.arms:
            assert arm.predicted_archive_bytes_delta_vs_canonical == 0

    def test_build_ladder_rejects_canonical_outside_axis(self) -> None:
        with pytest.raises(ValueError):
            build_per_assumption_ablation_ladder(canonical_luma=99)

    def test_build_ladder_rejects_wrong_axis_length(self) -> None:
        with pytest.raises(ValueError):
            build_per_assumption_ablation_ladder(luma_axis=(8, 16, 32, 64))

    def test_arm_axis_tag_matches_canonical_provenance(self) -> None:
        ladder = build_per_assumption_ablation_ladder()
        for arm in ladder.arms:
            assert "[prediction;" in arm.axis_tag
            assert "canonical-equation-26-grounded" in arm.axis_tag

    def test_arm_rejects_invalid_assumption_index(self) -> None:
        with pytest.raises(ValueError):
            PerAssumptionAblationArm(
                arm_id="x",
                assumption_index=4,  # invalid
                axis_name="x",
                axis_value=8,
                canonical_default_value=16,
                is_canonical_arm=False,
                predicted_archive_bytes_delta_vs_canonical=0,
                predicted_delta_s_canonical_equation_26=-0.002706,
            )


# ---------------------------------------------------------------------------
# REVISION #2: multi-scale Dykstra-feasibility verdict
# ---------------------------------------------------------------------------


class TestRevision2MultiScaleFeasibility:
    """REVISION #2 (Daubechies + Mallat CO-LEAD) Dykstra-feasibility tests."""

    def test_verify_returns_verdict_type(self) -> None:
        verdict = verify_multi_scale_dykstra_feasibility()
        assert isinstance(verdict, MultiScaleDykstraFeasibilityVerdict)

    def test_pre_smoke_is_additive(self) -> None:
        """At PRE-SMOKE the verdict is structurally additive; FIRST-PAIRED-SMOKE
        empirically validates."""
        verdict = verify_multi_scale_dykstra_feasibility()
        assert verdict.is_additive is True

    def test_intersection_non_empty(self) -> None:
        verdict = verify_multi_scale_dykstra_feasibility()
        assert verdict.intersection_non_empty is True

    def test_canonical_lut_shape(self) -> None:
        verdict = verify_multi_scale_dykstra_feasibility()
        assert verdict.canonical_lut_shape == (16, 5, 3)

    def test_coarse_scale_dimension(self) -> None:
        verdict = verify_multi_scale_dykstra_feasibility()
        assert verdict.coarse_scale_dimension == 5

    def test_fine_scale_dimension(self) -> None:
        verdict = verify_multi_scale_dykstra_feasibility()
        assert verdict.fine_scale_dimension == 16 * 3  # 48

    def test_rate_axis_predicted_delta_matches_canonical_equation_26(self) -> None:
        verdict = verify_multi_scale_dykstra_feasibility()
        # Canonical equation #26 closed form: -25 * (4096 - 32) / 37_545_489
        expected = -25.0 * (4096 - 32) / 37_545_489
        assert verdict.rate_axis_predicted_delta == pytest.approx(expected, rel=1e-9)

    def test_seg_pose_placeholders_zero_at_pre_smoke(self) -> None:
        verdict = verify_multi_scale_dykstra_feasibility()
        assert verdict.seg_axis_predicted_delta_placeholder == 0.0
        assert verdict.pose_axis_predicted_delta_placeholder == 0.0

    def test_dykstra_iteration_count_canonical_one(self) -> None:
        """Closed-form additive case requires 1 alternating projection."""
        verdict = verify_multi_scale_dykstra_feasibility()
        assert verdict.dykstra_iteration_count == 1

    def test_no_unwind_test_recommended_at_pre_smoke(self) -> None:
        verdict = verify_multi_scale_dykstra_feasibility()
        assert verdict.unwind_test_recommended_assumptions == ()

    def test_axis_tag_canonical_provenance(self) -> None:
        verdict = verify_multi_scale_dykstra_feasibility()
        assert "[prediction;" in verdict.axis_tag

    def test_coarse_axis_label(self) -> None:
        verdict = verify_multi_scale_dykstra_feasibility()
        assert verdict.coarse_axis_label == "segnet_class"

    def test_fine_axis_label(self) -> None:
        verdict = verify_multi_scale_dykstra_feasibility()
        assert verdict.fine_axis_label == "(level, channel)"

    def test_verdict_rejects_invalid_unwind_assumption(self) -> None:
        with pytest.raises(ValueError):
            MultiScaleDykstraFeasibilityVerdict(
                is_additive=True,
                coarse_axis_label="x",
                fine_axis_label="y",
                coarse_scale_dimension=5,
                fine_scale_dimension=48,
                canonical_lut_shape=(16, 5, 3),
                additivity_tolerance=1e-6,
                rate_axis_predicted_delta=-0.0027,
                seg_axis_predicted_delta_placeholder=0.0,
                pose_axis_predicted_delta_placeholder=0.0,
                dykstra_iteration_count=1,
                intersection_non_empty=True,
                unwind_test_recommended_assumptions=(5,),  # invalid
            )


# ---------------------------------------------------------------------------
# REVISION #3: Carmack MVP-first 5-step pre-smoke verification
# ---------------------------------------------------------------------------


class TestRevision3CarmackMvpFirstVerification:
    """REVISION #3 (Carmack + Hotz) 5-step pre-smoke verification tests."""

    def test_run_returns_verdict_type(self, tmp_path: Path) -> None:
        verdict = run_carmack_mvp_first_pre_smoke_verification(tmp_dir=tmp_path)
        assert isinstance(verdict, CarmackMvpFirstPreSmokeVerificationVerdict)

    def test_run_has_five_steps(self, tmp_path: Path) -> None:
        verdict = run_carmack_mvp_first_pre_smoke_verification(tmp_dir=tmp_path)
        assert len(verdict.steps) == 5

    def test_run_steps_ordered_a_through_e(self, tmp_path: Path) -> None:
        verdict = run_carmack_mvp_first_pre_smoke_verification(tmp_dir=tmp_path)
        letters = [s.step_letter for s in verdict.steps]
        assert letters == ["a", "b", "c", "d", "e"]

    def test_run_all_steps_pass_locally(self, tmp_path: Path) -> None:
        """All 5 steps should pass in the canonical local environment."""
        verdict = run_carmack_mvp_first_pre_smoke_verification(tmp_dir=tmp_path)
        for step in verdict.steps:
            assert step.passed, f"step {step.step_letter} FAILED: {step.details}"
        assert verdict.all_steps_passed is True

    def test_run_ready_for_first_paired_smoke(self, tmp_path: Path) -> None:
        verdict = run_carmack_mvp_first_pre_smoke_verification(tmp_dir=tmp_path)
        assert verdict.ready_for_first_paired_smoke is True

    def test_run_step_b_canonical_raw_bytes_count(self, tmp_path: Path) -> None:
        """Verify step (b) inflate roundtrip writes the canonical bytes count."""
        verdict = run_carmack_mvp_first_pre_smoke_verification(tmp_dir=tmp_path)
        step_b = verdict.steps[1]
        assert step_b.step_letter == "b"
        assert step_b.passed
        # 1 pair x 2 frames x 32 x 64 x 3 = 12288 bytes
        assert "12288" in step_b.details

    def test_run_step_d_byte_mutation_distinguishing(self, tmp_path: Path) -> None:
        """Step (d) byte-mutation distinguishing-feature per Catalog #272."""
        verdict = run_carmack_mvp_first_pre_smoke_verification(tmp_dir=tmp_path)
        step_d = verdict.steps[3]
        assert step_d.step_letter == "d"
        assert step_d.passed
        assert "Catalog #272" in step_d.details

    def test_run_step_e_mps_refused(self, tmp_path: Path) -> None:
        """Step (e) MUST refuse PACT_INFLATE_DEVICE=mps per Catalog #205."""
        verdict = run_carmack_mvp_first_pre_smoke_verification(tmp_dir=tmp_path)
        step_e = verdict.steps[4]
        assert step_e.step_letter == "e"
        assert step_e.passed
        assert "MPS refused" in step_e.details

    def test_run_default_tmp_dir(self) -> None:
        """Calling without tmp_dir uses a tmpfs path (sister to canonical)."""
        verdict = run_carmack_mvp_first_pre_smoke_verification()
        assert verdict.all_steps_passed is True

    def test_step_result_rejects_invalid_letter(self) -> None:
        with pytest.raises(ValueError):
            CarmackMvpFirstStepResult(
                step_label="x", step_letter="f", passed=True, details="", elapsed_seconds=0.0
            )

    def test_verdict_rejects_wrong_step_count(self) -> None:
        with pytest.raises(ValueError):
            CarmackMvpFirstPreSmokeVerificationVerdict(
                all_steps_passed=True,
                steps=(),
                total_elapsed_seconds=0.0,
                ready_for_first_paired_smoke=True,
            )

    def test_verdict_rejects_unordered_steps(self) -> None:
        good = CarmackMvpFirstStepResult(
            step_label="x", step_letter="a", passed=True, details="", elapsed_seconds=0.0
        )
        bad = CarmackMvpFirstStepResult(
            step_label="x", step_letter="b", passed=True, details="", elapsed_seconds=0.0
        )
        with pytest.raises(ValueError):
            CarmackMvpFirstPreSmokeVerificationVerdict(
                all_steps_passed=True,
                steps=(bad, good, good, good, good),
                total_elapsed_seconds=0.0,
                ready_for_first_paired_smoke=True,
            )


# ---------------------------------------------------------------------------
# REVISION #4: machine-readable JSON ablation table
# ---------------------------------------------------------------------------


class TestRevision4JsonAblationTable:
    """REVISION #4 (Assumption-Adversary) JSON ablation table tests."""

    def test_canonical_schema_version_pinned(self) -> None:
        assert (
            PER_ASSUMPTION_ABLATION_TABLE_SCHEMA_VERSION
            == "nscs06_v8_per_assumption_ablation_v1_20260521"
        )

    def test_canonical_dir_name_pinned(self) -> None:
        assert PER_ASSUMPTION_ABLATION_DIR_NAME == "nscs06_v8_per_assumption_ablation"

    def test_canonical_path_under_omx_state(self, tmp_path: Path) -> None:
        path = build_per_assumption_ablation_table_path(
            repo_root=tmp_path, utc_now="20260521T060000Z"
        )
        rel = path.relative_to(tmp_path)
        assert rel.parts[0] == ".omx"
        assert rel.parts[1] == "state"
        assert rel.parts[2] == PER_ASSUMPTION_ABLATION_DIR_NAME
        assert rel.parts[3] == "nscs06_v8_per_assumption_ablation_20260521T060000Z.json"

    def test_emit_writes_canonical_json(self, tmp_path: Path) -> None:
        ladder = build_per_assumption_ablation_ladder()
        path = emit_per_assumption_ablation_table_json(
            ladder, repo_root=tmp_path, utc_now="2026-05-21T06:00:00Z"
        )
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == PER_ASSUMPTION_ABLATION_TABLE_SCHEMA_VERSION

    def test_emit_carries_canonical_provenance(self, tmp_path: Path) -> None:
        ladder = build_per_assumption_ablation_ladder()
        path = emit_per_assumption_ablation_table_json(
            ladder, repo_root=tmp_path, utc_now="2026-05-21T06:00:00Z"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        prov = data["canonical_provenance"]
        # Per Catalog #287 + #323: score_claim=False + promotable=False + evidence_grade=predicted
        assert prov["score_claim"] is False
        assert prov["promotable"] is False
        assert prov["evidence_grade"] == "predicted"
        assert "[prediction;" in prov["axis_tag"]
        assert prov["in_domain_context"] == "nscs06_v8_chroma_lut"

    def test_emit_includes_canonical_equation_id(self, tmp_path: Path) -> None:
        ladder = build_per_assumption_ablation_ladder()
        path = emit_per_assumption_ablation_table_json(
            ladder, repo_root=tmp_path
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert (
            data["canonical_provenance"]["canonical_equation_id"]
            == "procedural_codebook_from_seed_compression_savings_v1"
        )

    def test_emit_includes_seven_arms(self, tmp_path: Path) -> None:
        ladder = build_per_assumption_ablation_ladder()
        path = emit_per_assumption_ablation_table_json(
            ladder, repo_root=tmp_path
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["ladder"]["arms"]) == 7

    def test_emit_includes_multi_scale_feasibility(self, tmp_path: Path) -> None:
        ladder = build_per_assumption_ablation_ladder()
        path = emit_per_assumption_ablation_table_json(
            ladder, repo_root=tmp_path
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        ms = data["multi_scale_dykstra_feasibility"]
        assert ms["is_additive"] is True
        assert ms["intersection_non_empty"] is True
        assert ms["canonical_lut_shape"] == [16, 5, 3]

    def test_emit_includes_carmack_verdict_when_supplied(
        self, tmp_path: Path
    ) -> None:
        ladder = build_per_assumption_ablation_ladder()
        carmack = run_carmack_mvp_first_pre_smoke_verification(tmp_dir=tmp_path)
        path = emit_per_assumption_ablation_table_json(
            ladder,
            repo_root=tmp_path,
            carmack_mvp_verdict=carmack,
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "carmack_mvp_first_verification" in data
        assert data["carmack_mvp_first_verification"]["all_steps_passed"] is True
        assert len(data["carmack_mvp_first_verification"]["steps"]) == 5

    def test_emit_byte_stable_sort_keys(self, tmp_path: Path) -> None:
        """Two emissions of the same payload MUST produce byte-stable JSON."""
        ladder = build_per_assumption_ablation_ladder()
        path1 = tmp_path / "a.json"
        path2 = tmp_path / "b.json"
        emit_per_assumption_ablation_table_json(
            ladder, out_path=path1, utc_now="2026-05-21T06:00:00Z"
        )
        emit_per_assumption_ablation_table_json(
            ladder, out_path=path2, utc_now="2026-05-21T06:00:00Z"
        )
        assert path1.read_bytes() == path2.read_bytes()

    def test_emit_supports_extra_provenance(self, tmp_path: Path) -> None:
        ladder = build_per_assumption_ablation_ladder()
        path = emit_per_assumption_ablation_table_json(
            ladder,
            repo_root=tmp_path,
            extra_provenance={"sister_lane_id": "lane_test"},
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["canonical_provenance"]["sister_lane_id"] == "lane_test"

    def test_emit_includes_horizon_class(self, tmp_path: Path) -> None:
        ladder = build_per_assumption_ablation_ladder()
        path = emit_per_assumption_ablation_table_json(
            ladder, repo_root=tmp_path
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        # Per Catalog #309 horizon_class declaration
        assert data["horizon_class"] == "plateau_adjacent"


# ---------------------------------------------------------------------------
# Integration: end-to-end pre-smoke harvest plan
# ---------------------------------------------------------------------------


class TestRevisionsIntegration:
    """Sister end-to-end test that exercises REVISION #1 + #2 + #3 + #4 together."""

    def test_end_to_end_pre_smoke_harvest_plan(self, tmp_path: Path) -> None:
        """Construct + emit the full PRE-SMOKE-HARVEST plan per all 4 revisions."""
        # REVISION #1
        ladder = build_per_assumption_ablation_ladder()
        # REVISION #2
        multi_scale = verify_multi_scale_dykstra_feasibility()
        # REVISION #3
        carmack = run_carmack_mvp_first_pre_smoke_verification(tmp_dir=tmp_path)
        # REVISION #4 (consumes #1, #2, #3)
        path = emit_per_assumption_ablation_table_json(
            ladder,
            repo_root=tmp_path,
            multi_scale_verdict=multi_scale,
            carmack_mvp_verdict=carmack,
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        # Sanity: all 4 sections present
        assert "ladder" in data
        assert "multi_scale_dykstra_feasibility" in data
        assert "carmack_mvp_first_verification" in data
        assert "canonical_provenance" in data
        # Sanity: ready for first paired smoke
        assert carmack.ready_for_first_paired_smoke is True
        # Sanity: substrate contract unchanged (BUILD's contract preserved)
        assert NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT.id == "nscs06_v8_chroma_lut"
        assert NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT.recipe_research_only is True

    def test_canonical_equation_26_invariance(self, tmp_path: Path) -> None:
        """The 4 revisions MUST NOT change canonical equation #26's bytes-saved
        prediction (4096 - 32 = 4064 bytes)."""
        ladder = build_per_assumption_ablation_ladder()
        # All arms predict the same canonical equation #26 ΔS
        deltas = {a.predicted_delta_s_canonical_equation_26 for a in ladder.arms}
        assert len(deltas) == 1
        expected = -25.0 * (4096 - 32) / 37_545_489
        assert next(iter(deltas)) == pytest.approx(expected, rel=1e-9)
