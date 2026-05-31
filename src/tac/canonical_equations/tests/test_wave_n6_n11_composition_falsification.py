# SPDX-License-Identifier: MIT
"""Tests for the Wave N+11 QUAD HALT composition anchor + Wave N+6 ratification guard.

Per Catalog #307 paradigm-vs-implementation classification + Catalog #344 canonical
equations + Catalog #287 HONEST verdict recording + Catalog #371 auto-recalibrator
+ Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE.

HONEST scope (per Catalog #229 premise verification): the Wave N+6 TRIPLE empirical
FALSIFICATION (92.48 paired CUDA+CPU, residual 92.320194) was ALREADY RATIFIED on
2026-05-28 by a sister subagent as anchor
``wave_n6_triple_z6_v2_nscs06_v8_compound_c_paired_cuda_cpu_empirical_falsification_20260528``.
This lane appends ONLY the genuinely-missing Wave N+11 QUAD HALT anchor and verifies
the already-ratified Wave N+6 falsification is present + coherent (no duplication).

The isolated tests build a tmp_path registry and exercise the canonical
``update_equation_with_empirical_anchor`` round-trip with a REAL
``tac.provenance.Provenance`` object (via the canonical research-sidecar builder).
The production-registry regression guard asserts the real landed anchors are present
+ non-promotable (no signal loss per the 2026-05-30 recovery directive).

[verified-against: tac.canonical_equations registry contract via get_equation_by_id]
"""

from __future__ import annotations

import json
from pathlib import Path

from tac.canonical_equations import (
    CanonicalEquation,
    EmpiricalAnchor,
    auto_recalibrate_from_continual_learning_posterior,
    get_equation_by_id,
    query_equations,
    register_canonical_equation,
    update_equation_with_empirical_anchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

TARGET = "triple_substrate_composition_orthogonal_pose_axis_savings_v1"

# The Wave N+6 empirical FALSIFICATION anchor that was ALREADY ratified 2026-05-28.
WAVE_N6_EXISTING_ID = (
    "wave_n6_triple_z6_v2_nscs06_v8_compound_c_paired_cuda_cpu_empirical_falsification_20260528"
)
# The Wave N+11 QUAD HALT anchor this lane appends.
WAVE_N11_ID = "wave_n11_quad_halt_phantom_provenance_pre_check_failed"

PRED = -0.036
# Residual of the already-ratified Wave N+6 falsification (92.48 - (-0.036) rounded).
WAVE_N6_RATIFIED_RESIDUAL = 92.320194


def _research_sidecar_provenance(measurement_axis: str = "[macOS-MLX research-signal]"):
    return build_provenance_for_research_sidecar(
        sidecar_path=(
            ".omx/research/wave_n11_quad_composition_sub015_cascade_halt_"
            "phantom_provenance_pre_check_failed_landed_20260530.md"
        ),
        reactivation_criteria=(
            "Path A standalone paired-CUDA per Catalog #246; Path B Wave N+12 "
            "reroute Z6-v2 pose-axis through PR101 HNeRV renderer; Path C defer "
            "until >=2 of 4 substrates have standalone contest-CUDA anchors."
        ),
        measurement_axis=measurement_axis,
        hardware_substrate="macos_arm64",
    )


def _predicted_provenance():
    return build_provenance_for_predicted(
        model_id="tac.optimization.substrate_composition_matrix.predicted_composite_delta",
        inputs_sha256="0" * 64,
        measurement_axis="[predicted]",
        hardware_substrate="macos_arm64",
    )


def _isolated_equation(equation_id: str = TARGET) -> CanonicalEquation:
    """Mirror the production triple-composition equation in an isolated registry."""
    return CanonicalEquation(
        equation_id=equation_id,
        name="Triple substrate composition orthogonal pose-axis savings",
        one_line_summary=(
            "Triple substrate composition predicts additive orthogonal-axis savings."
        ),
        latex_form="\\Delta S_{triple} = \\Delta S_{seg} + \\Delta S_{pose} + \\Delta S_{rate}",
        python_callable_module_path="tac.optimization.substrate_composition_matrix:predicted_composite_delta",
        domain_of_validity={"composition_arity": [3, 4]},
        units_in={"per_substrate_delta": "contest_score_delta"},
        units_out={"composite_delta": "contest_score_delta"},
        empirical_anchors=(
            # Mirror the already-ratified Wave N+6 empirical FALSIFICATION anchor.
            # NOTE: the PRODUCTION Wave N+6 anchor carries CONTEST_ARCHIVE_MEMBER
            # provenance (score_claim_valid=True) per the real paired-CUDA archive;
            # this isolated-registry MIRROR uses predicted provenance because the
            # tests assert the falsification RESIDUAL (92.320194) + anchor presence,
            # NOT the mirror's provenance grade. The production grade is verified by
            # the read-only TestProductionRegistryRegressionGuard against the real
            # registry. Building real CONTEST_ARCHIVE_MEMBER provenance here would
            # require a real archive.zip on disk (hermeticity violation).
            EmpiricalAnchor(
                anchor_id=WAVE_N6_EXISTING_ID,
                measurement_utc="2026-05-28T22:06:00Z",
                inputs={"composition_class": "triple"},
                predicted_output=PRED,
                empirical_output={
                    "composite_empirical_score_contest_cuda": 92.4795,
                    "composite_empirical_score_contest_cpu": 92.4762,
                    "posenet_distortion_contest_cuda": 162.52,
                    "verdict": "IMPLEMENTATION_LEVEL_FALSIFICATION_COMPOUND_C_RENDERER_NOT_POSENET_RECOGNIZABLE",
                },
                residual=WAVE_N6_RATIFIED_RESIDUAL,
                source_artifact="experiments/results/triple_z6_v2_plus_nscs06_v8_plus_compound_c_wave_n6_20260528/archive.zip",
                measurement_method="paired_modal_cpu_plus_cuda_t4_auth_eval_on_corrected_archive_grammar_per_catalog_246",
                provenance=_predicted_provenance(),
            ),
        ),
        predicted_vs_empirical_residual={
            "paired_modal_cpu_plus_cuda_t4_auth_eval_on_corrected_archive_grammar_per_catalog_246": WAVE_N6_RATIFIED_RESIDUAL,
        },
        last_calibration_utc="2026-05-28T22:06:00Z",
        next_recalibration_trigger="when_3+_new_empirical_anchors_in_domain",
        canonical_consumers=("tools.cathedral_autopilot_autonomous_loop",),
        canonical_producers=("tac.optimization.substrate_composition_matrix",),
        provenance=_predicted_provenance(),
    )


def _wave_n11_halt_anchor() -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id=WAVE_N11_ID,
        measurement_utc="2026-05-30T23:55:00Z",
        inputs={
            "composition_class": "quad_z6_v2_plus_z7_mamba2_plus_nscs06_v8_plus_compound_c",
            "predicted_registration_point": PRED,
            "halt_reason": "phantom_provenance_pre_check_failed_catalog_321_322",
            "dispatch_fired": False,
            "paid_spend_usd": 0.0,
            "inherits_wave_n6_implementation_falsification": WAVE_N6_EXISTING_ID,
        },
        predicted_output={"contest_score": PRED, "predicted_band": [-0.04, -0.07]},
        empirical_output={
            "contest_score": None,
            "halt": "phantom_provenance_pre_check_failed_catalog_321_322",
            "dispatch_fired": False,
            "paid_spend_usd": 0.0,
            "verdict": "HALT_AND_DEFER_NO_MEASUREMENT_TAKEN",
            "catalog_307_classification": "IMPLEMENTATION_LEVEL_FALSIFICATION_INHERITED_FROM_WAVE_N6",
            "paradigm_status": "INTACT_multi_substrate_composition",
        },
        residual=0.0,
        source_artifact=(
            ".omx/research/wave_n11_quad_composition_sub015_cascade_halt_"
            "phantom_provenance_pre_check_failed_landed_20260530.md"
        ),
        measurement_method="wave_n11_quad_halt_phantom_provenance_pre_check_no_dispatch",
        provenance=_research_sidecar_provenance(),
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
    )


def _events_for(repo_root: Path, equation_id: str) -> list[dict]:
    path = repo_root / ".omx" / "state" / "canonical_equations_registry.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("equation_id") == equation_id:
            rows.append(row)
    return rows


def _reg(repo_root: Path) -> Path:
    p = repo_root / ".omx" / "state" / "canonical_equations_registry.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _anchor(eq: CanonicalEquation, anchor_id: str) -> EmpiricalAnchor:
    return [a for a in eq.empirical_anchors if a.anchor_id == anchor_id][0]


def _prov_dict(anchor: EmpiricalAnchor) -> dict:
    prov = anchor.provenance
    if isinstance(prov, dict):
        return prov
    return dict(vars(prov))


# ---------------------------------------------------------------------------
# Wave N+11 QUAD HALT append (the genuinely-NEW work)
# ---------------------------------------------------------------------------


class TestWaveN11Halt:

    def test_wave_n11_halt_round_trip(self, tmp_path: Path) -> None:
        register_canonical_equation(_isolated_equation(), path=_reg(tmp_path))
        updated = update_equation_with_empirical_anchor(
            TARGET, _wave_n11_halt_anchor(), path=_reg(tmp_path)
        )
        anchor_ids = [a.anchor_id for a in updated.empirical_anchors]
        assert WAVE_N11_ID in anchor_ids

    def test_wave_n11_halt_records_anchor_appended_event(self, tmp_path: Path) -> None:
        register_canonical_equation(_isolated_equation(), path=_reg(tmp_path))
        update_equation_with_empirical_anchor(
            TARGET, _wave_n11_halt_anchor(), path=_reg(tmp_path)
        )
        event_types = [r.get("event_type") for r in _events_for(tmp_path, TARGET)]
        assert "anchor_appended" in event_types

    def test_wave_n11_halt_is_assumed_awaiting_verification(self, tmp_path: Path) -> None:
        register_canonical_equation(_isolated_equation(), path=_reg(tmp_path))
        updated = update_equation_with_empirical_anchor(
            TARGET, _wave_n11_halt_anchor(), path=_reg(tmp_path)
        )
        match = _anchor(updated, WAVE_N11_ID)
        assert match.empirical_verification_status == "ASSUMED_AWAITING_VERIFICATION"
        # HALT: no empirical measurement was taken (phantom-provenance refused dispatch).
        assert match.empirical_output["contest_score"] is None
        assert match.empirical_output["dispatch_fired"] is False
        assert match.empirical_output["paid_spend_usd"] == 0.0

    def test_wave_n11_halt_provenance_non_promotable(self, tmp_path: Path) -> None:
        # Catalog #321/#322/#323: the HALT anchor records a non-dispatch with NO
        # promotable score; score_claim_valid MUST be False so the phantom-provenance
        # gates cannot promote it.
        register_canonical_equation(_isolated_equation(), path=_reg(tmp_path))
        updated = update_equation_with_empirical_anchor(
            TARGET, _wave_n11_halt_anchor(), path=_reg(tmp_path)
        )
        prov = _prov_dict(_anchor(updated, WAVE_N11_ID))
        assert prov.get("score_claim_valid") is False
        assert prov.get("promotion_eligible") is False
        assert prov.get("measurement_axis") == "[macOS-MLX research-signal]"

    def test_wave_n11_halt_classification_is_implementation_level(
        self, tmp_path: Path
    ) -> None:
        # Catalog #307: the QUAD HALT inherits the IMPLEMENTATION-LEVEL classification
        # from the Wave N+6 TRIPLE falsification; paradigm INTACT.
        register_canonical_equation(_isolated_equation(), path=_reg(tmp_path))
        updated = update_equation_with_empirical_anchor(
            TARGET, _wave_n11_halt_anchor(), path=_reg(tmp_path)
        )
        match = _anchor(updated, WAVE_N11_ID)
        assert (
            match.empirical_output["catalog_307_classification"]
            == "IMPLEMENTATION_LEVEL_FALSIFICATION_INHERITED_FROM_WAVE_N6"
        )
        assert match.empirical_output["paradigm_status"] == "INTACT_multi_substrate_composition"

    def test_wave_n11_halt_residual_is_finite_zero(self, tmp_path: Path) -> None:
        # The dataclass refuses NaN + negative residual; the HALT carries residual=0.0
        # because no measurement was taken (the verification_status carries the signal).
        register_canonical_equation(_isolated_equation(), path=_reg(tmp_path))
        updated = update_equation_with_empirical_anchor(
            TARGET, _wave_n11_halt_anchor(), path=_reg(tmp_path)
        )
        match = _anchor(updated, WAVE_N11_ID)
        assert match.residual == 0.0


# ---------------------------------------------------------------------------
# Wave N+6 already-ratified guard (no duplication; APPEND-ONLY discipline)
# ---------------------------------------------------------------------------


class TestWaveN6AlreadyRatified:

    def test_isolated_equation_carries_wave_n6_falsification(self, tmp_path: Path) -> None:
        register_canonical_equation(_isolated_equation(), path=_reg(tmp_path))
        eq = get_equation_by_id(TARGET, path=_reg(tmp_path))
        anchor_ids = [a.anchor_id for a in eq.empirical_anchors]
        assert WAVE_N6_EXISTING_ID in anchor_ids

    def test_appending_n11_does_not_duplicate_or_mutate_n6(self, tmp_path: Path) -> None:
        register_canonical_equation(_isolated_equation(), path=_reg(tmp_path))
        updated = update_equation_with_empirical_anchor(
            TARGET, _wave_n11_halt_anchor(), path=_reg(tmp_path)
        )
        ids = [a.anchor_id for a in updated.empirical_anchors]
        # Wave N+6 falsification present exactly once; not duplicated.
        assert ids.count(WAVE_N6_EXISTING_ID) == 1
        # Wave N+6 anchor empirical detail unchanged.
        n6 = _anchor(updated, WAVE_N6_EXISTING_ID)
        assert n6.residual == WAVE_N6_RATIFIED_RESIDUAL
        assert (
            n6.empirical_output["verdict"]
            == "IMPLEMENTATION_LEVEL_FALSIFICATION_COMPOUND_C_RENDERER_NOT_POSENET_RECOGNIZABLE"
        )

    def test_wave_n6_residual_dominates_summary_after_n11_append(
        self, tmp_path: Path
    ) -> None:
        # The Wave N+6 92.320194 falsification residual must remain in the summary
        # after the Wave N+11 HALT (residual=0.0) is appended.
        register_canonical_equation(_isolated_equation(), path=_reg(tmp_path))
        updated = update_equation_with_empirical_anchor(
            TARGET, _wave_n11_halt_anchor(), path=_reg(tmp_path)
        )
        residual = updated.predicted_vs_empirical_residual
        assert any(
            abs(float(v) - WAVE_N6_RATIFIED_RESIDUAL) < 1e-6 for v in residual.values()
        )


# ---------------------------------------------------------------------------
# APPEND-ONLY discipline (Catalog #110/#113)
# ---------------------------------------------------------------------------


class TestAppendOnlyDiscipline:

    def test_appends_do_not_mutate_original_registration(self, tmp_path: Path) -> None:
        register_canonical_equation(_isolated_equation(), path=_reg(tmp_path))
        update_equation_with_empirical_anchor(
            TARGET, _wave_n11_halt_anchor(), path=_reg(tmp_path)
        )
        rows = _events_for(tmp_path, TARGET)
        registered = [r for r in rows if r.get("event_type") == "registered"]
        assert len(registered) == 1
        eq = registered[0].get("equation_payload") or registered[0].get("equation")
        # original registration carried exactly 1 anchor (the Wave N+6 falsification)
        assert len(eq["empirical_anchors"]) == 1
        assert eq["empirical_anchors"][0]["anchor_id"] == WAVE_N6_EXISTING_ID

    def test_event_order_registered_first(self, tmp_path: Path) -> None:
        register_canonical_equation(_isolated_equation(), path=_reg(tmp_path))
        update_equation_with_empirical_anchor(
            TARGET, _wave_n11_halt_anchor(), path=_reg(tmp_path)
        )
        event_types = [r.get("event_type") for r in _events_for(tmp_path, TARGET)]
        assert event_types[0] == "registered"
        assert "anchor_appended" in event_types[1:]


# ---------------------------------------------------------------------------
# Catalog #371 auto-recalibrator
# ---------------------------------------------------------------------------


class TestCatalog371Recalibration:

    def test_recalibrator_runs_without_error_after_n11_append(self, tmp_path: Path) -> None:
        register_canonical_equation(_isolated_equation(), path=_reg(tmp_path))
        update_equation_with_empirical_anchor(
            TARGET, _wave_n11_halt_anchor(), path=_reg(tmp_path)
        )
        report = auto_recalibrate_from_continual_learning_posterior(path=_reg(tmp_path))
        # The recalibrator runs; equations_checked counts the single equation.
        assert report.equations_checked >= 1

    def test_recalibrator_preserves_wave_n6_residual(self, tmp_path: Path) -> None:
        # After the HALT append + recalibration, the Wave N+6 92.320194 residual
        # must still dominate (the residual=0.0 HALT must NOT erase it).
        register_canonical_equation(_isolated_equation(), path=_reg(tmp_path))
        update_equation_with_empirical_anchor(
            TARGET, _wave_n11_halt_anchor(), path=_reg(tmp_path)
        )
        auto_recalibrate_from_continual_learning_posterior(path=_reg(tmp_path))
        eq = get_equation_by_id(TARGET, path=_reg(tmp_path))
        residual = eq.predicted_vs_empirical_residual
        assert any(
            abs(float(v) - WAVE_N6_RATIFIED_RESIDUAL) < 1e-6 for v in residual.values()
        )


# ---------------------------------------------------------------------------
# Production-registry regression guard (read-only; no signal loss)
# ---------------------------------------------------------------------------


class TestProductionRegistryRegressionGuard:
    """HONEST regression guard against the ACTUAL landed production registry."""

    def test_production_target_equation_exists(self) -> None:
        assert get_equation_by_id(TARGET) is not None

    def test_production_wave_n6_falsification_already_ratified(self) -> None:
        eq = get_equation_by_id(TARGET)
        assert eq is not None
        anchor_ids = [a.anchor_id for a in eq.empirical_anchors]
        assert WAVE_N6_EXISTING_ID in anchor_ids

    def test_production_wave_n11_halt_anchor_landed(self) -> None:
        eq = get_equation_by_id(TARGET)
        assert eq is not None
        anchor_ids = [a.anchor_id for a in eq.empirical_anchors]
        assert WAVE_N11_ID in anchor_ids

    def test_production_wave_n6_falsification_residual_present(self) -> None:
        eq = get_equation_by_id(TARGET)
        assert eq is not None
        residual = eq.predicted_vs_empirical_residual
        assert any(
            abs(float(v) - WAVE_N6_RATIFIED_RESIDUAL) < 1e-6 for v in residual.values()
        )

    def test_production_wave_n11_halt_is_non_promotable(self) -> None:
        # Catalog #321/#322/#323: the Wave N+11 HALT anchor carries score_claim_valid=False.
        eq = get_equation_by_id(TARGET)
        assert eq is not None
        n11 = [a for a in eq.empirical_anchors if a.anchor_id == WAVE_N11_ID]
        assert n11, "Wave N+11 HALT anchor must be landed in production registry"
        prov = _prov_dict(n11[0])
        assert prov.get("score_claim_valid") is False
        assert prov.get("measurement_axis") == "[macOS-MLX research-signal]"

    def test_production_wave_n11_no_duplicate(self) -> None:
        eq = get_equation_by_id(TARGET)
        assert eq is not None
        ids = [a.anchor_id for a in eq.empirical_anchors]
        assert ids.count(WAVE_N11_ID) == 1
