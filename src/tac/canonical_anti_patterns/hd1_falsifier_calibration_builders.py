# SPDX-License-Identifier: MIT
"""HD1 falsifier-calibration canonical anti-pattern.

Registers the law the LR-rung finding produced: a falsifier band placed on a STOCHASTIC
endpoint must be calibrated against a SEED ENSEMBLE, never against a single control run.

A single control cannot separate a treatment effect from seed variance, so the resulting
band measures the wrong thing in both directions — it can REFUTE a real effect that happens
to land inside one seed's spread, and it can CONFIRM noise that happens to land outside it.
The empirical anchor is the LR ladder: `LR6E5`/`LR1E5D` were read as REFUTED against a
single control, then re-graded to WEAKENED-DIRECTIONAL (lr-up) and NULL (lr-down) once a
2-seed ensemble supplied the band. The verdict moved without any new treatment run.

This is a class-level guardrail. The numbers are advisory measurements from the re-grade
receipt, never a contest score claim.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tac.canonical_anti_patterns.anti_pattern import (
    PARADIGM_RIGOR_LOSS,
    RECALIBRATE_ON_NEW_FALSIFICATIONS,
    SEVERITY_HIGH,
    AntiPattern,
)
from tac.canonical_anti_patterns.registry import register_anti_pattern
from tac.provenance.builders import build_provenance_for_predicted

if TYPE_CHECKING:
    from tac.provenance.contract import Provenance

_HD1_LANDING_UTC = "2026-08-18T00:00:00Z"
_DESIGN_PROV_SHA = "0" * 64


def _design_provenance(anti_pattern_id: str) -> Provenance:
    """Build non-promotable design provenance for the HD1 anti-pattern class."""
    return build_provenance_for_predicted(
        model_id=f"canonical_anti_patterns.hd1_falsifier_calibration_builders.{anti_pattern_id}",
        inputs_sha256=_DESIGN_PROV_SHA,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
        captured_at_utc=_HD1_LANDING_UTC,
    )


def build_single_seed_falsifier_on_stochastic_endpoint_v1() -> AntiPattern:
    """Build the single-seed-falsifier-on-a-stochastic-endpoint anti-pattern."""
    anti_pattern_id = "single_seed_falsifier_on_stochastic_endpoint_v1"
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "Pre-registering a falsifier band on a stochastic training endpoint and "
            "calibrating it against a SINGLE control run. One control cannot separate a "
            "treatment effect from seed variance, so the band refutes real effects that "
            "fall inside one seed's spread and confirms noise that falls outside it. The "
            "LR ladder is the anchor: LR6E5/LR1E5D read REFUTED against a single control, "
            "then re-graded to WEAKENED-DIRECTIONAL and NULL on a 2-seed ensemble with no "
            "new treatment run. The verdict moved because the instrument moved."
        ),
        forbidden_pattern_predicate=(
            "verdict.endpoint_is_stochastic == True AND verdict.falsifier_band_source == "
            "'single_control_run' AND conclusion.promotes_refutation_or_confirmation"
        ),
        falsification_band={
            "min_control_seeds_for_a_band": 2.0,
            "lr_ladder_control_seeds_used": 1.0,
            "lr_ladder_regrade_seeds_used": 2.0,
            "verdicts_moved_on_regrade": 2.0,
            "new_treatment_runs_required_for_regrade": 0.0,
        },
        recurrence_conditions=(
            "a falsifier is written for a metric whose endpoint depends on seed, init, "
            "data order, or any other sampled quantity",
            "a single baseline or control run supplies the threshold the treatment is judged against",
            "an A/B is graded from one run per arm without a within-arm spread estimate",
            "a REFUTED or CONFIRMED verdict is promoted without stating the seed count behind its band",
        ),
        canonical_source_anchor=(
            "LR-rung re-grade (LR6E5 / LR1E5D): single-control REFUTED -> 2-seed ensemble "
            "WEAKENED-DIRECTIONAL (lr-up) and NULL (lr-down); carried forward by "
            ".omx/research/ddm_na9_gestalt_negative_audit_20260818.md N6 and landed as a "
            "canonical class by ddm_hd1 2026-08-18"
        ),
        canonical_unwind_path=(
            "Calibrate the band from a seed ensemble of the CONTROL arm (>=2 seeds, more when "
            "the effect is within one spread), state the seed count and the measured spread "
            "beside the band, and grade the treatment against that spread rather than against "
            "one run. Where an ensemble is unaffordable, the verdict scope is INSTANCE and the "
            "row must say the band is uncalibrated rather than reporting REFUTED or CONFIRMED."
        ),
        canonical_producers=(
            "ddm_hd1 na9-hazard hardening 2026-08-18",
            "LR-rung ensemble re-grade receipt",
        ),
        canonical_consumers=(
            "src/tac/canonical_anti_patterns.match_stack_against_anti_patterns",
            "tools/codex_arm_queue.py::lint_charter_optimal_form",
            "tac.preflight.check_compound_stack_proposal_acknowledges_known_anti_patterns",
        ),
        paradigm_class=PARADIGM_RIGOR_LOSS,
        severity=SEVERITY_HIGH,
        provenance=_design_provenance(anti_pattern_id),
        empirical_falsifications=(),
        last_recalibration_utc=_HD1_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_all_hd1_falsifier_calibration_anti_patterns() -> tuple[AntiPattern, ...]:
    """Return all HD1 falsifier-calibration anti-patterns."""
    return (build_single_seed_falsifier_on_stochastic_endpoint_v1(),)


def populate_hd1_falsifier_calibration_anti_patterns(
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str = "ddm_hd1",
    subagent_id: str | None = None,
    notes: str | None = None,
) -> tuple[AntiPattern, ...]:
    """Append HD1 falsifier-calibration anti-pattern registrations to the registry."""
    anti_patterns = build_all_hd1_falsifier_calibration_anti_patterns()
    for anti_pattern in anti_patterns:
        register_anti_pattern(
            anti_pattern,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes=notes or "HD1 ensemble-calibrated-falsifier registration (na9 N6)",
        )
    return anti_patterns


__all__ = [
    "build_all_hd1_falsifier_calibration_anti_patterns",
    "build_single_seed_falsifier_on_stochastic_endpoint_v1",
    "populate_hd1_falsifier_calibration_anti_patterns",
]
