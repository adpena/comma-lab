# SPDX-License-Identifier: MIT
"""NA5 native-coordinate shipping canonical anti-patterns."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tac.canonical_anti_patterns.anti_pattern import (
    PARADIGM_PROVENANCE,
    RECALIBRATE_ON_NEW_FALSIFICATIONS,
    SEVERITY_CRITICAL,
    AntiPattern,
)
from tac.canonical_anti_patterns.registry import register_anti_pattern
from tac.provenance.builders import build_provenance_for_predicted

if TYPE_CHECKING:
    from tac.provenance.contract import Provenance

_NA5_LANDING_UTC = "2026-08-05T00:00:00Z"
_DESIGN_PROV_SHA = "0" * 64


def _design_provenance(anti_pattern_id: str) -> Provenance:
    """Build non-promotable design provenance for NA5 anti-pattern classes."""
    return build_provenance_for_predicted(
        model_id=f"canonical_anti_patterns.na5_native_coordinate_builders.{anti_pattern_id}",
        inputs_sha256=_DESIGN_PROV_SHA,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
        captured_at_utc=_NA5_LANDING_UTC,
    )


def build_lossy_projection_shipped_expecting_decode_realization_v1() -> AntiPattern:
    """Build the addendum-8 native-coordinate shipping anti-pattern."""
    anti_pattern_id = "lossy_projection_shipped_expecting_decode_realization_v1"
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "A solver/trainer computes a native object, ships a lossy projection "
            "instead, then expects decode-side realization to recover the missing "
            "degrees of freedom. Addendum 8 found realization walls in six dead "
            "rows and two successes where the shipped object kept the solve or "
            "renderer native coordinates."
        ),
        forbidden_pattern_predicate=(
            "solve.native_dof_persisted == false AND payload_schema_declared_before_run "
            "== false AND decode_realizer_expected_to_reconstruct_projected_state == true"
        ),
        falsification_band={
            "measured_instances_total": 8.0,
            "dead_projection_instances": 6.0,
            "native_coordinate_success_instances": 2.0,
            "minimum_payload_schema_timing": 1.0,
        },
        recurrence_conditions=(
            "solver starts before its counted payload schema is declared",
            "training run discards optimizer, token, field, or solve DOF after measurement",
            "archive stores rendered paint, stamps, or summaries instead of the solved coordinates",
            "realization failure is answered with a cleverer decoder rather than native DOF persistence",
        ),
        canonical_source_anchor=(
            ".omx/research/operator_directive_per_edge_optimality_criteria_20260805.md "
            "ADDENDUM 8 native-coordinates shipping law; NA5 charter "
            ".omx/tmp/codex_runs/na5_prompt.md"
        ),
        canonical_unwind_path=(
            "Declare the payload schema before the solver runs; persist the "
            "trainer or solver native DOF atomically from byte one; make decode "
            "a mechanical application of those coordinates, then measure the "
            "counted bytes and receiver/scorer survival."
        ),
        canonical_producers=(
            "experiments/*solve*.py",
            "experiments/*carrier*.py",
            "tools/*materialize*.py",
            "src/tac/witness_control/*.py",
            "src/tac/witness_dsl/*.py",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/canonical_anti_patterns.match_stack_against_anti_patterns",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
        ),
        paradigm_class=PARADIGM_PROVENANCE,
        severity=SEVERITY_CRITICAL,
        provenance=_design_provenance(anti_pattern_id),
        empirical_falsifications=(),
        last_recalibration_utc=_NA5_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_all_na5_native_coordinate_anti_patterns() -> tuple[AntiPattern, ...]:
    """Return all NA5 native-coordinate anti-patterns."""
    return (build_lossy_projection_shipped_expecting_decode_realization_v1(),)


def populate_na5_native_coordinate_anti_patterns(
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str = "codex",
    subagent_id: str | None = None,
    notes: str | None = None,
) -> tuple[AntiPattern, ...]:
    """Append NA5 native-coordinate anti-pattern registrations to the registry."""
    anti_patterns = build_all_na5_native_coordinate_anti_patterns()
    for anti_pattern in anti_patterns:
        register_anti_pattern(
            anti_pattern,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes=notes or "NA5 addendum-8 native-coordinate registration",
        )
    return anti_patterns


__all__ = [
    "build_all_na5_native_coordinate_anti_patterns",
    "build_lossy_projection_shipped_expecting_decode_realization_v1",
    "populate_na5_native_coordinate_anti_patterns",
]
