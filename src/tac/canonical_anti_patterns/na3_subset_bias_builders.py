# SPDX-License-Identifier: MIT
"""NA3 subset-bias canonical anti-patterns.

Registers the two missing #923/#931 classes from the NA3 charter:

* prefix-bias sign inversion between seg and pose axes
* subset defaults that silently under-sample the 600-pair population

These entries are class-level guardrails. The numerical bands are advisory
measurements from the NA2/NA3 re-derivation receipts, not contest score claims.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tac.canonical_anti_patterns.anti_pattern import (
    PARADIGM_DIAGNOSIS,
    PARADIGM_RIGOR_LOSS,
    RECALIBRATE_ON_NEW_FALSIFICATIONS,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    AntiPattern,
)
from tac.canonical_anti_patterns.registry import register_anti_pattern
from tac.provenance.builders import build_provenance_for_predicted

if TYPE_CHECKING:
    from tac.provenance.contract import Provenance

_NA3_LANDING_UTC = "2026-08-05T00:00:00Z"
_DESIGN_PROV_SHA = "0" * 64


def _design_provenance(anti_pattern_id: str) -> Provenance:
    """Build non-promotable design provenance for NA3 anti-pattern classes."""
    return build_provenance_for_predicted(
        model_id=f"canonical_anti_patterns.na3_subset_bias_builders.{anti_pattern_id}",
        inputs_sha256=_DESIGN_PROV_SHA,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
        captured_at_utc=_NA3_LANDING_UTC,
    )


def build_prefix_bias_sign_inversion_pose_axis_v1() -> AntiPattern:
    """Build the pose-axis prefix-bias sign-inversion anti-pattern."""
    anti_pattern_id = "prefix_bias_sign_inversion_pose_axis_v1"
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "Treating prefix bias as uniformly conservative after checking only "
            "seg-axis drift. On the pose axis, video-order prefixes are harder "
            "(2.535x at n24 and 4.207x at n96), so prefix pose NO-GO verdicts "
            "are false-negative shaped rather than conservative population walls."
        ),
        forbidden_pattern_predicate=(
            "verdict.selection_mode == 'video_order_prefix' AND verdict.axis == "
            "'pose' AND conclusion.promotes_prefix_negative_as_population_or_"
            "conservative"
        ),
        falsification_band={
            "population_pairs": 600.0,
            "pose_prefix_ratio_n24": 2.535475579649216,
            "pose_prefix_ratio_n48": 2.640181689154513,
            "pose_prefix_ratio_n64": 2.6477688499984713,
            "pose_prefix_ratio_n96": 4.206770932037034,
            "hardest_over_easiest_block60": 79.43661398538532,
        },
        recurrence_conditions=(
            "pose-family verdict uses pairs [0:n] or max_pairs=n without an "
            "explicit non-prefix selection receipt",
            "seg-axis prefix drift is cited as if it transfers to pose-axis "
            "sampling difficulty",
            "a pose NO-GO is promoted from n8/n24/n96 prefix evidence to a "
            "population-family verdict",
        ),
        canonical_source_anchor=(
            ".omx/research/ddm_na2_negative_audit_20260803.md section 2.5.1 and "
            ".omx/research/ddm_na3_20260805 #931 re-derivation; "
            "src/tac/subset_selection.py pose-prefix ratio tests"
        ),
        canonical_unwind_path=(
            "Downgrade prefix pose verdicts to INSTANCE_ON_PREFIX unless a "
            "seeded-random, stratified, or strided non-prefix receipt states "
            "mode, seed or stride, denominator, and governing population ratio."
        ),
        canonical_producers=(
            ".omx/research/*pose*verdict*.md",
            "experiments/*pose*.py",
            "tools/*pose*.py",
            "src/tac/subset_selection.py",
        ),
        canonical_consumers=(
            "tac.subset_selection.assert_population_matched",
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/canonical_anti_patterns.match_stack_against_anti_patterns",
        ),
        paradigm_class=PARADIGM_DIAGNOSIS,
        severity=SEVERITY_HIGH,
        provenance=_design_provenance(anti_pattern_id),
        empirical_falsifications=(),
        last_recalibration_utc=_NA3_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_subset_default_silent_under_sampling_v1() -> AntiPattern:
    """Build the silent subset-default under-sampling anti-pattern."""
    anti_pattern_id = "subset_default_silent_under_sampling_v1"
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "Measurement code that defaults to a small slice, prefix, or "
            "max_pairs value without explicit selection provenance silently "
            "under-samples the 600-pair population. NA2 found 110 slice tools, "
            "0 representative selectors, and 71.5% silent same-line candidates."
        ),
        forbidden_pattern_predicate=(
            "(tool.default_pair_count < 600 OR code.contains('[:n]') OR "
            "code.contains('max_pairs=')) AND selection_provenance.mode IS NULL"
        ),
        falsification_band={
            "population_pairs": 600.0,
            "slice_tool_count": 110.0,
            "representative_selector_count": 0.0,
            "silent_same_line_candidate_fraction": 0.715,
            "minimum_admissible_nonprefix_pose_n": 32.0,
        },
        recurrence_conditions=(
            "a harness has a default n, n_pairs, max_pairs, or frames limit below "
            "the canonical 600-pair population",
            "selection is implied by array order instead of declared by a "
            "typed mode such as seeded_random, stratified, or strided",
            "a verdict omits denominator, seed or stride, selection mode, and "
            "population-match rationale",
        ),
        canonical_source_anchor=(
            ".omx/research/ddm_na2_negative_audit_20260803.md subset-default "
            "audit and NA3 #923/#931 charter; Catalog #344 registry pattern"
        ),
        canonical_unwind_path=(
            "Require explicit subset provenance before measurement launch: "
            "population denominator, selected denominator, mode, seed or stride, "
            "strata when used, and a stated governing population ratio."
        ),
        canonical_producers=(
            "experiments/*.py",
            "tools/*.py",
            "src/tac/subset_selection.py",
            "tools/build_strided_subset_gt.py",
        ),
        canonical_consumers=(
            "tac.subset_selection.selection_receipt",
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/canonical_anti_patterns.match_stack_against_anti_patterns",
        ),
        paradigm_class=PARADIGM_RIGOR_LOSS,
        severity=SEVERITY_CRITICAL,
        provenance=_design_provenance(anti_pattern_id),
        empirical_falsifications=(),
        last_recalibration_utc=_NA3_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_all_na3_subset_bias_anti_patterns() -> tuple[AntiPattern, ...]:
    """Return all NA3 subset-bias anti-patterns."""
    return (
        build_prefix_bias_sign_inversion_pose_axis_v1(),
        build_subset_default_silent_under_sampling_v1(),
    )


def populate_na3_subset_bias_anti_patterns(
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str = "codex",
    subagent_id: str | None = None,
    notes: str | None = None,
) -> tuple[AntiPattern, ...]:
    """Append NA3 subset-bias anti-pattern registrations to the registry."""
    anti_patterns = build_all_na3_subset_bias_anti_patterns()
    for anti_pattern in anti_patterns:
        register_anti_pattern(
            anti_pattern,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes=notes or "NA3 #923/#931 subset-bias registration",
        )
    return anti_patterns


__all__ = [
    "build_all_na3_subset_bias_anti_patterns",
    "build_prefix_bias_sign_inversion_pose_axis_v1",
    "build_subset_default_silent_under_sampling_v1",
    "populate_na3_subset_bias_anti_patterns",
]
