# SPDX-License-Identifier: MIT
"""Register the Wave-F Stage-2 correspondence-first canonical equations (triality: EQUATIONS leg).

Registers the 4 MEASURED/DERIVED equations from the 2026-07-02 Wave-F unified-xi BUILD + optimal-
tracking survey, AND appends the APPEND-ONLY superseding MEASURED anchor to the Stage-1
``lane_band_ego_factorization_source_reparam_v1`` equation (per Catalog #110/#113 HISTORICAL_
PROVENANCE: the prior ego-warp-mechanism payload is preserved; the measured row records that the
SOURCE-RE-PARAMETERIZATION THESIS is CONFIRMED but the MECHANISM is correspondence+denoising, NOT
ego-warp predictive coding).

The 4 new equations:
  1. index_permutation_discontinuity_defeats_temporal_model_v1   (DERIVED unifying law)
  2. lane_band_source_reparam_measured_resolution_v1             (MEASURED: ego REFUTED / smooth -42% / xi pure-pose)
  3. correspondence_first_lane_coding_optimal_pipeline_v1        ([prediction] optimal pipeline)
  4. openpilot_unified_physical_prior_both_scored_axes_v1        (DERIVED design-pattern)

Idempotent: registration is an append-only 'registered' event keyed by equation_id; re-running
appends a fresh event (latest-payload-wins). pointer contest-CPU 0.19110 UNMOVED (means, not goal).

Usage:
    .venv/bin/python tools/register_wave_f_correspondence_first_lane_coding_equations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_equations.equation import EmpiricalAnchor  # noqa: E402
from tac.canonical_equations.lane_band_correspondence_first_source_reparam import (  # noqa: E402
    build_all_correspondence_first_lane_coding_equations,
)
from tac.canonical_equations.registry import (  # noqa: E402
    register_canonical_equation,
    update_equation_with_empirical_anchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar  # noqa: E402

_SUBAGENT = "triality-219-correspondence-first-lane-coding"
_EGO_EQ_ID = "lane_band_ego_factorization_source_reparam_v1"
_LBND2_BYTES = 41_526
_EGO_LANE_OPT_BYTES = 47_453  # 1.143x LBND2 (WORSE)


def _superseding_anchor_for_ego_equation() -> EmpiricalAnchor:
    """APPEND-ONLY measured row: ego-warp MECHANISM refuted, source-reparam THESIS confirmed."""

    build = ".omx/research/wave_f_unified_xi_build_measured_20260702.md"
    # The Stage-1 ASSUMED anchor predicted the ego-warp path collapses the 26179 B floor toward
    # ~1-4 KB (ratio ~0.06x LBND2). MEASURED ego-predictive LBND3 = 1.143x LBND2 (WORSE). The
    # residual is the honest miss of the ego-warp MECHANISM sub-claim; the equation's THESIS
    # (source re-parameterization) is CONFIRMED via smoothing (a DIFFERENT mechanism, -42%).
    predicted_ratio = 0.06
    measured_ratio = _EGO_LANE_OPT_BYTES / _LBND2_BYTES  # ~1.143
    return EmpiricalAnchor(
        anchor_id="ego_warp_mechanism_refuted_source_reparam_thesis_confirmed_n600_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"stage_1_assumed_mechanism": "SE(3) ego-warp predictive coding collapses the floor to ~1-4 KB",
                "measured_mechanism": "correspondence + temporal smoothing (denoising, zero xi)"},
        predicted_output={"ego_warp_ratio_vs_lbnd2": predicted_ratio},
        empirical_output={"ego_predictive_lbnd3_ratio_vs_lbnd2": round(measured_ratio, 3),
                          "verdict_mechanism": "ego-warp REFUTED (LBND3 1.04-1.34x WORSE)",
                          "verdict_thesis": "source re-param CONFIRMED via smoothing (-42%, 0.02765->0.01608, below floor)",
                          "supersedes_via": "lane_band_source_reparam_measured_resolution_v1"},
        residual=abs(measured_ratio - predicted_ratio),  # ~1.08 -- flags the ego-warp mechanism miss
        source_artifact=build,
        measurement_method="n600_ego_warp_mechanism_refuted_source_reparam_confirmed_via_smoothing",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=build,
            reactivation_criteria="a JOINT (batch, not causal) world-BEV fit with a measured-reliable ego trajectory that beats the per-track smoother re-admits ego-warp",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )


def main() -> int:
    registered = []
    for eq in build_all_correspondence_first_lane_coding_equations():
        register_canonical_equation(eq, subagent_id=_SUBAGENT,
                                    notes="Wave-F Stage-2 correspondence-first lane coding (triality #219)")
        registered.append(eq.equation_id)
        print(f"registered: {eq.equation_id}")

    # APPEND-ONLY superseding measured anchor on the Stage-1 ego-factorization equation.
    updated = update_equation_with_empirical_anchor(
        _EGO_EQ_ID,
        _superseding_anchor_for_ego_equation(),
        subagent_id=_SUBAGENT,
        notes="APPEND-ONLY supersede: ego-warp mechanism refuted; source-reparam thesis confirmed via smoothing",
    )
    print(f"appended superseding anchor to: {updated.equation_id} "
          f"(now {len(updated.empirical_anchors)} anchors)")

    print(f"\nDONE: {len(registered)} new equations registered + 1 superseding anchor appended.")
    print("pointer contest-CPU 0.19110 UNMOVED (means, not goal progress).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
