# SPDX-License-Identifier: MIT
"""ddm_cw1 -- the three LAWS the five win families rest on, registered as canonical equations.

The ``ddm_cw1`` canonicalization wave (task #1143, memo
``.omx/research/ddm_cw1_win_family_canonicalization_20260819.md``) standardised five
measured win families into engines under ``src/tac/win_families/`` plus
``tac.local_contest_instruments``.  Three of those families rest on a law that had been
re-derived per-arm and never registered.  This module registers them.

1. **Realized-acceptance monotonicity** (F1).  A descent that accepts only on the REALIZED
   joint objective can never worsen a coordinate: the accepted set has exactly zero
   regressions, by construction rather than by luck.  ``ddm_up2`` measured 429 pairs
   improved and **0 worsened** over n600.
2. **GT-lineage additive pose offset** (F5).  PyAV-lineage ``d_pose`` equals DALI-lineage
   ``d_pose`` plus a fixed additive constant ``C = 1.4061e-04`` on ``0.mkv``.  This is the
   instrument-refusal basis: because ``C`` dominates any good carrier's PyAV total, a PyAV
   pose ABSOLUTE is nearly floor-constant and must not be minimised against.
3. **Container archive-vs-payload delta** (F3).  The archive byte delta is not the payload
   delta; the difference is a container-attributable term.  ``ddm_up3`` measured **+7
   payload bits costing +48 archive bytes**, then recovered all 48 B by re-choosing the
   container alone.

All three are ``[macOS-CPU advisory / exact byte]``; ``score_claim=false``.  Only
``upstream/evaluate.py`` on contest hardware is a score.
"""

from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

MEMO = ".omx/research/ddm_cw1_win_family_canonicalization_20260819.md"

#: MEASURED ``ddm_pi2`` 2026-08-16: the MSE between the PyAV and DALI GT pose tables.
#: float32 reference 0.00014061325055081397; float64 re-reduction 0.00014061324889363773.
#: A property of ONE CLIP and TWO DECODERS -- never a general CPU/CUDA law.
GT_LINEAGE_ADDITIVE_POSE_OFFSET = 1.4061e-04
GT_LINEAGE_OFFSET_CLIP = "upstream/videos/0.mkv"


# ---------------------------------------------------------------------------
# The three predictors.
# ---------------------------------------------------------------------------


def realized_acceptance_worsened_count(
    start_values: list[float], final_values: list[float], *, tolerance: float = 0.0
) -> int:
    """Coordinates whose realized objective got WORSE across a realized-acceptance descent.

    The law predicts **0**.  A non-zero count falsifies either the acceptance rule (it
    accepted on something other than the realized objective) or the realization path (the
    objective measured at acceptance is not the objective measured at the end).
    """
    if len(start_values) != len(final_values):
        raise ValueError(
            f"{len(start_values)} start values but {len(final_values)} final values"
        )
    return sum(
        1
        for start, final in zip(start_values, final_values, strict=True)
        if final > start + tolerance
    )


def pose_pyav_from_dali(d_pose_dali: float, *, offset: float = GT_LINEAGE_ADDITIVE_POSE_OFFSET) -> float:
    """Predict PyAV-lineage ``d_pose`` from the DALI-lineage value.

    ADDITIVE, not multiplicative.  That form is what makes the PyAV absolute nearly
    floor-constant for a good carrier: at the live pointer's DALI ``d_pose`` of 7.77e-06
    the offset is 94.8% of the PyAV total, so a perfect carrier moves the PyAV number by
    only a few percent.
    """
    return float(d_pose_dali) + float(offset)


def container_attributable_bytes(payload_delta_bits: int, archive_delta_bytes: int) -> int:
    """Archive bytes NOT explained by the payload change.

    The payload term is its bit count in whole bytes, rounded away from zero (a coder
    cannot emit a fraction of a byte).  Whatever remains is the container's own term.
    """
    magnitude = math.ceil(abs(int(payload_delta_bits)) / 8.0)
    payload_bytes = magnitude if payload_delta_bits >= 0 else -magnitude
    return int(archive_delta_bytes) - payload_bytes


def _provenance():
    return build_provenance_for_research_sidecar(
        sidecar_path=MEMO,
        reactivation_criteria=(
            "law 1: any realized-acceptance run reporting a worsened coordinate reopens "
            "the acceptance/realization contract. law 2: the offset is 0.mkv-specific and "
            "must be re-measured for any other source video, or if either decoder changes. "
            "law 3: each new container search on a new body appends an anchor; the "
            "container term is body-specific and is never carried as a constant"
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="apple_macos_cpu_torch",
    )


# ---------------------------------------------------------------------------
# Law 1 -- realized-acceptance monotonicity.
# ---------------------------------------------------------------------------


def build_realized_acceptance_monotonicity_v1() -> CanonicalEquation:
    """The F1 method law: realized-only acceptance yields zero regressions."""
    provenance = _provenance()
    anchors = (
        EmpiricalAnchor(
            anchor_id="cw1_realized_acceptance_up2_n600_20260819",
            measurement_utc="2026-08-19T00:00:00Z",
            inputs={"pairs": 600, "improved": 429, "worsened": 0},
            predicted_output={"worsened": 0},
            empirical_output={"worsened": 0},
            residual=0.0,
            source_artifact=".omx/research/ddm_up2_shipping_object_pose_solve_20260819.md",
            measurement_method=(
                "n600 uncapped greedy descent on the int12 carrier lattice; every "
                "candidate rendered through the receiver path and scored by the frozen "
                "CPU PoseNet; d_pose 7.769484e-06 -> 7.649247e-06 (ratio 0.98452)"
            ),
            provenance=provenance,
            empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
            noise_floor=None,
            noise_floor_provenance=None,
        ),
        EmpiricalAnchor(
            anchor_id="cw1_realized_acceptance_tq1_phase_b_20260805",
            measurement_utc="2026-08-05T00:00:00Z",
            inputs={"queued_moves": 12, "accepted": 8, "worsened": 0},
            predicted_output={"worsened": 0},
            empirical_output={"worsened": 0},
            residual=0.0,
            source_artifact=".omx/research/ddm_tq1_20260805/tq1c/phase_b_realized_acceptance_receipt.json",
            measurement_method=(
                "phase-B realized acceptance, SATURATED_ACCEPTED_PREFIX; "
                "dS -0.000188043296 at d_bytes +1"
            ),
            provenance=provenance,
            empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
            noise_floor=None,
            noise_floor_provenance=None,
        ),
    )
    return CanonicalEquation(
        equation_id="cw1_realized_acceptance_monotonicity_v1",
        name="Realized-acceptance monotonicity",
        one_line_summary=(
            "accepting only on the REALIZED joint objective yields exactly zero worsened "
            "coordinates by construction (up2 n600: 429 improved, 0 worsened)"
        ),
        latex_form=r"\#\{i: J_{\mathrm{final}}(i) > J_{\mathrm{start}}(i)\} = 0",
        python_callable_module_path=(
            "tac.canonical_equations.ddm_cw1_win_family_laws_20260819:"
            "realized_acceptance_worsened_count"
        ),
        domain_of_validity={
            "included": [
                "coordinate descents whose acceptance test re-scores through the real decode",
                "per-coordinate-independent objectives (up2 pairs, jg1 per-pair token maps)",
            ],
            "excluded": [
                "any acceptance on a surrogate, linearisation, or predicted delta",
                "summing per-coordinate improvements into a score delta without an "
                "established independence claim",
                "runs truncated by an iteration cap (converged=False)",
            ],
            "authority": "[macOS-CPU advisory]",
        },
        units_in={"start_values": "score units", "final_values": "score units"},
        units_out={"worsened": "count of coordinates"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={
            "max_anchor_residual": max(a.residual for a in anchors)
        },
        last_calibration_utc="2026-08-19T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.win_families.realized_acceptance.RealizedAcceptanceEngine",
            "tac.win_families.realized_acceptance.DescentReport",
        ),
        canonical_producers=(
            "experiments.ddm_up2_shipping_pose_solve",
            "experiments.ddm_jg1_seg_solve",
        ),
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Law 2 -- the additive GT-lineage pose offset (the instrument-refusal basis).
# ---------------------------------------------------------------------------


def build_gt_lineage_additive_pose_offset_v1() -> CanonicalEquation:
    """The F5 law: PyAV d_pose = DALI d_pose + C, with C = 1.4061e-04 on 0.mkv."""
    provenance = _provenance()
    ra3_predicted = pose_pyav_from_dali(6.88e-06)
    anchors = (
        EmpiricalAnchor(
            anchor_id="cw1_gt_lineage_offset_ra3_closure_20260816",
            measurement_utc="2026-08-16T00:00:00Z",
            inputs={"d_pose_dali": 6.88e-06},
            predicted_output={"d_pose_pyav": ra3_predicted},
            empirical_output={"d_pose_pyav": 1.4747e-04},
            residual=abs(ra3_predicted - 1.4747e-04),
            source_artifact=".omx/research/ddm_ra3_subspace_trust_region_refit_20260816.md",
            measurement_method=(
                "independent closure: 6.88e-06 (contest-CUDA authority baseline) + "
                "1.4061e-04 = 1.474900e-04 against a measured advisory base of 1.4747e-04, "
                "0.014% apart; the scorer-forward CPU/CUDA term is 3.57e-12, falsified as "
                "the cause by nine orders of magnitude"
            ),
            provenance=provenance,
            empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
            noise_floor=None,
            noise_floor_provenance=None,
        ),
        EmpiricalAnchor(
            anchor_id="cw1_gt_lineage_offset_up2_finalize_20260819",
            measurement_utc="2026-08-19T00:00:00Z",
            inputs={"d_pose_dali": 7.76948388629175e-06},
            predicted_output={"d_pose_pyav": pose_pyav_from_dali(7.76948388629175e-06)},
            empirical_output={"d_pose_pyav": 0.0001482928},
            residual=abs(pose_pyav_from_dali(7.76948388629175e-06) - 0.0001482928),
            source_artifact=".omx/research/ddm_up2_finalize_n600_20260819.json",
            measurement_method=(
                "up2 n600 finalize cross-price: the same solved carrier measured against "
                "both GT tables; av_over_dali_before 19.0866 is the population ratio the "
                "additive form explains"
            ),
            provenance=provenance,
            empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
            noise_floor=None,
            noise_floor_provenance=None,
        ),
    )
    return CanonicalEquation(
        equation_id="cw1_gt_lineage_additive_pose_offset_v1",
        name="GT-lineage additive pose offset",
        one_line_summary=(
            "d_pose_pyav = d_pose_dali + 1.4061e-04 on 0.mkv (ADDITIVE); the offset "
            "dominates a good carrier's PyAV total, so the PyAV absolute is near-constant"
        ),
        latex_form=r"d_{pose}^{\mathrm{PyAV}} = d_{pose}^{\mathrm{DALI}} + C,\ C = 1.4061\times10^{-4}",
        python_callable_module_path=(
            "tac.canonical_equations.ddm_cw1_win_family_laws_20260819:pose_pyav_from_dali"
        ),
        domain_of_validity={
            "included": [
                f"the {GT_LINEAGE_OFFSET_CLIP} clip with the DALI/nvdec and "
                "PyAV/yuv420_to_rgb decoders as they stand",
                "converting a DALI-lineage pose measurement into its PyAV counterpart",
                "explaining why a PyAV pose DELTA is meaningful while the ABSOLUTE is not",
            ],
            "excluded": [
                "any other source video -- the constant is clip-specific and must be "
                "re-measured (ddm_pi2 states this explicitly)",
                "a general CPU-vs-CUDA hardware-drift law; the scorer-forward term is "
                "3.57e-12, nine orders below this offset",
                "per-pair use: the per-pair ratio C/d_dali spans 0.887 to 1,627, so 19.09x "
                "is a population median and not a per-pair conversion",
            ],
            "authority": "[macOS-CPU advisory]",
        },
        units_in={"d_pose_dali": "PoseNet MSE over the first 6 components"},
        units_out={"d_pose_pyav": "PoseNet MSE over the first 6 components"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={
            "max_anchor_residual": max(a.residual for a in anchors)
        },
        last_calibration_utc="2026-08-19T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.local_contest_instruments.assert_pose_absolute_quotable",
            "tac.local_contest_instruments.ADVISORY_POSE_ADDITIVE_FLOOR",
            "tac.win_families.terminal_compile.CompilePipeline.assert_stage_lineages",
        ),
        canonical_producers=(
            "experiments.ddm_up2_shipping_pose_solve",
            "experiments.ddm_ra3_advisory_pose_pricing",
        ),
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Law 3 -- the container archive-vs-payload delta.
# ---------------------------------------------------------------------------


def build_container_archive_vs_payload_delta_v1() -> CanonicalEquation:
    """The F3 law: archive delta is not payload delta; the remainder is the container."""
    provenance = _provenance()
    up3_predicted = container_attributable_bytes(7, 48)
    anchors = (
        EmpiricalAnchor(
            anchor_id="cw1_container_term_up3_thirteenth_move_20260819",
            measurement_utc="2026-08-19T00:00:00Z",
            inputs={"payload_delta_bits": 7, "archive_delta_bytes": 48},
            predicted_output={"container_attributable_bytes": up3_predicted},
            empirical_output={"container_attributable_bytes": 47},
            residual=abs(up3_predicted - 47),
            source_artifact=".omx/research/ddm_up3_thirteenth_move_byteclose_20260819.md",
            measurement_method=(
                "a re-solved n600 carrier moved the Rice payload 78,065 -> 78,072 bits "
                "(+7) yet grew the archive +48 B under the shipped container; the whole "
                "48 B was then recovered by CK2-off + brotli q10/lgwin16, landing at "
                "dB = 0 on archive 7ce46fd7... at 176,420 B"
            ),
            provenance=provenance,
            empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
            noise_floor=None,
            noise_floor_provenance=None,
        ),
        EmpiricalAnchor(
            anchor_id="cw1_container_term_canonical_consumer_proof_20260819",
            measurement_utc="2026-08-19T00:00:00Z",
            inputs={"payload_delta_bits": 7, "archive_delta_bytes": 48},
            predicted_output={"container_search_delta_bytes": -48},
            empirical_output={"container_search_delta_bytes": -48},
            residual=0.0,
            source_artifact=".omx/research/ddm_cw1_container_consumer_proof_20260819.json",
            measurement_method=(
                "the canonical optimizer re-ran up3's declared 8-config space on the same "
                "body: incumbent 176,468 B, winner 176,420 B (dB -48), winner sha "
                "7ce46fd7... byte-identical to the shipped pointer archive; identity and "
                "determinism controls both PASS"
            ),
            provenance=provenance,
            empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
            noise_floor=None,
            noise_floor_provenance=None,
        ),
    )
    return CanonicalEquation(
        equation_id="cw1_container_archive_vs_payload_delta_v1",
        name="Container archive-vs-payload delta",
        one_line_summary=(
            "archive dB minus the payload's whole-byte delta is a container term: up3 "
            "measured +7 payload bits costing +48 archive bytes, all recoverable"
        ),
        latex_form=r"\Delta B_{\mathrm{container}} = \Delta B_{\mathrm{archive}} - \lceil |\Delta b_{\mathrm{payload}}|/8 \rceil \mathrm{sgn}(\Delta b)",
        python_callable_module_path=(
            "tac.canonical_equations.ddm_cw1_win_family_laws_20260819:"
            "container_attributable_bytes"
        ),
        domain_of_validity={
            "included": [
                "attributing an archive.zip size change between its payload and container terms",
                "refusing a rate claim measured on payload length rather than archive size",
                "encoder-only knobs: brotli quality/lgwin, plane interleave, section order",
            ],
            "excluded": [
                "carrying a winning container config to a different body -- ck2 won -657 B "
                "with interleave ON and up3 won 48 B back with it OFF, same knob, opposite "
                "sign, different payload",
                "searches over an unsealed option space (the laundering shape)",
                "any candidate that has not been proven to parse back to its payload",
            ],
            "authority": "[macOS-CPU exact byte measurement]",
        },
        units_in={
            "payload_delta_bits": "bits of coded payload",
            "archive_delta_bytes": "bytes of archive.zip",
        },
        units_out={"container_attributable_bytes": "bytes of archive.zip"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={
            "max_anchor_residual": max(a.residual for a in anchors)
        },
        last_calibration_utc="2026-08-19T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.win_families.container_optimizer.archive_delta_report",
            "tac.win_families.container_optimizer.search_container_space",
            "experiments.ddm_cw1_container_consumer_proof",
        ),
        canonical_producers=(
            "experiments.ddm_up3_carrier_splice",
            "experiments.ddm_ck2_rate_ceiling_probe",
        ),
        provenance=provenance,
    )


ALL_CW1_WIN_FAMILY_BUILDERS = (
    build_realized_acceptance_monotonicity_v1,
    build_gt_lineage_additive_pose_offset_v1,
    build_container_archive_vs_payload_delta_v1,
)


def populate_cw1_win_family_laws(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> tuple[CanonicalEquation, ...]:
    """Append all three laws through the locked registry helper."""
    from tac.canonical_equations.registry import register_canonical_equation

    built = []
    for builder in ALL_CW1_WIN_FAMILY_BUILDERS:
        equation = builder()
        register_canonical_equation(
            equation,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes=(
                "ddm_cw1 win-family canonicalization (task #1143); "
                "[macOS-CPU advisory]; score_claim=false; consumers are the "
                "tac.win_families engines and tac.local_contest_instruments"
            ),
        )
        built.append(equation)
    return tuple(built)
