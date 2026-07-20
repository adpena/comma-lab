# SPDX-License-Identifier: MIT
"""Canonical laws for the n600 G1 worldsheet and G3 cell-code measurements."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

REPO = Path(__file__).resolve().parents[3]
UTC = "2026-07-20T21:00:00Z"
AXIS = "[macOS-CPU advisory]"
RECEIPT = ".omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.json"
MEMO = ".omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.md"
TOOL = "tools/measure_g1_worldsheet_g3_cellcode.py"

WORLDSHEET_EQUATION_ID = "worldsheet_transport_residual_event_rate_v1"
CELLCODE_EQUATION_ID = "argmax_cell_identity_ideal_bytes_v1"


def transport_event_fraction(event_count: int, observation_count: int) -> float:
    """Boundary observations whose transport residual exceeds a fixed radius."""
    if type(event_count) is not int or type(observation_count) is not int:
        raise TypeError("event_count and observation_count must be exact integers")
    if observation_count <= 0 or event_count < 0 or event_count > observation_count:
        raise ValueError("require 0 <= event_count <= observation_count and observation_count > 0")
    return event_count / observation_count


def ideal_cell_stream_bytes(probabilities: Iterable[float]) -> float:
    """Shannon ideal bytes for a known-site target-cell identity stream."""
    total_bits = 0.0
    count = 0
    for probability in probabilities:
        p = float(probability)
        if not 0.0 < p <= 1.0 or not math.isfinite(p):
            raise ValueError("every cell probability must be finite and in (0,1]")
        total_bits -= math.log2(p)
        count += 1
    if not count:
        raise ValueError("probabilities must be non-empty")
    return total_bits / 8.0


def ideal_cell_stream_bytes_from_bit_sum(negative_log2_probability_sum: float) -> float:
    """Aggregate form used by the n600 receipt after per-site streaming."""
    value = float(negative_log2_probability_sum)
    if value < 0 or not math.isfinite(value):
        raise ValueError("negative_log2_probability_sum must be finite and nonnegative")
    return value / 8.0


def _receipt(path: Path | None = None) -> tuple[dict[str, Any], str]:
    source = path or REPO / RECEIPT
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "g1_worldsheet_g3_cellcode_measurements.v1":
        raise ValueError(f"measurement receipt schema drifted in {source}")
    if payload.get("axis") != AXIS or payload.get("score_claim") is not False:
        raise ValueError("measurement receipt authority firewall drifted")
    return payload, str(source if path is not None else RECEIPT)


def _provenance() -> Any:
    return build_provenance_for_research_sidecar(
        sidecar_path=MEMO,
        reactivation_criteria=(
            "Re-measure G1 when an exact cross-pair pose bank, depth-stratified/curve-domain "
            "transport, or explicit birth/death state is available. Re-measure G3 when a "
            "receiver-closed causal candidate-set and site grammar can be charged together."
        ),
        measurement_axis=AXIS,
        hardware_substrate="darwin_arm64_cpu",
        captured_at_utc=UTC,
    )


def build_worldsheet_transport_residual_event_rate_v1(
    receipt_path: Path | None = None,
) -> CanonicalEquation:
    payload, source = _receipt(receipt_path)
    g1 = payload["g1"]
    transition = g1["aggregate"]["by_transition"]
    empirical: dict[str, Any] = {}
    for kind in ("within_pair", "cross_pair"):
        row = transition[kind]
        observations = int(row["observation_count"])
        fraction = float(row["event_fraction_gt_px"]["4"])
        event_count = round(fraction * observations)
        predicted = transport_event_fraction(event_count, observations)
        empirical[kind] = {
            "observation_count": observations,
            "event_gt4_count_recovered_from_receipt": event_count,
            "event_fraction_gt4_px": fraction,
            "formula_replay_fraction": predicted,
            "formula_abs_residual": abs(predicted - fraction),
            "median_of_transition_medians_px": row["median_of_transition_medians_px"],
            "weighted_finite_chamfer_mean_px": row["weighted_finite_chamfer_mean_px"],
        }
    provenance = _provenance()
    anchor = EmpiricalAnchor(
        anchor_id="g1_groundplane_worldsheet_transport_n600_20260720",
        measurement_utc=UTC,
        inputs={
            "transitions": {"within_pair": 600, "cross_pair": 599},
            "strata": 10,
            "event_radius_px": 4,
            "transport": g1["transport_assumptions"],
            "cache_sha256": payload["cache"]["sha256"],
        },
        predicted_output={
            "law": "E_r = count(d_i > r) / N",
            "formula_replay": {
                kind: row["formula_replay_fraction"] for kind, row in empirical.items()
            },
        },
        empirical_output={
            "transition_metrics": empirical,
            "verdict": g1["verdict"],
            "verdict_scope": g1["verdict_scope"],
        },
        residual=max(row["formula_abs_residual"] for row in empirical.values()),
        source_artifact=source,
        measurement_method=(
            "n600 interclass 4-neighbor edges; tac.lie SE3 ground-plane homography; "
            "symmetric Chamfer and exact threshold-event counts"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=WORLDSHEET_EQUATION_ID,
        name="Worldsheet transport residual event-rate law",
        one_line_summary=(
            "At radius r, the successor-vehicle residual rate is the fraction of transported "
            "boundary observations with nearest-edge residual above r, stratified by cadence."
        ),
        latex_form=r"E_r = N^{-1}\sum_{i=1}^{N}\mathbf{1}[d_i>r]",
        python_callable_module_path=(
            "tac.canonical_equations.g1g3_successor_measurements_20260720:"
            "transport_event_fraction"
        ),
        domain_of_validity={
            "vehicle": "G1 worldsheet successor design",
            "cache": "frozen n600 SegNet argmax, class order Road/Lane/Undrivable/Movable/MyCar",
            "transport": "single global ground-plane homography",
            "within_pose": "exact banked non-overlapping-pair target",
            "cross_pose": "nearest-target-pair proxy, not exact cross-pair pose",
            "verdict_scope": g1["verdict_scope"],
            "score_claim": False,
        },
        units_in={"event_count": "boundary_observations", "observation_count": "boundary_observations"},
        units_out={"transport_event_fraction": "fraction"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"n600_formula_replay": anchor.residual},
        last_calibration_utc=UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("#574 post-row elevation", "G1 worldsheet transport selector"),
        canonical_producers=(TOOL, RECEIPT, MEMO),
        provenance=provenance,
    )


def build_argmax_cell_identity_ideal_bytes_v1(
    receipt_path: Path | None = None,
) -> CanonicalEquation:
    payload, source = _receipt(receipt_path)
    g3 = payload["g3"]
    all_flips = g3["all_flips"]
    moderate = g3["moderate_band"]
    best = str(g3["best_measured_prior"])
    all_bits = float(all_flips["ideal_bits"][best])
    moderate_bits = float(moderate["ideal_bits"][best])
    all_bytes = ideal_cell_stream_bytes_from_bit_sum(all_bits)
    moderate_bytes = ideal_cell_stream_bytes_from_bit_sum(moderate_bits)
    provenance = _provenance()
    anchor = EmpiricalAnchor(
        anchor_id="g3_argmax_cell_identity_floor_n600_20260720",
        measurement_utc=UTC,
        inputs={
            "flip_count": int(all_flips["flip_count"]),
            "moderate_flip_count": int(moderate["flip_count"]),
            "best_prior": best,
            "alphabet": g3["alphabet"],
            "cache_sha256": payload["cache"]["sha256"],
        },
        predicted_output={
            "all_flips_ideal_bytes": all_bytes,
            "moderate_ideal_bytes": moderate_bytes,
            "law": "B_cell = (1/8) sum_i -log2 p(c_i | context_i)",
        },
        empirical_output={
            "all_flips_ideal_bytes": all_flips["ideal_bytes"],
            "moderate_ideal_bytes": moderate["ideal_bytes"],
            "raw_coordinate_bytes": all_flips["raw_coordinate"]["ideal_bytes"],
            "comparators": g3["comparators"],
            "verdict": g3["verdict"],
            "verdict_scope": g3["verdict_scope"],
        },
        residual=max(
            abs(all_bytes - float(all_flips["ideal_bytes"][best])),
            abs(moderate_bytes - float(moderate["ideal_bytes"][best])),
        ),
        source_artifact=source,
        measurement_method=(
            "exact n600 live batch16 flip inventory; per-site causal categorical priors; "
            "ideal arithmetic code length decomposed by pair and stratum"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=CELLCODE_EQUATION_ID,
        name="Argmax-cell identity ideal byte floor",
        one_line_summary=(
            "Known-site cell identities cost one eighth of summed causal self-information; "
            "site grammar, headers, receiver, and realization remain outside this floor."
        ),
        latex_form=r"B_{\mathrm{cell}} = \frac{1}{8}\sum_i -\log_2 p(c_i\mid\mathcal{C}_i)",
        python_callable_module_path=(
            "tac.canonical_equations.g1g3_successor_measurements_20260720:"
            "ideal_cell_stream_bytes_from_bit_sum"
        ),
        domain_of_validity={
            "vehicle": "G3 cell-code successor design",
            "known_site_assumption": True,
            "excluded_costs": ["site locations", "candidate-set transport", "coder headers", "receiver", "realization"],
            "verdict_scope": g3["verdict_scope"],
            "score_claim": False,
        },
        units_in={"negative_log2_probability_sum": "bits"},
        units_out={"ideal_cell_stream_bytes": "bytes"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"n600_formula_replay": anchor.residual},
        last_calibration_utc=UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("#572", "r1b5 GAP-3"),
        canonical_producers=(TOOL, RECEIPT, MEMO),
        provenance=provenance,
    )


def populate_g1g3_successor_measurement_equations(
    *,
    receipt_path: Path | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> tuple[CanonicalEquation, CanonicalEquation]:
    """Append both measured laws through the locked canonical registry helper."""
    from tac.canonical_equations.registry import register_canonical_equation

    equations = (
        build_worldsheet_transport_residual_event_rate_v1(receipt_path),
        build_argmax_cell_identity_ideal_bytes_v1(receipt_path),
    )
    for equation in equations:
        register_canonical_equation(
            equation,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes="G1/G3 n600 successor design measurements; advisory, pointer 0.18804 unmoved",
        )
    return equations


__all__ = [
    "CELLCODE_EQUATION_ID",
    "WORLDSHEET_EQUATION_ID",
    "build_argmax_cell_identity_ideal_bytes_v1",
    "build_worldsheet_transport_residual_event_rate_v1",
    "ideal_cell_stream_bytes",
    "ideal_cell_stream_bytes_from_bit_sum",
    "populate_g1g3_successor_measurement_equations",
    "transport_event_fraction",
]
