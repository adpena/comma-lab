# SPDX-License-Identifier: MIT
"""Canonical G2 bucket to five-type description-stream correspondence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.optimization.ddm_min_description_contract import StreamType
from tac.provenance.builders import build_provenance_for_research_sidecar

REPO = Path(__file__).resolve().parents[3]
UTC = "2026-07-24T03:30:00Z"
EQUATION_ID = "ddm_g2_five_type_correspondence_v1"
G2_RECEIPT = ".omx/research/ddm_g2_solve_diff_op_mining_n600_20260722T194000Z/aggregate_ledger.json"
G2_RECEIPT_SHA256 = "061220fd8c1ca047b210841235fc805194a96175e933ee110ba4ac8bb2077d84"

_CORRESPONDENCE = {
    "scorer-invisible": (StreamType.GAUGE.value,),
    "xi-predictable": (StreamType.CONNECTION.value,),
    "chart-expressible": (
        StreamType.SKELETON.value,
        StreamType.FIBER.value,
    ),
    "irreducible": (StreamType.RESIDUAL.value,),
}


def g2_bucket_to_stream_types(bucket: str) -> tuple[str, ...]:
    """Return the lawful description type(s) for one sealed G2 bucket."""

    if not isinstance(bucket, str) or bucket not in _CORRESPONDENCE:
        raise ValueError("bucket must be one of scorer-invisible, xi-predictable, chart-expressible, irreducible")
    return _CORRESPONDENCE[bucket]


def evaluate_ddm_g2_five_type_correspondence(
    inputs: Mapping[str, Any],
) -> tuple[str, ...]:
    """Uniform evaluator adapter registered for LawRef consumers."""

    if not isinstance(inputs, Mapping) or set(inputs) != {"bucket"}:
        raise ValueError("inputs must contain exactly one bucket")
    return g2_bucket_to_stream_types(inputs["bucket"])


register_evaluator(
    EQUATION_ID,
    evaluate_ddm_g2_five_type_correspondence,
)


def _g2_receipt() -> dict[str, Any]:
    source = REPO / G2_RECEIPT
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != G2_RECEIPT_SHA256:
        raise ValueError("G2 receipt SHA-256 differs")
    payload = json.loads(raw)
    if (
        payload.get("schema") != "solve_diff_aggregate_ledger.v1"
        or payload.get("research_only") is not True
        or payload.get("score_claim") is not False
        or payload.get("pointer_moved") is not False
        or payload.get("archive_emitted") is not False
        or payload.get("immutable_stage_validation", {}).get("pair_count") != 600
        or payload.get("candidate_admission", {}).get("status") != "BLOCKED_NO_RECEIVER_DELTA_DSEG"
    ):
        raise ValueError("G2 receipt authority or scope firewall differs")
    return payload


def build_ddm_g2_five_type_correspondence_v1() -> CanonicalEquation:
    """Build the scoped law with the preserved G2 n600 receipt as anchor."""

    payload = _g2_receipt()
    endpoint = payload["endpoint_delta"]
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=G2_RECEIPT,
        reactivation_criteria=(
            "Re-anchor after a receiver-closed coefficient perturbation assigns "
            "actual positive byte homes to the G2 buckets. Byte rank alone never "
            "admits a carrier or mutates the frontier."
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="darwin_arm64_cpu_torch_threads4",
        captured_at_utc=UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="ddm_g2_n600_operator_bucket_crosswalk_20260722",
        measurement_utc="2026-07-22T20:01:21Z",
        inputs={
            "pair_count": 600,
            "typed_member_rows": payload["immutable_stage_validation"]["typed_member_rows"],
            "typed_member_bytes": payload["immutable_stage_validation"]["typed_member_bytes"],
            "source_sha256": G2_RECEIPT_SHA256,
            "archive_emitted": False,
            "score_claim": False,
        },
        predicted_output={
            "scorer-invisible": ["GAUGE"],
            "xi-predictable": ["CONNECTION"],
            "chart-expressible": ["SKELETON", "FIBER"],
            "irreducible": ["RESIDUAL"],
        },
        empirical_output={
            "range_fraction_weighted": endpoint["range_fraction_weighted"],
            "ker_fraction_weighted": endpoint["ker_fraction_weighted"],
            "candidate_operator_byte_waterfall": payload["candidate_operator_byte_waterfall"],
            "candidate_admission": payload["candidate_admission"]["status"],
            "xi_only_transport_predictive": False,
            "correspondence_scope": (
                "type vocabulary crosswalk only; the receipt does not empirically "
                "admit any operator or classify every coefficient"
            ),
        },
        residual=0.0,
        source_artifact=G2_RECEIPT,
        measurement_method=(
            "source inspection of the immutable n600 G2 aggregate ledger plus "
            "the settled four-way recursive-scorer bucket law; residual counts "
            "crosswalk-key disagreements only, not carrier efficacy"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        noise_floor=None,
        noise_floor_provenance=("deterministic schema crosswalk; empirical carrier efficacy remains blocked"),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM G2 five-type correspondence",
        one_line_summary=(
            "Map recursive-scorer buckets to GAUGE, CONNECTION, SKELETON/FIBER, or RESIDUAL stream roles."
        ),
        latex_form=(
            r"\ker(R)\mapsto G,\quad \Xi\text{-predictable}\mapsto C,\quad "
            r"\text{chart-expressible}\mapsto S\oplus F,\quad "
            r"\text{irreducible}\mapsto E"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_five_type_correspondence_20260724:g2_bucket_to_stream_types"
        ),
        domain_of_validity={
            "input_buckets": list(_CORRESPONDENCE),
            "stream_type_vocabulary": [member.value for member in StreamType],
            "chart_expressible_is_two_part": True,
            "connection_operator_code_free": True,
            "connection_video_derived_parameters_counted": True,
            "gauge_counted_bytes": 0,
            "excluded": [
                "per-coefficient classification without byte-home custody",
                "carrier admission from coded-byte rank",
                "receiver-visible efficacy",
                "contest score or frontier movement",
            ],
            "verdict_scope": (
                "FORMULATION: correspondence of four recursive-scorer buckets "
                "to the five stream types; not an empirical carrier verdict"
            ),
            "score_claim": False,
        },
        units_in={"bucket": "recursive_scorer_bucket"},
        units_out={"stream_types": "ordered_stream_type_tuple"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"source_crosswalk_key_disagreement_count": 0.0},
        last_calibration_utc=UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.ddm_min_description_contract.TypedStreamTag",
            "DDM RD1 type column after byte-home custody closure",
        ),
        canonical_producers=(
            G2_RECEIPT,
            "tac.canonical_equations.ddm_five_type_correspondence_20260724",
        ),
        provenance=provenance,
    )


def populate_ddm_g2_five_type_correspondence_v1(
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_g2_five_type_correspondence_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "G2 five-type correspondence with preserved n600 research anchor; "
            "no carrier admission, score claim, or pointer movement; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "G2_RECEIPT",
    "G2_RECEIPT_SHA256",
    "build_ddm_g2_five_type_correspondence_v1",
    "evaluate_ddm_g2_five_type_correspondence",
    "g2_bucket_to_stream_types",
    "populate_ddm_g2_five_type_correspondence_v1",
]
