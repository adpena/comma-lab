# SPDX-License-Identifier: MIT
"""Canonical exchange law for DM2 semantic records and realized RGB writes.

The semantic price is the exact DM1 record price.  The realized price is the
exact coded L3 RGB record plus only *positive* off-support Seg/Pose collateral,
converted to a byte equivalent at the contest rate dual.  Negative collateral
does not create a fictitious byte credit.

This law is research-only and anchored to the SHA-bound DM2 25-row receipt on
the ``[macOS-CPU frozen-scorer advisory]`` axis.  It is not a contest score or
a minimum-preimage certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "ddm_dm2_semantic_to_realized_rgb_exchange_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT_REL = (
    ".omx/research/ddm_dm2_l3_realization_race_25_rows_20260724T133300Z/"
    "ddm_dm2_l3_realization_race_receipt.json"
)
RECEIPT = REPO / RECEIPT_REL
RECEIPT_SHA256 = "8897241b7fc0ded7d4d6d1100c4d23ea162111050754e36c8ad8b3e57e294229"
MEASUREMENT_UTC = "2026-07-24T14:00:57Z"
AXIS = "[macOS-CPU frozen-scorer advisory]"
SOURCE_VIDEO_BYTES = 37_545_489
RATE_WEIGHT = 25.0


def semantic_realization_exchange(
    *,
    semantic_bytes: int,
    realized_rgb_bytes: int,
    collateral_score_delta: float,
    source_video_bytes: int = SOURCE_VIDEO_BYTES,
    rate_weight: float = RATE_WEIGHT,
) -> dict[str, float]:
    """Return measured and collateral-adjusted realized/semantic ratios.

    ``collateral_score_delta`` contains off-target Seg score delta plus the
    affected-pair Pose score delta.  Only its positive part is charged.
    """

    integer_values = (semantic_bytes, realized_rgb_bytes, source_video_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
        raise ValueError("byte counts must be integers")
    if semantic_bytes <= 0 or realized_rgb_bytes < 0 or source_video_bytes <= 0:
        raise ValueError("require semantic/source bytes > 0 and realized bytes >= 0")
    numeric_values = (collateral_score_delta, rate_weight)
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in numeric_values
    ):
        raise ValueError("score delta and rate weight must be finite numbers")
    if rate_weight <= 0.0:
        raise ValueError("rate weight must be positive")

    collateral_byte_equivalent = (
        max(0.0, float(collateral_score_delta))
        * source_video_bytes
        / float(rate_weight)
    )
    effective_bytes = realized_rgb_bytes + collateral_byte_equivalent
    return {
        "positive_collateral_byte_equivalent_at_rate_dual": (
            collateral_byte_equivalent
        ),
        "realized_bytes_per_semantic_byte": realized_rgb_bytes / semantic_bytes,
        "effective_realized_plus_positive_collateral_bytes": effective_bytes,
        "effective_bytes_per_semantic_byte": effective_bytes / semantic_bytes,
    }


def _load_bound_receipt(path: Path = RECEIPT) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RECEIPT_SHA256:
        raise ValueError("DM2 receipt SHA-256 differs")
    payload = json.loads(raw)
    if (
        payload.get("schema") != "ddm_dm2_l3_realization_race.v1"
        or payload.get("row_count") != 25
        or payload.get("aggregate", {}).get(
            "semantic_records_joint_exact_after_composition"
        )
        is not True
        or payload.get("score_claim") is not False
        or payload.get("pointer_moved") is not False
    ):
        raise ValueError("DM2 receipt authority contract differs")
    return payload


def build_ddm_dm2_semantic_to_realized_rgb_exchange_v1(
    *, source_receipt: Path = RECEIPT
) -> CanonicalEquation:
    """Build the exchange law with the exact 25-row DM2 empirical anchor."""

    payload = _load_bound_receipt(source_receipt)
    aggregate = payload["aggregate"]
    semantic_bytes = int(aggregate["semantic_bytes_dm1_joint"])
    realized_bytes = int(aggregate["realized_rgb_joint"]["exact_counted_bytes"])
    collateral_delta = float(aggregate["collateral"]["joint_collateral_score_delta"])
    predicted = semantic_realization_exchange(
        semantic_bytes=semantic_bytes,
        realized_rgb_bytes=realized_bytes,
        collateral_score_delta=collateral_delta,
    )
    measured_ratio = aggregate["ratio"]
    residual = abs(
        predicted["effective_bytes_per_semantic_byte"]
        - float(measured_ratio["effective_bytes_per_semantic_byte"])
    )
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=RECEIPT_SHA256,
        source_path=RECEIPT_REL,
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="ddm_dm2_25_exact_semantic_rows_l3_realization_20260724",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "semantic_rows": 25,
            "semantic_joint_bytes": semantic_bytes,
            "realized_rgb_joint_bytes": realized_bytes,
            "joint_collateral_score_delta": collateral_delta,
            "source_video_bytes": SOURCE_VIDEO_BYTES,
            "rate_weight": RATE_WEIGHT,
            "receipt_sha256": RECEIPT_SHA256,
        },
        predicted_output=predicted,
        empirical_output={
            "semantic_records_joint_exact_after_composition": True,
            "realized_bytes_per_semantic_byte": float(
                measured_ratio["realized_bytes_per_semantic_byte"]
            ),
            "effective_bytes_per_semantic_byte": float(
                measured_ratio["effective_bytes_per_semantic_byte"]
            ),
            "bound_status": measured_ratio["bound_status"],
            "fallback_pair_ids": aggregate["fallback_pair_ids"],
            "joint_score_delta": aggregate["joint_score_accounting"][
                "joint_score_delta"
            ],
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=residual,
        source_artifact=RECEIPT_REL,
        measurement_method=(
            "SHA-bound DM1 exact semantic records -> bounded fixed-quantum L3 "
            "camera preimages -> real factor2 R -> uint8 -> frozen batch-1 "
            "SegNet exact record checks plus PoseNet first-six SSE -> exact "
            "zlib9/lzma9 parse-back pricing; pair compositions freshly remeasured"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DM2 semantic-to-realized RGB exchange with priced collateral",
        one_line_summary=(
            "Exact semantic record bytes exchange for coded L3 RGB bytes plus "
            "only positive off-target Seg/Pose collateral at the contest rate dual."
        ),
        latex_form=(
            r"\rho=\frac{B_{\mathrm{RGB}}+\frac{N}{25}"
            r"[\Delta S_{\mathrm{offseg}}+\Delta S_{\mathrm{pose}}]_+}"
            r"{B_{\mathrm{semantic}}}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_dm2_semantic_realization_exchange:"
            "semantic_realization_exchange"
        ),
        domain_of_validity={
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "axis": AXIS,
            "receiver": "camera 874x1164 pre-R -> exact factor2 R -> uint8 -> frozen SegNet/PoseNet",
            "demand_set": "25 SHA-bound DM1 rows; 1,569 exact joint semantic bytes",
            "realizer": (
                "bounded local/global fixed-quantum candidate menu; full solved-target "
                "positive-control fallback for non-telescoping pair conflicts"
            ),
            "bound_status": "constructive upper bound; no minimum-preimage certificate",
            "collateral_policy": "positive Seg/Pose collateral only; beneficial collateral is not a byte credit",
            "verdict_scope": "INSTANCE x SHA-bound demand set x bounded realization menu",
            "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        },
        units_in={
            "semantic_bytes": "exact DM1 coded bytes",
            "realized_rgb_bytes": "exact coded L3 RGB record bytes",
            "collateral_score_delta": "contest objective score units",
            "source_video_bytes": "bytes",
            "rate_weight": "score units times source bytes per archive byte",
        },
        units_out={
            "positive_collateral_byte_equivalent_at_rate_dual": "bytes",
            "realized_bytes_per_semantic_byte": "dimensionless",
            "effective_realized_plus_positive_collateral_bytes": "bytes",
            "effective_bytes_per_semantic_byte": "dimensionless",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"effective_ratio_absolute": residual},
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "DDM family-(d) emitter admission",
            "DDM family-(b) realization comparison",
            "tac.bit_allocator",
            "tac.cathedral_autopilot",
        ),
        canonical_producers=(
            "tac.optimization.ddm_dm2_l3_realization_race:materialize",
        ),
        provenance=provenance,
    )


def populate_ddm_dm2_semantic_to_realized_rgb_exchange(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the DM2 law through the locked canonical-equation registry."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_dm2_semantic_to_realized_rgb_exchange_v1(
        source_receipt=source_receipt
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "DM2 25-row exact semantic realization; constructive upper bound; "
            "score_claim=false; pointer unmoved; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "AXIS",
    "EQUATION_ID",
    "RECEIPT_SHA256",
    "build_ddm_dm2_semantic_to_realized_rgb_exchange_v1",
    "populate_ddm_dm2_semantic_to_realized_rgb_exchange",
    "semantic_realization_exchange",
]
