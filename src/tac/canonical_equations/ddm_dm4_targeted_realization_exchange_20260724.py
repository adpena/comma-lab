# SPDX-License-Identifier: MIT
"""Append the DM4 scorer-recursive anchor to the DM2 exchange law.

The exchange equation remains the landed DM2 callable.  DM4 contributes a new
empirical anchor: exact resize-adjoint/Fisher support, measured ERF-r50,
stride-2 SegNet stem blocks, genuine #502 frames, and Pose xi6 secants reduce
the measured constructive upper bound without claiming a minimum or score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tac.canonical_equations.ddm_dm2_semantic_realization_exchange import (
    EQUATION_ID,
    RATE_WEIGHT,
    SOURCE_VIDEO_BYTES,
    semantic_realization_exchange,
)
from tac.canonical_equations.equation import (
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

REPO = Path(__file__).resolve().parents[3]
RECEIPT_REL = (
    ".omx/research/ddm_dm4_targeted_realization_cures_20260724T142722Z/ddm_dm4_targeted_realization_cures_receipt.json"
)
RECEIPT = REPO / RECEIPT_REL
RECEIPT_SHA256 = "9644ef24c8037485a6350193d9368f65f463ae102db70c3d1412a550362bf5bb"
ANCHOR_ID = "ddm_dm4_scorer_recursive_targeted_realization_cures_20260724"
MEASUREMENT_UTC = "2026-07-24T15:09:20Z"
AXIS = "[macOS-CPU frozen-scorer advisory]"


def _load_bound_receipt(path: Path = RECEIPT) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RECEIPT_SHA256:
        raise ValueError("DM4 receipt SHA-256 differs")
    payload = json.loads(raw)
    aggregate = payload.get("aggregate", {})
    recursive = payload.get("custody", {}).get("scorer_recursive_support", {})
    if (
        payload.get("schema") != "ddm_dm4_targeted_realization_cures.v1"
        or payload.get("row_count") != 25
        or aggregate.get("semantic_records_joint_exact_after_composition") is not True
        or payload.get("score_claim") is not False
        or payload.get("pointer_moved") is not False
        or payload.get("main_review_required") is not True
        or recursive.get("stem_stride") != 2
        or recursive.get("erf_r50_pixels") != 85.0
        or "never disk radii" not in recursive.get("write_support_rule", "")
    ):
        raise ValueError("DM4 receipt authority or scorer-recursive contract differs")
    return payload


def build_dm4_exchange_anchor(*, source_receipt: Path = RECEIPT) -> EmpiricalAnchor:
    """Build the SHA-bound DM4 empirical anchor for the DM2 exchange law."""

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
    measured = aggregate["ratio"]
    residual = abs(
        predicted["effective_bytes_per_semantic_byte"] - float(measured["effective_bytes_per_semantic_byte"])
    )
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=RECEIPT_SHA256,
        source_path=RECEIPT_REL,
        captured_at_utc=MEASUREMENT_UTC,
    )
    return EmpiricalAnchor(
        anchor_id=ANCHOR_ID,
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "semantic_rows": 25,
            "targeted_rows": payload["targeted_row_indices"],
            "semantic_joint_bytes": semantic_bytes,
            "realized_rgb_joint_bytes": realized_bytes,
            "joint_collateral_score_delta": collateral_delta,
            "source_video_bytes": SOURCE_VIDEO_BYTES,
            "rate_weight": RATE_WEIGHT,
            "receipt_sha256": RECEIPT_SHA256,
            "write_support": ("exact shared-resize Fisher adjoint x measured ERF-r50 x stride-2 SegNet stem lattice"),
            "historical_control_label": "[naive-menu upper bound]",
        },
        predicted_output=predicted,
        empirical_output={
            "semantic_records_joint_exact_after_composition": True,
            "realized_bytes_per_semantic_byte": float(measured["realized_bytes_per_semantic_byte"]),
            "effective_bytes_per_semantic_byte": float(measured["effective_bytes_per_semantic_byte"]),
            "old_dm2_effective_bytes_per_semantic_byte": float(measured["old_dm2_effective_bytes_per_semantic_byte"]),
            "ratio_fraction_of_dm2": float(measured["ratio_fraction_of_dm2"]),
            "bound_status": measured["bound_status"],
            "union_conflict_pair_ids": aggregate["union_conflict_pair_ids"],
            "fallback_pair_ids": aggregate["fallback_pair_ids"],
            "pose_cured_row_indices": [5, 23],
            "localized_global_tail_cured_row_indices": [19],
            "joint_score_delta": aggregate["joint_score_accounting"]["joint_score_delta"],
            "score_claim": False,
            "pointer_moved": False,
            "main_review_required": True,
        },
        residual=residual,
        source_artifact=RECEIPT_REL,
        measurement_method=(
            "SHA-bound DM1 exact rows -> exact shared-resize Fisher-margin "
            "adjoint -> empirical ERF-r50/stride-2 stem support -> genuine "
            "#502 curvelet/shearlet or Pose xi6 secant candidates -> canonical "
            "factor2 R/uint8 -> frozen SegNet hard L4 and PoseNet first-six "
            "checks -> exact zlib9/lzma9 parse-back pricing -> fresh 25-row "
            "non-telescoping composition"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def append_dm4_exchange_anchor(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append DM4 to the existing law and refresh its residual posterior."""

    from tac.canonical_equations.bayesian_posterior_update import (
        append_empirical_anchor_to_equation_with_posterior_update,
    )

    anchor = build_dm4_exchange_anchor(source_receipt=source_receipt)
    updated, _posterior = append_empirical_anchor_to_equation_with_posterior_update(
        EQUATION_ID,
        anchor,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "DM4 scorer-recursive targeted cures; exact 25-row composition; "
            "constructive upper bound; score_claim=false; pointer unmoved; "
            "MAIN review required"
        ),
    )
    return updated


__all__ = [
    "ANCHOR_ID",
    "RECEIPT_SHA256",
    "append_dm4_exchange_anchor",
    "build_dm4_exchange_anchor",
]
