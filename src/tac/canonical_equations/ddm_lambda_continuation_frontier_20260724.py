# SPDX-License-Identifier: MIT
"""Canonical restricted realized λ-continuation law for DDM RD1."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.optimization.ddm_lambda_continuation_frontier import (
    EVIDENCE_AXIS,
    realized_distortion,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

REPO = Path(__file__).resolve().parents[3]
UTC = "2026-07-24T01:55:00Z"
EQUATION_ID = "ddm_restricted_realized_lambda_continuation_v1"
RECEIPT = (
    ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
    "ddm_rd1_lambda_continuation_frontier_receipt_v2.json"
)
RECEIPT_SHA256 = "cdfa9a400d9633ea7f8f698dee6d55c65ac478ddcfed3ec01e6d2e6cefe6bbae"


def restricted_realized_lambda_argmin(
    lambda_value: float,
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    """Return the deterministic finite-domain minimizer of ``R + lambda*D``.

    This callable deliberately accepts measured typed rows rather than reading
    the canonical receipt.  The result is therefore a pure function suitable
    for LawRef evaluation.  It is not a global uint8-lattice optimizer.
    """

    lam = float(lambda_value)
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("lambda must be finite and nonnegative")
    if not isinstance(candidates, Sequence) or not candidates:
        raise ValueError("candidates must be a nonempty sequence")
    ranked: list[tuple[float, float, int, str]] = []
    seen = set()
    for row in candidates:
        if not isinstance(row, Mapping):
            raise ValueError("each candidate must be a mapping")
        candidate_id = row.get("candidate_id")
        counted_bytes = row.get("counted_bytes")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise ValueError("candidate ids must be nonempty and unique")
        seen.add(candidate_id)
        if isinstance(counted_bytes, bool) or not isinstance(counted_bytes, int) or counted_bytes <= 0:
            raise ValueError("counted_bytes must be a positive integer")
        if row.get("score_claim") is not False:
            raise ValueError("restricted evaluator accepts false-authority rows only")
        if row.get("pair_count") != 600:
            raise ValueError("restricted evaluator accepts exact n600 rows only")
        if row.get("own_stored_problem") is not True or row.get("donor_conditioned") is not False:
            raise ValueError("restricted evaluator rejects donor-conditioned rows")
        distortion = realized_distortion(float(row["d_seg"]), float(row["d_pose"]))
        ranked.append(
            (
                counted_bytes + lam * distortion,
                distortion,
                counted_bytes,
                candidate_id,
            )
        )
    return min(ranked)[-1]


def evaluate_ddm_restricted_realized_lambda_continuation(
    inputs: Mapping[str, Any],
) -> str:
    """Uniform evaluator adapter registered for LawRef consumers."""

    return restricted_realized_lambda_argmin(
        float(inputs["lambda"]),
        inputs["candidates"],
    )


register_evaluator(
    EQUATION_ID,
    evaluate_ddm_restricted_realized_lambda_continuation,
)


def _receipt(path: Path | None = None) -> tuple[dict[str, Any], str]:
    source = path or REPO / RECEIPT
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "ddm_rd1_lambda_continuation_frontier_receipt.v1":
        raise ValueError("RD1 receipt schema differs")
    if (
        payload.get("evidence_axis") != EVIDENCE_AXIS
        or payload.get("score_claim") is not False
        or payload.get("pointer_moved") is not False
        or payload.get("objective", {}).get("global_uint8_lattice_optimality_claim") is not False
    ):
        raise ValueError("RD1 authority or scope firewall differs")
    if len(payload.get("continuation", [])) < 8:
        raise ValueError("RD1 continuation does not contain the required lambda ladder")
    return payload, str(source if path is not None else RECEIPT)


def build_ddm_restricted_realized_lambda_continuation_v1(
    receipt_path: Path | None = None,
) -> CanonicalEquation:
    """Build the measured restricted-domain equation and empirical anchor."""

    payload, source = _receipt(receipt_path)
    mismatches = sum(row["restricted_global_rank_verified"] is not True for row in payload["continuation"])
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=RECEIPT,
        reactivation_criteria=(
            "Recompute after any new archive-receiver-closed n600 point, "
            "donor-free MS1 landing, scorer-axis replay, or Menu1 deployment closure. "
            "Never extrapolate finite-domain support to the global uint8 lattice."
        ),
        measurement_axis=EVIDENCE_AXIS,
        hardware_substrate="darwin_arm64_cpu_torch_threads4",
        captured_at_utc=UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="ddm_rd1_n600_restricted_lambda_sweep_20260724",
        measurement_utc=UTC,
        inputs={
            "candidate_count": payload["objective"]["restricted_domain_candidate_count"],
            "supported_hull_count": len(payload["supported_hull"]),
            "lambda_count": len(payload["continuation"]),
            "lambda_ladder": payload["lambda_ladder"],
            "typed_config_sha256": payload["typed_config_sha256"],
            "pair_count": 600,
            "threads": 4,
            "donor_conditioned": False,
        },
        predicted_output={
            "law": "x*(lambda)=argmin_x R_counted(x)+lambda*(100*d_seg(x)+sqrt(10*d_pose(x)))",
            "domain": "SHA-custodied measured n600 descriptions only",
            "corrector": "neighbor-only with full-rank finite-domain check",
        },
        empirical_output={
            "selected_candidate_ids": [row["selected_candidate_id"] for row in payload["continuation"]],
            "supported_hull_candidate_ids": [row["candidate_id"] for row in payload["supported_hull"]],
            "duals": payload["duals"],
            "knee_candidate_id": payload["knee"]["candidate_id"],
            "R6_CANDIDATE": payload["knee"]["R6_CANDIDATE"],
            "verdict": payload["verdict"],
            "verdict_scope": payload["verdict_scope"],
        },
        residual=float(mismatches),
        source_artifact=source,
        measurement_method=(
            "fresh four-thread frozen CPU scorer replay of the exact C1 receiver "
            "output; immutable reaggregation of Menu1 n600 batches; SHA/ZIP-byte-home "
            "verification of 104 V19C accepted archives; adjacent finite-hull continuation"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=(
            "fresh local C1 row differs from the preserved contest-CPU display; "
            "axes remain separate and no cross-axis noise floor is inferred"
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM restricted realized lambda continuation",
        one_line_summary=(
            "On a SHA-custodied n600 description set, choose the real counted-byte "
            "plus lambda-weighted realized-distortion minimizer."
        ),
        latex_form=(
            r"x_{\mathcal C}^{*}(\lambda)=\arg\min_{x\in\mathcal C}"
            r"\left[R_{\rm counted}(x)+\lambda"
            r"\left(100d_{\rm seg}(x)+\sqrt{10d_{\rm pose}(x)}\right)\right]"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_lambda_continuation_frontier_20260724:restricted_realized_lambda_argmin"
        ),
        domain_of_validity={
            "optimization_domain": "SHA-custodied measured n600 descriptions only",
            "receiver_closures": [
                "archive_receiver_closed",
                "measurement_harness_receiver_closed",
            ],
            "two_type_rate": "skeleton tokens x fiber coefficients",
            "own_stored_problem_only": True,
            "donor_conditioned": False,
            "neighbor_only_corrector": True,
            "full_rank_finite_domain_check": True,
            "excluded": [
                "global uint8-lattice optimality",
                "contest score or frontier mutation",
                "Menu1 contest archive closure",
                "cross-axis equality inference",
                "additive combination with solver_member_selection",
            ],
            "verdict_scope": payload["verdict_scope"],
            "score_claim": False,
        },
        units_in={
            "lambda": "counted_bytes_per_distortion_unit",
            "counted_bytes": "bytes",
            "d_seg": "dimensionless",
            "d_pose": "dimensionless",
        },
        units_out={"candidate_id": "typed_description_identifier"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"finite_domain_full_rank_mismatch_count": float(mismatches)},
        last_calibration_utc=UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.measure_ddm_rd1_lambda_continuation_frontier",
            "DDM train-decision SOLVE column",
        ),
        canonical_producers=(
            "tools/measure_ddm_rd1_lambda_continuation_frontier.py",
            RECEIPT,
        ),
        provenance=provenance,
    )


def populate_ddm_restricted_realized_lambda_continuation_v1(
    *,
    receipt_path: Path | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the measured law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_restricted_realized_lambda_continuation_v1(receipt_path)
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "RD1 restricted n600 lambda continuation; four-thread local advisory; "
            "stable v2 receipt excludes volatile exact free-space observations; "
            "V19C current unsupported; knee outside R6; pointer unmoved; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "RECEIPT_SHA256",
    "build_ddm_restricted_realized_lambda_continuation_v1",
    "evaluate_ddm_restricted_realized_lambda_continuation",
    "populate_ddm_restricted_realized_lambda_continuation_v1",
    "restricted_realized_lambda_argmin",
]
