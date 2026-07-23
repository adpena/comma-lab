#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the AT1 evaluator-semantics divergence receipt.

The tool reads already-landed, research-only receipts.  It does not execute a
scorer, decode a video, or mutate any source evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from tac.canonical_equations.ddm_runtime_export_identity_20260723 import score_row
from tac.contest_score import compute_contest_score
from tac.optimization.scorer_analytic_atlas import (
    fp32_aggregation_order_envelope,
)
from tac.optimization.scorer_module_inventory import (
    canonical_json_bytes,
    read_and_validate_receipt,
    sha256_file,
)

SCHEMA = "ddm_at1_scorer_semantic_divergence_receipt.v1"


class DivergenceReceiptError(ValueError):
    """A source receipt or semantic comparison failed closed."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DivergenceReceiptError(f"{path}: expected JSON object")
    return value


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _batch_envelope(batch_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cursor = 0
    for path in sorted(batch_root.glob("batch_*.json")):
        row = _load_json(path)
        pair_ids = row.get("source_pair_ids")
        if (
            not isinstance(pair_ids, list)
            or not pair_ids
            or pair_ids != list(range(cursor, cursor + len(pair_ids)))
        ):
            raise DivergenceReceiptError(f"{path}: noncontiguous pair coverage")
        cursor += len(pair_ids)
        pose_coordinates = int(row["pose_coordinates"])
        if pose_coordinates != 6 * len(pair_ids):
            raise DivergenceReceiptError(f"{path}: Pose coordinate count is not 6/pair")
        sites = int(row["sites"])
        if sites != len(pair_ids) * 384 * 512:
            raise DivergenceReceiptError(f"{path}: Seg site count changed")
        rows.append(
            {
                "pair_range": [pair_ids[0], pair_ids[-1] + 1],
                "pose_batch_sum_of_pair_mse": (
                    float(row["pose_squared_error_sum"]) / 6.0
                ),
                "seg_batch_sum_of_pair_means": int(row["errors"]) / (384 * 512),
                "source": _identity(path),
            }
        )
    if cursor != 600 or len(rows) != 38:
        raise DivergenceReceiptError(
            f"expected 38 batches covering n600, observed {len(rows)}/{cursor}"
        )
    envelope = fp32_aggregation_order_envelope(
        pose_batch_sums=[
            float(row["pose_batch_sum_of_pair_mse"]) for row in rows
        ],
        seg_batch_sums=[
            float(row["seg_batch_sum_of_pair_means"]) for row in rows
        ],
    )
    return envelope, rows


def _wrapper_delta(
    *,
    name: str,
    verification_path: Path,
    harness_path: Path,
) -> dict[str, Any]:
    verification = _load_json(verification_path)
    harness = _load_json(harness_path)
    internal = verification["score"]
    upstream = harness["parsed_report"]
    internal_pose = float(internal["d_pose"])
    internal_seg = float(internal["d_seg"])
    upstream_pose = float(upstream["d_pose"])
    upstream_seg = float(upstream["d_seg"])
    internal_score = compute_contest_score(
        internal_seg,
        internal_pose,
        int(internal["archive_bytes"]),
    )
    equation_score = score_row(
        archive_bytes=int(internal["archive_bytes"]),
        d_seg=internal_seg,
        d_pose=internal_pose,
    )["total"]
    return {
        "wrapper": name,
        "internal": {"d_pose": internal_pose, "d_seg": internal_seg},
        "upstream_report_printed_8dp": {
            "d_pose": upstream_pose,
            "d_seg": upstream_seg,
        },
        "delta_internal_minus_upstream_printed": {
            "d_pose": internal_pose - upstream_pose,
            "d_seg": internal_seg - upstream_seg,
            "pose_score_term": (
                math.sqrt(10.0 * internal_pose)
                - math.sqrt(10.0 * upstream_pose)
            ),
            "seg_score_term": 100.0 * (internal_seg - upstream_seg),
            "combined_distortion_terms": (
                math.sqrt(10.0 * internal_pose)
                - math.sqrt(10.0 * upstream_pose)
                + 100.0 * (internal_seg - upstream_seg)
            ),
        },
        "interpretation": (
            "upper-bound comparison is quantized by upstream report formatting "
            "to eight decimals; internal meter uses cached GT cells/poses and "
            "fp64 aggregation, then a separate upstream harness runs evaluate.py"
        ),
        "canonical_score_helper_parity": {
            "tac_contest_score": internal_score,
            "ddm_equation_score_row": equation_score,
            "absolute_difference": abs(internal_score - equation_score),
            "status": (
                "EXACT_REAL_ARITHMETIC_FORMULA_MATCH"
                if internal_score == equation_score
                else "MISMATCH"
            ),
        },
        "sources": {
            "internal_verification": _identity(verification_path),
            "upstream_harness": _identity(harness_path),
        },
    }


def _wrap(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "body": body,
        "body_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _write_once(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite different receipt: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--e1-verification", required=True, type=Path)
    parser.add_argument("--e1-harness", required=True, type=Path)
    parser.add_argument("--e2-verification", required=True, type=Path)
    parser.add_argument("--e2-harness", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--created-at-utc", required=True)
    args = parser.parse_args()

    inventory = read_and_validate_receipt(args.inventory)
    envelope, batch_rows = _batch_envelope(args.batch_root)
    wrappers = [
        _wrapper_delta(
            name="E1",
            verification_path=args.e1_verification,
            harness_path=args.e1_harness,
        ),
        _wrapper_delta(
            name="E2",
            verification_path=args.e2_verification,
            harness_path=args.e2_harness,
        ),
    ]
    if any(
        row["canonical_score_helper_parity"]["status"]
        != "EXACT_REAL_ARITHMETIC_FORMULA_MATCH"
        for row in wrappers
    ):
        raise DivergenceReceiptError("canonical score helpers diverged")

    drift = inventory["body"]["source_strata"]["B_imported_library_sources"][
        "version_drift"
    ]
    body = {
        "schema": SCHEMA,
        "created_at_utc": args.created_at_utc,
        "research_only": True,
        "score_claim": False,
        "execution_allowed": False,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "first_rung": True,
        "pair_count": 600,
        "inventory": {
            **_identity(args.inventory),
            "body_sha256": inventory["body_sha256"],
        },
        "ranked_divergences": [
            {
                "rank": 1,
                "impact": "HIGH_BINDING_BLOCKER",
                "id": "locked_library_sources_not_materialized",
                "status": "CONFIRMED",
                "finding": (
                    "Observed import sources differ from upstream uv.lock on "
                    f"{len(drift)} packages; the graph inventory is not locked "
                    "contest evaluator-source authority."
                ),
                "details": drift,
                "required_action": (
                    "Materialize the exact lock-selected evaluator environment, "
                    "rerun this inventory, and require zero version drift before "
                    "consuming library-source-bound closed forms."
                ),
                "verdict_scope": (
                    "library source binding only; immutable modules.py and frozen "
                    "checkpoint byte inventory remain exact"
                ),
            },
            {
                "rank": 2,
                "impact": "HIGH_DERIVATIVE_PORT_REQUIREMENT",
                "id": "upstream_rgb_to_yuv6_no_grad",
                "status": "CONFIRMED",
                "finding": (
                    "upstream frame_utils.rgb_to_yuv6 is decorated torch.no_grad, "
                    "so unpatched Pose input gradients are identically severed."
                ),
                "required_action": (
                    "Atlas gaze materializers must use the hash-stamped "
                    "tac.scorer.make_scorers_differentiable port and retain its "
                    "upstream-mirror fidelity proof."
                ),
                "verdict_scope": (
                    "derivatives only; upstream forward/evaluator outputs remain "
                    "the frozen authority"
                ),
            },
            {
                "rank": 3,
                "impact": "MEDIUM_EXACT_SCORE_CUSTODY",
                "id": "internal_wrapper_vs_upstream_evaluator",
                "status": "MEASURED_PRINTED_PRECISION_BOUND",
                "finding": (
                    "E1/E2 internal wrappers use cached GT cells/poses and fp64 "
                    "aggregation; separate upstream harness reports differ at the "
                    "few-micro-score-unit level after eight-decimal formatting."
                ),
                "details": wrappers,
                "required_action": (
                    "Keep internal rows advisory and require upstream evaluate.py "
                    "on the receiver-closed bytes for any score claim."
                ),
                "verdict_scope": "E1/E2 receipts named here only",
            },
            {
                "rank": 4,
                "impact": "LOW_BUT_REAL_FP32_REDUCTION",
                "id": "fp32_zero_dim_batch_accumulation_order",
                "status": "DERIVED_FROM_MEASURED_N600_BATCH_ROWS",
                "finding": (
                    "Changing only the order of the 38 fp32 batch scalars spans "
                    f"{envelope['score_span_upper_bound_if_term_extrema_cooccur']:.12g} "
                    "score units in the measured v19b row."
                ),
                "details": envelope,
                "required_action": (
                    "Exact mirrors must preserve batch size, pair order, fp32 "
                    "zero-dimensional accumulators, and device axis."
                ),
                "verdict_scope": (
                    "v19b measured batch scalars; this is not a cross-hardware bound"
                ),
            },
            {
                "rank": 5,
                "impact": "NO_DIVERGENCE_REAL_ARITHMETIC",
                "id": "canonical_score_formula",
                "status": "CONFIRMED",
                "finding": (
                    "tac.contest_score.compute_contest_score and the E1/E2 score_row "
                    "helper agree exactly on both named internal rows."
                ),
                "details": [
                    row["canonical_score_helper_parity"] for row in wrappers
                ],
                "verdict_scope": (
                    "real arithmetic formula only; evaluator reduction and forward "
                    "custody are separate"
                ),
            },
            {
                "rank": 6,
                "impact": "NO_DIVERGENCE_STRUCTURAL",
                "id": "seg_pair_weight_frame_slice_pose6_and_remainder",
                "status": "CONFIRMED_FROM_IMMUTABLE_SOURCE",
                "finding": (
                    "Uniform 384x512 Seg sites make per-pair and global-pixel "
                    "weighting equal; Seg consumes frame1 only; Pose consumes both "
                    "frames but prices output coordinates 0:6; n600 batches are "
                    "37x16 plus one remainder batch of 8."
                ),
                "verdict_scope": "current immutable upstream source hash only",
            },
            {
                "rank": 7,
                "impact": "LOW_EVAL_INERT",
                "id": "posenet_num_batches_tracked_absent",
                "status": "CONFIRMED",
                "finding": (
                    "The Pose checkpoint omits 88 num_batches_tracked buffers that "
                    "PyTorch BatchNorm load compatibility defaults; eval-mode "
                    "closed forms consume running mean/variance, not those counters."
                ),
                "verdict_scope": (
                    "eval-mode frozen checkpoint load only; training/resume semantics "
                    "are outside this atlas"
                ),
            },
        ],
        "fp32_batch_source_rows": batch_rows,
        "consumer_gate": {
            "status": "BLOCKED_LOCKED_LIBRARY_SOURCE_NOT_MATERIALIZED",
            "allows": [
                "checkpoint tensor inventory",
                "immutable upstream composition laws",
                "advisory factor schema and controller wiring",
            ],
            "refuses": [
                "locked-source closed-form materialization",
                "n600 exact gaze claim",
                "contest score claim",
            ],
        },
    }
    receipt = _wrap(body)
    _write_once(args.output, receipt)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "body_sha256": receipt["body_sha256"],
                "fp32_score_span": envelope[
                    "score_span_upper_bound_if_term_extrema_cooccur"
                ],
                "consumer_gate": body["consumer_gate"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
