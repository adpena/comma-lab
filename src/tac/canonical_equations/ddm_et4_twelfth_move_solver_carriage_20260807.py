# SPDX-License-Identifier: MIT
"""ET4 twelfth-move solver/carriage split law.

This law is non-promoting. It consumes the byte-closed ET4 advisory receipts
and separates the real solver reach from the failed correction carriage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_et4_twelfth_move_solver_carriage_split_v1"
SUMMARY_PATH = ".omx/research/ddm_et4_20260806/et4_solve_within_cvp_summary.json"
BYTECLOSE_RECEIPT_PATH = ".omx/research/ddm_et4_20260806/byteclose_archive_receipt.json"
SOURCE_ARTIFACT = ".omx/research/ddm_et4_20260806/TWELFTH_MOVE_ADJUDICATION.md"
CONTEST_UNCOMPRESSED_BYTES = 37_545_489
S_PER_FLIP = 100.0 / (600.0 * 384.0 * 512.0)
W_BREAK_EVEN_BYTES_PER_FLIP = S_PER_FLIP / (25.0 / CONTEST_UNCOMPRESSED_BYTES)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def et4_solver_carriage_split(
    *,
    net_flip_reduction: float,
    patch_compressed_bytes: float,
    patch_nnz: float,
    baseline_d_seg: float,
    realized_d_seg: float,
    archive_delta_bytes: float,
) -> dict[str, float | bool]:
    """Compute ET4's solver reach and carriage over-break from receipt fields."""

    if net_flip_reduction <= 0:
        raise ValueError("net_flip_reduction must be positive")
    if patch_compressed_bytes < 0 or patch_nnz < 0 or archive_delta_bytes < 0:
        raise ValueError("byte and nnz fields must be non-negative")
    seg_delta_s = 100.0 * (float(realized_d_seg) - float(baseline_d_seg))
    break_even_bytes = float(net_flip_reduction) * W_BREAK_EVEN_BYTES_PER_FLIP
    patch_b_per_flip = float(patch_compressed_bytes) / float(net_flip_reduction)
    archive_rate_delta_s = 25.0 * float(archive_delta_bytes) / CONTEST_UNCOMPRESSED_BYTES
    return {
        "solver_reduces_d_seg": seg_delta_s < 0.0,
        "seg_delta_s": seg_delta_s,
        "break_even_bytes": break_even_bytes,
        "patch_b_per_flip": patch_b_per_flip,
        "patch_over_break_even_ratio": float(patch_compressed_bytes) / break_even_bytes,
        "nnz_per_flip": float(patch_nnz) / float(net_flip_reduction),
        "archive_rate_delta_s": archive_rate_delta_s,
    }


def et4_solver_carriage_split_from_receipts(
    *,
    summary_path: str | Path = SUMMARY_PATH,
    byteclose_receipt_path: str | Path = BYTECLOSE_RECEIPT_PATH,
) -> dict[str, float | bool]:
    """Read ET4 summary/byteclose receipts and compute the split."""

    summary = _read_json(summary_path)
    receipt = _read_json(byteclose_receipt_path)
    aggregate = summary["aggregate"]
    patch = receipt["patch"]
    baseline = aggregate["baseline"]
    archive_delta_bytes = float(receipt["archive"]["archive_bytes"]) - float(
        baseline["archive_bytes"]
    )
    return et4_solver_carriage_split(
        net_flip_reduction=float(aggregate["net_flip_reduction"]),
        patch_compressed_bytes=float(patch["compressed_bytes"]),
        patch_nnz=float(patch["total_nnz"]),
        baseline_d_seg=float(baseline["d_seg"]),
        realized_d_seg=float(aggregate["d_seg_after_completed_scope"]),
        archive_delta_bytes=archive_delta_bytes,
    )


def build_ddm_et4_twelfth_move_solver_carriage_split_v1(
    *,
    summary_path: str | Path = SUMMARY_PATH,
    byteclose_receipt_path: str | Path = BYTECLOSE_RECEIPT_PATH,
) -> CanonicalEquation:
    """Build the ET4 split law from the real receipt paths."""

    summary = _read_json(summary_path)
    receipt = _read_json(byteclose_receipt_path)
    aggregate = summary["aggregate"]
    patch = receipt["patch"]
    split = et4_solver_carriage_split_from_receipts(
        summary_path=summary_path,
        byteclose_receipt_path=byteclose_receipt_path,
    )
    provenance = build_provenance_for_research_sidecar(
        SOURCE_ARTIFACT,
        reactivation_criteria=(
            "append an anchor when the same solve-within/CVP family is recarried by a "
            "different description grammar or re-evaluated on a new base"
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="apple_cpu",
        captured_at_utc="2026-08-06T22:40:10Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="et4_twelfth_move_byteclosed_solver_reach_carriage_fail_20260806",
        measurement_utc="2026-08-06T22:40:10Z",
        inputs={
            "summary_path": str(summary_path),
            "byteclose_receipt_path": str(byteclose_receipt_path),
            "net_flip_reduction": aggregate["net_flip_reduction"],
            "patch_compressed_bytes": patch["compressed_bytes"],
            "baseline_archive_bytes": aggregate["baseline"]["archive_bytes"],
            "et4_archive_bytes": receipt["archive"]["archive_bytes"],
        },
        predicted_output={
            "w_break_even_bytes_per_flip": W_BREAK_EVEN_BYTES_PER_FLIP,
            "break_even_bytes": split["break_even_bytes"],
            "solver_reduces_d_seg": True,
        },
        empirical_output={
            "d_seg_before": aggregate["baseline"]["d_seg"],
            "d_seg_after": aggregate["d_seg_after_completed_scope"],
            "seg_delta_s": split["seg_delta_s"],
            "patch_b_per_flip": split["patch_b_per_flip"],
            "patch_over_break_even_ratio": split["patch_over_break_even_ratio"],
            "nnz_per_flip": split["nnz_per_flip"],
            "pointer_moved": False,
        },
        residual=0.0,
        source_artifact=str(summary_path),
        measurement_method=(
            "byte-closed ET4 archive receipt plus n600 macOS-CPU advisory evaluate; "
            "W closure recomputed from aggregate receipt net flips and scored bytes"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="ET4 twelfth-move solver reach versus carriage split",
        one_line_summary=(
            "Within-CVP solves reduced realized d_seg on tq1c, but the sparse i16 "
            "patch carriage spent far above W break-even, so the row is rate-dead."
        ),
        latex_form=(
            r"B_{break}=F\,W,\quad W={25/37545489\over 100/(600\cdot384\cdot512)}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_et4_twelfth_move_solver_carriage_20260807:"
            "et4_solver_carriage_split"
        ),
        domain_of_validity={
            "included": [
                "ET4 solve-within/CVP sparse frame_1 i16 delta carriage on tq1c base",
                "byte-closed advisory n600 archive receipts with aggregate net flips",
            ],
            "excluded": [
                "contest-CPU/CUDA promotion",
                "new carriage grammars before they have their own measured bytes",
                "claiming solver-family failure from carriage failure",
            ],
            "verdict_scope": "INSTANCE: ET4 correction field on tq1c parent",
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={
            "net_flip_reduction": "flips",
            "patch_compressed_bytes": "bytes",
            "patch_nnz": "nonzero i16 deltas",
            "baseline_d_seg": "fraction",
            "realized_d_seg": "fraction",
            "archive_delta_bytes": "bytes",
        },
        units_out={
            "seg_delta_s": "contest S units",
            "break_even_bytes": "bytes",
            "patch_over_break_even_ratio": "unitless",
            "nnz_per_flip": "nonzero deltas per net flip",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"et4_w_closure_receipt_residual": 0.0},
        last_calibration_utc="2026-08-06T22:40:10Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "campaign_984_composition_route",
            "et5_restricted_carriage_pricing",
            "costate_sense_solver_reach_accounting",
        ),
        canonical_producers=(
            ".omx/research/ddm_et4_20260806/TWELFTH_MOVE_ADJUDICATION.md",
            ".omx/research/ddm_et4_20260806/et4_solve_within_cvp_summary.json",
            ".omx/research/ddm_et4_20260806/byteclose_archive_receipt.json",
        ),
        provenance=provenance,
    )


def populate_ddm_et4_twelfth_move_solver_carriage_split_v1(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_ddm_et4_twelfth_move_solver_carriage_split_v1()
    register_canonical_equation(
        eq,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="ddm_cq1 registration: ET4 twelfth-move solver reach versus carriage split",
    )
    return eq


__all__ = [
    "BYTECLOSE_RECEIPT_PATH",
    "EQUATION_ID",
    "SOURCE_ARTIFACT",
    "SUMMARY_PATH",
    "W_BREAK_EVEN_BYTES_PER_FLIP",
    "build_ddm_et4_twelfth_move_solver_carriage_split_v1",
    "et4_solver_carriage_split",
    "et4_solver_carriage_split_from_receipts",
    "populate_ddm_et4_twelfth_move_solver_carriage_split_v1",
]
