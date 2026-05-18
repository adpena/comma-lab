# SPDX-License-Identifier: MIT
"""Per-byte strategy for the per-X optimal codec planner.

[verified-against: .omx/research/empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md Section 5.3]
[verified-against: .omx/research/master_gradient_xray_fields_medal_research_wave_20260518.md Section 4.1 sensitivity_mask_aware_quantizr_v1]
[verified-against: tac.master_gradient.compute_marginal_coefficients]
[verified-against: Catalog #265 canonical-contract tokens]
[verified-against: Catalog #287 evidence-tag discipline]
[verified-against: Catalog #323 canonical Provenance contract]

The per-byte strategy is the CANONICAL FIRST INSTANCE of the per-X planner. Given
(archive_sha256, codec_menu, byte_budget, sensitivity_threshold_quantiles) it emits
a per-byte codec assignment matching the Fields-Medal sensitivity_mask_aware_quantizr_v1
design.

Algorithm:
1. Load master_gradient anchor for the archive (canonical fcntl-locked JSONL)
2. Load (N, 3) per-byte gradient .npy sidecar
3. Compute per-byte L1 sensitivity: |grad_seg| + |grad_pose| + |grad_rate|
4. Argsort descending → cumulative quantile assignment to codec buckets
5. For each bucket, compute (n_bytes, codec_bits, total_bytes_after_codec)
6. Compute predicted ΔS = rate_savings_delta + quantization_distortion_delta
7. Emit typed PerXCodecAssignmentPlan with canonical Provenance per Catalog #323

The CONTEST RATE FORMULA per upstream/evaluate.py:92:
    R = 25 * archive_bytes / 37_545_489
    ∂R/∂byte = 25 / 37_545_489 ≈ 6.66e-7

The CANONICAL OPERATING-POINT MARGINAL per tac.master_gradient.compute_marginal_coefficients:
    ∂S/∂d_seg = 100
    ∂S/∂d_pose = 5 / sqrt(10 * d_pose)
    ∂S/∂R = 25
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

from tac.empirical_per_x_optimal_codec_planner.codec_menu import (
    CODEC_NAMES,
    codec_bits_per_sample,
    codec_bytes_for_n_samples,
)
from tac.empirical_per_x_optimal_codec_planner.contract import (
    PerXAssignmentRow,
    PerXCodecAssignmentPlan,
    PlannerError,
)
from tac.master_gradient import is_authoritative_axis_anchor


CONTEST_RATE_DENOM_BYTES: int = 37_545_489
"""Per upstream/evaluate.py:92"""


def _compute_marginal_coefficients_from_op(op: dict) -> tuple[float, float, float]:
    """Local copy of tac.master_gradient.compute_marginal_coefficients."""
    d_pose = float(op.get("d_pose", 0.0))
    if d_pose <= 0:
        raise PlannerError(f"d_pose must be > 0; got {d_pose}")
    seg_marginal = 100.0
    pose_marginal = 5.0 / math.sqrt(10.0 * d_pose)
    rate_marginal_per_byte = 25.0 / CONTEST_RATE_DENOM_BYTES
    return seg_marginal, pose_marginal, rate_marginal_per_byte


def _build_predicted_provenance(
    *,
    archive_sha256: str,
    npy_path: Path,
    measurement_axis: str,
) -> dict:
    """Build canonical Provenance per Catalog #323 for a predicted plan."""
    return {
        "kind": "predicted_from_master_gradient",
        "source_artifact_path": str(npy_path),
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "archive_sha256": archive_sha256,
        "measurement_axis": measurement_axis,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "predicted",
        "rationale": (
            "Per-byte codec assignment plan derived from master_gradient sensitivity "
            "quantiles. Predicted ΔS bands are NOT score claims per Catalog #287; "
            "promotion requires paired empirical landing per Catalog #323."
        ),
        "axis_tag": "[predicted, empirical-grounded-from-master-gradient]",
    }


def _resolve_master_gradient_anchor(
    archive_sha256: str,
    repo_root: Path,
) -> tuple[dict, Path]:
    """Find the master gradient anchor + .npy path for an archive.

    Returns (anchor_dict, resolved_npy_path).

    Raises PlannerError if no anchor exists or .npy file missing.
    """
    jsonl_path = repo_root / ".omx/state/master_gradient_anchors.jsonl"
    if not jsonl_path.exists():
        raise PlannerError(
            f"no master_gradient_anchors.jsonl at {jsonl_path}; "
            "materialize one with `tools/extract_master_gradient.py "
            "--archive <archive.zip> --inflate-py <submission_dir/inflate.py> "
            "--upstream-dir upstream --axis '[macOS-CPU advisory]' "
            "--device cpu --output-npy <sidecar.npy>` for local advisory "
            "planning, or use a full-pair contest axis on authoritative "
            f"hardware. Requested archive_sha256={archive_sha256!r}."
        )

    matching_anchors = []
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and r.get("archive_sha256") == archive_sha256:
            if (
                r.get("gradient_tensor_kind", "aggregate_per_byte_v1")
                == "aggregate_per_byte_v1"
                and is_authoritative_axis_anchor(r)
            ):
                matching_anchors.append(r)

    if not matching_anchors:
        raise PlannerError(
            f"no master_gradient anchor for archive_sha256={archive_sha256!r}; "
            f"available anchors require tools/extract_master_gradient.py to materialize. "
            f"Per CPU-frontier campaign plan §1.1 PR101_lc_v2 + a1_baseline + PR106 format0d + "
            f"PR107 apogee are highest-EV ~$0 op-routables."
        )

    # Latest by measurement_utc
    anchor = max(matching_anchors, key=lambda r: r.get("measurement_utc", ""))
    npy_path_rel = anchor.get("gradient_array_path")
    if not npy_path_rel:
        raise PlannerError(f"anchor {archive_sha256!r} missing gradient_array_path")
    npy_path = repo_root / npy_path_rel
    if not npy_path.exists():
        raise PlannerError(
            f"gradient_array_path={npy_path_rel} does not exist at {npy_path}"
        )

    return anchor, npy_path


def plan_per_byte_from_master_gradient(
    *,
    archive_sha256: str,
    codec_menu: tuple[str, ...] = ("fp16", "int8", "int6", "int4"),
    byte_budget: int = 300_000,
    sensitivity_threshold_quantiles: tuple[float, ...] = (0.02, 0.05, 0.20, 1.00),
    repo_root: Path | str = ".",
) -> PerXCodecAssignmentPlan:
    """Emit per-byte codec assignment plan from master_gradient anchor.

    This is the CANONICAL FIRST INSTANCE of the per-X planner. Given the fec6
    archive sha256 + the canonical Quantizr codec menu + 300KB byte budget + the
    Fields-Medal quantile schedule, it emits the sensitivity_mask_aware_quantizr_v1
    design.

    Args:
        archive_sha256: SHA-256 hex of the target archive (must have master_gradient anchor)
        codec_menu: tuple of codec names IN ORDER OF DECREASING PRECISION
                    e.g. ("fp16", "int8", "int6", "int4")
        byte_budget: maximum allowed encoded bytes
        sensitivity_threshold_quantiles: cumulative quantile cutoffs from top
                    e.g. (0.02, 0.05, 0.20, 1.00) means top 2% → codec[0],
                    next 3% (0.05-0.02) → codec[1], next 15% (0.20-0.05) → codec[2],
                    remaining 80% (1.00-0.20) → codec[3]
                    MUST be len(codec_menu) and end with 1.0
        repo_root: repo root for resolving paths

    Returns:
        PerXCodecAssignmentPlan with x_granularity='byte', evidence_grade='predicted'
    """
    if np is None:  # pragma: no cover
        raise PlannerError("numpy required for per-byte sensitivity computation")

    repo = Path(repo_root)

    # Validate codec menu
    for codec in codec_menu:
        if codec not in CODEC_NAMES:
            raise PlannerError(f"codec {codec!r} not in canonical CODEC_NAMES")

    # Validate quantile schedule
    if len(sensitivity_threshold_quantiles) != len(codec_menu):
        raise PlannerError(
            f"len(sensitivity_threshold_quantiles)={len(sensitivity_threshold_quantiles)} "
            f"must equal len(codec_menu)={len(codec_menu)}"
        )
    if any(q < 0 or q > 1 for q in sensitivity_threshold_quantiles):
        raise PlannerError(
            f"sensitivity_threshold_quantiles must be in [0, 1]; "
            f"got {sensitivity_threshold_quantiles}"
        )
    if list(sensitivity_threshold_quantiles) != sorted(sensitivity_threshold_quantiles):
        raise PlannerError(
            f"sensitivity_threshold_quantiles must be monotonically increasing; "
            f"got {sensitivity_threshold_quantiles}"
        )
    if abs(sensitivity_threshold_quantiles[-1] - 1.0) > 1e-9:
        raise PlannerError(
            f"sensitivity_threshold_quantiles[-1] must equal 1.0 "
            f"(complete partition); got {sensitivity_threshold_quantiles[-1]}"
        )

    # Load anchor + .npy
    anchor, npy_path = _resolve_master_gradient_anchor(archive_sha256, repo)
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise PlannerError(
            f"gradient array {npy_path} shape {arr.shape} != (N, 3)"
        )
    n_bytes = arr.shape[0]
    declared_n = anchor.get("n_bytes")
    if declared_n is not None and int(declared_n) != n_bytes:
        raise PlannerError(
            f"anchor declared n_bytes={declared_n} but .npy shape={arr.shape}"
        )

    # Per-byte L1 sensitivity
    sens = np.abs(arr).sum(axis=1).astype(np.float32)

    # Argsort descending
    order = np.argsort(-sens)

    # Cumulative byte counts per codec class
    cumulative_n = [int(n_bytes * q) for q in sensitivity_threshold_quantiles]

    # Operating point + marginal coefficients
    op = anchor.get("operating_point", {})
    seg_marginal, pose_marginal, rate_per_byte = _compute_marginal_coefficients_from_op(op)

    # Map class names by codec menu position (canonical naming)
    canonical_class_names = (
        ("top_2pct", "top_5pct", "top_20pct", "tail") if len(codec_menu) == 4
        else tuple(f"class_{i}" for i in range(len(codec_menu)))
    )

    # Assign bytes to codecs
    assignments: list[PerXAssignmentRow] = []
    total_score_delta = 0.0
    total_bytes_after = 0
    prev_n = 0
    for codec, cum_n, class_name in zip(codec_menu, cumulative_n, canonical_class_names):
        n_in_class = cum_n - prev_n
        if n_in_class <= 0:
            prev_n = cum_n
            continue
        codec_bits = codec_bits_per_sample(codec)
        bytes_after_codec_total = codec_bytes_for_n_samples(codec, n_in_class)
        # Per-byte rate savings: 1 input byte - (codec_bits / 8) output byte
        # When variable-rate (None), assume 1 byte preserved
        if codec_bits is None:
            per_byte_after = 1.0
        else:
            per_byte_after = codec_bits / 8.0
        per_byte_rate_savings = 1.0 - per_byte_after  # positive = saves bytes
        per_byte_rate_savings_delta_score = -per_byte_rate_savings * 25.0 / CONTEST_RATE_DENOM_BYTES
        # Negative = score improvement (we want LOWER score)

        # Quantization distortion factor: empirically Quantizr 0.33 → ~95% score retention
        # For uniform-8-bit baseline: noise_factor = max(0, (8 - codec_bits) / 8)
        if codec_bits is None or codec_bits >= 8:
            quant_noise_factor = 0.0
        else:
            quant_noise_factor = (8 - codec_bits) / 8.0
        # Scale by canonical Quantizr empirical: 5% score regression at fp4
        # so per-byte quantization distortion ≈ sens[byte] * noise_factor * 0.1 (heuristic)
        # Per the cargo-cult audit Section 9 row 5 this is HARD-EARNED at the algorithm
        # level; CARGO-CULTED at the magnitude level pending empirical validation.

        byte_indices = order[prev_n:cum_n]
        for byte_idx in byte_indices.tolist():
            quant_distortion_per_byte = float(sens[byte_idx]) * quant_noise_factor * 0.1
            net_delta = per_byte_rate_savings_delta_score + quant_distortion_per_byte
            per_byte_after_bytes = (codec_bits + 7) // 8 if codec_bits else 1
            assignments.append(PerXAssignmentRow(
                x_index=int(byte_idx),
                x_class=class_name,
                sensitivity_score=float(sens[byte_idx]),
                chosen_codec=codec,
                chosen_codec_bits=codec_bits,
                predicted_score_delta=net_delta,
                predicted_bytes_after_codec=per_byte_after_bytes,
            ))
            total_score_delta += net_delta
        total_bytes_after += bytes_after_codec_total
        prev_n = cum_n

    plan = PerXCodecAssignmentPlan(
        archive_sha256=archive_sha256,
        x_granularity="byte",
        codec_menu=tuple(codec_menu),
        byte_budget=byte_budget,
        sensitivity_threshold_quantiles=tuple(sensitivity_threshold_quantiles),
        assignments=tuple(assignments),
        total_predicted_score_delta=total_score_delta,
        total_predicted_bytes=total_bytes_after,
        total_predicted_bytes_within_budget=(total_bytes_after <= byte_budget),
        operating_point=op,
        measurement_axis=anchor.get("measurement_axis", "[unknown]"),
        evidence_grade="predicted",
        provenance=_build_predicted_provenance(
            archive_sha256=archive_sha256,
            npy_path=npy_path,
            measurement_axis=anchor.get("measurement_axis", "[unknown]"),
        ),
        captured_at_utc=datetime.now(UTC).isoformat(),
    )
    return plan


# Canonical convenience alias
plan_per_byte_for_archive_via_sensitivity_quantiles = plan_per_byte_from_master_gradient


__all__ = [
    "CONTEST_RATE_DENOM_BYTES",
    "plan_per_byte_for_archive_via_sensitivity_quantiles",
    "plan_per_byte_from_master_gradient",
]
