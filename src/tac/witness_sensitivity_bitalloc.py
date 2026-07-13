# SPDX-License-Identifier: MIT
"""Pure #336 witness precision-response and reverse-waterfill helpers.

The measurement producer lives in ``tools/probe_witness_sensitivity_bitalloc.py``.
This module contains only deterministic arithmetic over its n600 measured rows.  It
reuses the existing empirical KKT solver rather than introducing another allocator.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

from tac.losses.variable_level_waterfill_allocator import (
    solve_waterfill_allocation,
    verify_kkt_marginal_equalization,
)

SCHEMA = "witness_section_precision_response_curves.v1"
RATE_DENOMINATOR = 37_545_489
BITS_TO_LEVEL = {8: 127, 7: 63, 6: 31, 5: 15, 4: 7, 3: 3, 2: 1}
LEVEL_TO_BITS = {v: k for k, v in BITS_TO_LEVEL.items()}


def distortion_score_delta(
    d_seg: float,
    d_pose: float,
    baseline_d_seg: float,
    baseline_d_pose: float,
) -> float:
    """Exact task-term delta, including the nonlinear PoseNet square root."""
    values = (d_seg, d_pose, baseline_d_seg, baseline_d_pose)
    if not all(math.isfinite(float(v)) and float(v) >= 0.0 for v in values):
        raise ValueError(f"distortions must be finite and non-negative, got {values!r}")
    return 100.0 * (float(d_seg) - float(baseline_d_seg)) + (
        math.sqrt(10.0 * float(d_pose)) - math.sqrt(10.0 * float(baseline_d_pose))
    )


def score_delta(
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
    baseline_d_seg: float,
    baseline_d_pose: float,
    baseline_archive_bytes: int,
) -> float:
    """Exact advisory score delta versus the byte-closed baseline."""
    return distortion_score_delta(d_seg, d_pose, baseline_d_seg, baseline_d_pose) + (
        25.0 * (int(archive_bytes) - int(baseline_archive_bytes)) / RATE_DENOMINATOR
    )


def repeat_noise_floor(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, float]:
    """Componentwise absolute deterministic-repeat floor."""
    return {
        "d_seg": abs(float(second["d_seg"]) - float(first["d_seg"])),
        "d_pose": abs(float(second["d_pose"]) - float(first["d_pose"])),
    }


def classify_response_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline: Mapping[str, Any],
    noise_floor: Mapping[str, float],
    psuff_task_tolerance: float = 0.005,
) -> list[dict[str, Any]]:
    """Add #153 zero-invariance, P-SUFF, and within-tensor Pareto labels.

    A precision rung is Pareto-dominated only by another row for the *same tensor*.
    Deletion/coarsening safety is stricter: both component deltas must be within the
    measured deterministic-repeat floor.  The historical #153 ``+0.005`` task-term
    tolerance is reported separately and is never substituted for the zero-invariance
    admission predicate.
    """
    source = [dict(r) for r in rows]
    for row in source:
        row["distortion_delta_S"] = distortion_score_delta(
            row["d_seg"], row["d_pose"], baseline["d_seg"], baseline["d_pose"]
        )
    by_tensor: dict[str, list[dict[str, Any]]] = {}
    for row in source:
        by_tensor.setdefault(str(row["tensor"]), []).append(row)
    out: list[dict[str, Any]] = []
    for tensor_rows in by_tensor.values():
        for row in tensor_rows:
            dseg_delta = float(row["d_seg"]) - float(baseline["d_seg"])
            dpose_delta = float(row["d_pose"]) - float(baseline["d_pose"])
            dist = distortion_score_delta(
                row["d_seg"], row["d_pose"], baseline["d_seg"], baseline["d_pose"]
            )
            dominated_by: list[str] = []
            for other in tensor_rows:
                if other is row:
                    continue
                no_more_bytes = int(other["archive_bytes"]) <= int(row["archive_bytes"])
                no_more_dist = float(other["distortion_delta_S"]) <= dist + 1e-15
                one_strict = (
                    int(other["archive_bytes"]) < int(row["archive_bytes"])
                    or float(other["distortion_delta_S"]) < dist - 1e-15
                )
                if no_more_bytes and no_more_dist and one_strict:
                    dominated_by.append(str(other["label"]))
            row.update(
                {
                    "delta_d_seg": dseg_delta,
                    "delta_d_pose": dpose_delta,
                    "distortion_delta_S": dist,
                    "delta_archive_bytes": int(row["archive_bytes"])
                    - int(baseline["archive_bytes"]),
                    "net_delta_S_advisory": score_delta(
                        row["d_seg"],
                        row["d_pose"],
                        row["archive_bytes"],
                        baseline["d_seg"],
                        baseline["d_pose"],
                        baseline["archive_bytes"],
                    ),
                    "zero_invariant_within_repeat_floor": bool(
                        dseg_delta <= float(noise_floor["d_seg"])
                        and dpose_delta <= float(noise_floor["d_pose"])
                    ),
                    "psuff_task_tolerance_pass": bool(dist <= float(psuff_task_tolerance)),
                    "pareto_dominated": bool(dominated_by),
                    "dominated_by": sorted(dominated_by),
                }
            )
            out.append(row)
    return sorted(out, key=lambda r: (str(r["tensor"]), int(r.get("bits", -1))), reverse=False)


def build_rd_table(
    classified_rows: Iterable[Mapping[str, Any]], baseline: Mapping[str, Any]
) -> dict[str, dict[int, tuple[float, float]]]:
    """Convert measured precision rows to the existing KKT solver's RD table."""
    table: dict[str, dict[int, tuple[float, float]]] = {}
    for row in classified_rows:
        bits = row.get("bits")
        if bits not in BITS_TO_LEVEL:
            continue
        level = BITS_TO_LEVEL[int(bits)]
        saving = float(int(baseline["archive_bytes"]) - int(row["archive_bytes"]))
        dist = float(row["distortion_delta_S"])
        table.setdefault(str(row["tensor"]), {})[level] = (saving, dist)
    for tensor, curve in table.items():
        curve[BITS_TO_LEVEL[8]] = (0.0, 0.0)
        missing = set(BITS_TO_LEVEL.values()) - set(curve)
        if missing:
            raise ValueError(f"{tensor}: incomplete measured precision curve; missing levels {missing}")
    if not table:
        raise ValueError("no precision rows available for KKT allocation")
    return table


def solve_measured_reverse_waterfill(
    classified_rows: Iterable[Mapping[str, Any]], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the existing empirical KKT allocator on full measured curves."""
    table = build_rd_table(classified_rows, baseline)
    grid = tuple(sorted(BITS_TO_LEVEL.values(), reverse=True))
    allocation = solve_waterfill_allocation(table, level_grid=grid, net_stop=True)
    holds, explanation = verify_kkt_marginal_equalization(allocation)
    return {
        "levels": dict(allocation.levels),
        "nbits": {name: LEVEL_TO_BITS[level] for name, level in allocation.levels.items()},
        "total_byte_saving_separable_prediction": allocation.total_byte_saving,
        "total_distortion_delta_S_separable_prediction": allocation.total_dist_cost,
        "net_delta_S_separable_prediction": allocation.net_score_delta,
        "n_coarsened": allocation.n_coarsened,
        "kkt_holds": holds,
        "kkt_explanation": explanation,
        "trace": [asdict(step) for step in allocation.trace],
        "verdict_scope": (
            "DERIVED allocation over measured single-tensor n600 curves; combined replay is "
            "required because brotli bytes and scorer effects are non-additive"
        ),
    }


__all__ = [
    "BITS_TO_LEVEL",
    "LEVEL_TO_BITS",
    "RATE_DENOMINATOR",
    "SCHEMA",
    "build_rd_table",
    "classify_response_rows",
    "distortion_score_delta",
    "repeat_noise_floor",
    "score_delta",
    "solve_measured_reverse_waterfill",
]
