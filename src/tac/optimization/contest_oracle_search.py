# SPDX-License-Identifier: MIT
"""Planning primitives for contest scorer-oracle search.

This module turns a contest atom lattice into deterministic candidate queue
rows.  It is deliberately conservative:

* no scorer is loaded;
* no archive is mutated;
* no score or dispatch readiness is claimed;
* every row carries the byte-budget model that produced it.

The queue is meant to seed a later CPU scorer-oracle loop or genetic/beam
search.  That loop can materialize rows into archives, call the deterministic
CPU evaluator, cache exact raw/archive hashes, then update the lattice with
empirical deltas.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SCHEMA = "contest_oracle_search_plan_v1"


class ContestOracleSearchError(ValueError):
    """Raised when a lattice cannot become an oracle-search plan."""


def build_lfv1_pair_queue(
    lattice: Mapping[str, Any],
    *,
    max_archive_delta_bytes: int,
    max_candidates: int,
    alpha_grid: Sequence[float],
    top_pair_pool: int = 32,
    min_pairs: int = 1,
    max_pairs: int = 8,
    lfv1_row_bytes: int = 13,
    lfv1_header_bytes: int = 12,
    zip_member_overhead_bytes: int = 132,
    radius_scale_grid: Sequence[float] = (0.7,),
    power_grid: Sequence[float] = (1.3,),
    origin_y_frac_grid: Sequence[float] = (0.45,),
) -> dict[str, Any]:
    """Build deterministic LFV1 sparse-pair candidate rows from a lattice."""

    if max_archive_delta_bytes <= 0:
        raise ContestOracleSearchError("max_archive_delta_bytes must be positive")
    if max_candidates <= 0:
        raise ContestOracleSearchError("max_candidates must be positive")
    if min_pairs <= 0 or max_pairs < min_pairs:
        raise ContestOracleSearchError("invalid min/max pair bounds")
    if not alpha_grid:
        raise ContestOracleSearchError("alpha_grid must be non-empty")

    pair_rows = _candidate_pair_pool(lattice, top_pair_pool=top_pair_pool)
    buckets: list[list[dict[str, Any]]] = []
    pair_counts = list(range(min_pairs, max_pairs + 1))
    pair_counts.sort(key=lambda value: (value, abs(value - 4)))
    param_grid = _stratified_param_grid(
        alpha_grid=alpha_grid,
        radius_scale_grid=radius_scale_grid,
        power_grid=power_grid,
        origin_y_frac_grid=origin_y_frac_grid,
    )
    for grid_index, (alpha, radius_scale, power, origin_y_frac) in enumerate(param_grid):
        bucket: list[dict[str, Any]] = []
        for pair_count in pair_counts:
            archive_delta = (
                int(zip_member_overhead_bytes)
                + int(lfv1_header_bytes)
                + int(lfv1_row_bytes) * int(pair_count)
            )
            if archive_delta > max_archive_delta_bytes:
                continue
            selected_pairs = [int(row["pair_index"]) for row in pair_rows[:pair_count]]
            if len(selected_pairs) != pair_count:
                continue
            support = pair_rows[:pair_count]
            typed_support = _sum_typed_masses(support)
            candidate = {
                "candidate_id": _candidate_id(
                    selected_pairs=selected_pairs,
                    alpha=float(alpha),
                    radius_scale=float(radius_scale),
                    power=float(power),
                    origin_y_frac=float(origin_y_frac),
                ),
                "family": "lfv1_sparse_pair_micro_foveation",
                "candidate_builder": "tools/build_hfv1_sidecar_candidate.py",
                "sidecar_format": "lfv1",
                "lfv1_version": 2,
                "selected_pairs": selected_pairs,
                "selected_frames": sorted(
                    frame for pair in selected_pairs for frame in (2 * pair, 2 * pair + 1)
                ),
                "params": {
                    "alpha": float(alpha),
                    "radius_scale": float(radius_scale),
                    "power": float(power),
                    "origin_y_frac": float(origin_y_frac),
                },
                "grid": {
                    "grid_index": int(grid_index),
                    "alpha": float(alpha),
                    "radius_scale": float(radius_scale),
                    "power": float(power),
                    "origin_y_frac": float(origin_y_frac),
                },
                "archive_delta_budget": {
                    "estimated_archive_delta_bytes": archive_delta,
                    "zip_member_overhead_bytes": int(zip_member_overhead_bytes),
                    "lfv1_header_bytes": int(lfv1_header_bytes),
                    "lfv1_row_bytes": int(lfv1_row_bytes),
                    "pair_count": int(pair_count),
                    "max_archive_delta_bytes": int(max_archive_delta_bytes),
                },
                "support": support,
                "support_venn_signatures": sorted(
                    {str(row.get("venn_signature", "")) for row in support}
                ),
                "support_score_mass_sum": sum(
                    float(row.get("score_mass_sum", 0.0)) for row in support
                ),
                "support_typed_score_masses": typed_support,
                "support_mixed_unit_score_mass_sum_deprecated_for_ranking": any(
                    _mixed_unit_deprecated(row) for row in support
                ),
                "support_max_waterfill_priority": max(
                    (float(row.get("max_waterfill_priority", 0.0)) for row in support),
                    default=0.0,
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": [
                    "oracle_search_plan_not_materialized",
                    "requires_cpu_scorer_oracle_measurement",
                    "requires_exact_cuda_auth_eval_before_promotion",
                ],
            }
            bucket.append(candidate)
        if bucket:
            buckets.append(bucket)
    rows = _balanced_truncate(buckets, max_candidates=max_candidates)
    return _plan(lattice, rows, param_grid=param_grid)


def _balanced_truncate(
    buckets: Sequence[Sequence[dict[str, Any]]],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    depth = 0
    while len(rows) < max_candidates:
        added = False
        for bucket in buckets:
            if depth >= len(bucket):
                continue
            rows.append(dict(bucket[depth]))
            added = True
            if len(rows) >= max_candidates:
                return rows
        if not added:
            return rows
        depth += 1
    return rows


def _stratified_param_grid(
    *,
    alpha_grid: Sequence[float],
    radius_scale_grid: Sequence[float],
    power_grid: Sequence[float],
    origin_y_frac_grid: Sequence[float],
) -> list[tuple[float, float, float, float]]:
    indexed = [
        ((ai, ri, pi, oi), (float(alpha), float(radius), float(power), float(origin_y)))
        for (ai, alpha), (ri, radius), (pi, power), (oi, origin_y) in itertools.product(
            enumerate(alpha_grid),
            enumerate(radius_scale_grid),
            enumerate(power_grid),
            enumerate(origin_y_frac_grid),
        )
    ]
    remaining = list(indexed)
    covered: list[set[int]] = [set(), set(), set(), set()]
    out: list[tuple[float, float, float, float]] = []
    while remaining:
        best_index = min(
            range(len(remaining)),
            key=lambda idx: (
                -sum(
                    1
                    for axis, value_index in enumerate(remaining[idx][0])
                    if value_index not in covered[axis]
                ),
                remaining[idx][0],
            ),
        )
        index_tuple, value_tuple = remaining.pop(best_index)
        for axis, value_index in enumerate(index_tuple):
            covered[axis].add(value_index)
        out.append(value_tuple)
    return out


def _candidate_pair_pool(
    lattice: Mapping[str, Any],
    *,
    top_pair_pool: int,
) -> list[dict[str, Any]]:
    pair_overlap = lattice.get("pair_signal_overlap")
    if not isinstance(pair_overlap, Mapping):
        raise ContestOracleSearchError("lattice missing pair_signal_overlap")
    top_pairs = pair_overlap.get("top_pairs")
    if not isinstance(top_pairs, list):
        raise ContestOracleSearchError("lattice pair_signal_overlap missing top_pairs")
    rows = [row for row in top_pairs if isinstance(row, Mapping)]
    rows.sort(
        key=lambda row: (
            _overlap_weight(str(row.get("venn_signature", ""))),
            _typed_mass(row, "component_score_mass_sum", fallback="score_mass_sum"),
            _typed_mass(row, "seg_score_mass_sum"),
            _typed_mass(row, "pose_score_mass_sum"),
            float(row.get("max_waterfill_priority", 0.0)),
            -int(row.get("pair_index", 10**9)),
        ),
        reverse=True,
    )
    return [dict(row) for row in rows[: max(1, int(top_pair_pool))]]


def _overlap_weight(signature: str) -> int:
    parts = set(signature.split("&")) if signature else set()
    weight = len(parts)
    if "pair_component" in parts and "xray_pair" in parts:
        weight += 4
    if "sidecar_selected" in parts:
        weight += 3
    if "xray_pixel" in parts:
        weight += 1
    return weight


def _typed_mass(row: Mapping[str, Any], key: str, *, fallback: str | None = None) -> float:
    typed = row.get("typed_score_masses")
    value = typed.get(key) if isinstance(typed, Mapping) else None
    if value is None and fallback is not None:
        value = row.get(fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _mixed_unit_deprecated(row: Mapping[str, Any]) -> bool:
    typed = row.get("typed_score_masses")
    return bool(
        isinstance(typed, Mapping)
        and typed.get("mixed_unit_score_mass_sum_deprecated_for_ranking")
    )


def _sum_typed_masses(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = (
        "pose_score_mass_sum",
        "seg_score_mass_sum",
        "rate_score_mass_sum",
        "pixel_proxy_mass_sum",
        "gradient_proxy_mass_sum",
        "component_score_mass_sum",
        "mixed_unit_score_mass_sum",
    )
    return {key: sum(_typed_mass(row, key) for row in rows) for key in keys}


def _candidate_id(
    *,
    selected_pairs: Sequence[int],
    alpha: float,
    radius_scale: float,
    power: float,
    origin_y_frac: float,
) -> str:
    digest = hashlib.sha256(
        repr(
            (
                tuple(int(pair) for pair in selected_pairs),
                round(float(alpha), 10),
                round(float(radius_scale), 6),
                round(float(power), 6),
                round(float(origin_y_frac), 6),
            )
        ).encode("utf-8")
    ).hexdigest()[:12]
    return (
        f"lfv1v2_k{len(selected_pairs):02d}_"
        f"a{_float_token(alpha)}_r{_float_token(radius_scale)}_"
        f"p{_float_token(power)}_oy{_float_token(origin_y_frac)}_{digest}"
    )


def _float_token(value: float) -> str:
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _plan(
    lattice: Mapping[str, Any],
    rows: list[dict[str, Any]],
    *,
    param_grid: Sequence[tuple[float, float, float, float]] | None = None,
) -> dict[str, Any]:
    covered = {
        (
            float(row.get("params", {}).get("alpha")),
            float(row.get("params", {}).get("radius_scale")),
            float(row.get("params", {}).get("power")),
            float(row.get("params", {}).get("origin_y_frac")),
        )
        for row in rows
        if isinstance(row.get("params"), Mapping)
    }
    declared = {
        (float(alpha), float(radius), float(power), float(origin_y))
        for alpha, radius, power, origin_y in (param_grid or [])
    }
    return {
        "schema": SCHEMA,
        "source_lattice_schema": lattice.get("schema"),
        "source_lattice_atom_count": lattice.get("atom_count"),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_count": len(rows),
        "grid_coverage": {
            "declared_param_combo_count": len(declared),
            "covered_param_combo_count": len(covered),
            "omitted_param_combos_after_truncation": [
                {
                    "alpha": combo[0],
                    "radius_scale": combo[1],
                    "power": combo[2],
                    "origin_y_frac": combo[3],
                }
                for combo in sorted(declared - covered)
            ],
            "balanced_truncation": True,
        },
        "candidates": rows,
        "next_loop_contract": [
            "materialize candidate archive",
            "run official inflate locality/raw hash control",
            "run cached CPU raw advisory evaluator",
            "append measured deltas back into lattice/probe ledger",
            "dispatch exact CUDA only for positive byte-closed candidates",
        ],
    }


__all__ = [
    "ContestOracleSearchError",
    "SCHEMA",
    "build_lfv1_pair_queue",
]
