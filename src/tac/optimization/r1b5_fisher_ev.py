# SPDX-License-Identifier: MIT
"""Intrinsic-size Fisher ordering and exact resize-support partition for R1b5.

The 38,077 ordering population is the measured PDW1 n24 realization mismatch
set.  It is deliberately distinct from the later R2b n600 17,926-flip set;
only the 16,319 moderate-margin R2b rows are used for the coupling-component
audit.  No function in this module evaluates or claims a contest score.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

CAMERA_HW: Final = (874, 1164)
SCORER_HW: Final = (384, 512)
EXPECTED_PDW1_CANDIDATES: Final = 38_077
EXPECTED_MODERATE_R2B: Final = 16_319


class R1B5FisherEVError(ValueError):
    """Malformed custody or a population inconsistent with the sealed row."""


@dataclass(frozen=True, slots=True)
class AxisSupport:
    indices: tuple[int, ...]


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.weight = [1] * size

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a == b:
            return
        if self.weight[a] < self.weight[b]:
            a, b = b, a
        self.parent[b] = a
        self.weight[a] += self.weight[b]


def exact_half_pixel_axis_supports(input_size: int, output_size: int) -> tuple[AxisSupport, ...]:
    """Return exact nonzero bilinear taps for ``align_corners=False``."""

    if input_size <= 0 or output_size <= 0:
        raise R1B5FisherEVError("resize dimensions must be positive")
    denominator = 2 * output_size
    rows: list[AxisSupport] = []
    for output_index in range(output_size):
        numerator = (2 * output_index + 1) * input_size - output_size
        left = numerator // denominator
        fraction = numerator - left * denominator
        taps: set[int] = set()
        if denominator - fraction:
            taps.add(min(max(left, 0), input_size - 1))
        if fraction:
            taps.add(min(max(left + 1, 0), input_size - 1))
        if not taps or len(taps) > 2:
            raise R1B5FisherEVError("invalid half-pixel support")
        rows.append(AxisSupport(tuple(sorted(taps))))
    return tuple(rows)


def support_overlap_component_histogram(
    cells: Sequence[tuple[int, int, int]],
    *,
    camera_hw: tuple[int, int] = CAMERA_HW,
    scorer_hw: tuple[int, int] = SCORER_HW,
) -> dict[str, object]:
    """Partition cells whose exact camera-space bilinear supports intersect.

    Pair ids are part of the key: different video pairs are independent under
    the local resize operator.  The implementation is near-linear in the
    intrinsic cell count and never materializes an ambient camera mask.
    """

    row_supports = exact_half_pixel_axis_supports(camera_hw[0], scorer_hw[0])
    col_supports = exact_half_pixel_axis_supports(camera_hw[1], scorer_hw[1])
    union_find = _UnionFind(len(cells))
    owners: dict[tuple[int, int, int], int] = {}
    unique_cells: set[tuple[int, int, int]] = set()
    support_cardinality: Counter[int] = Counter()
    for index, (pair, row, col) in enumerate(cells):
        if not (0 <= row < scorer_hw[0] and 0 <= col < scorer_hw[1] and pair >= 0):
            raise R1B5FisherEVError("coupling cell lies outside scorer geometry")
        cell = (int(pair), int(row), int(col))
        if cell in unique_cells:
            raise R1B5FisherEVError(f"duplicate coupling cell {cell}")
        unique_cells.add(cell)
        taps = [
            (cell[0], camera_row, camera_col)
            for camera_row in row_supports[cell[1]].indices
            for camera_col in col_supports[cell[2]].indices
        ]
        support_cardinality[len(taps)] += 1
        for tap in taps:
            previous = owners.get(tap)
            if previous is None:
                owners[tap] = index
            else:
                union_find.union(previous, index)
    component_sizes: Counter[int] = Counter()
    roots: Counter[int] = Counter(union_find.find(index) for index in range(len(cells)))
    component_sizes.update(roots.values())
    largest = max(roots.values(), default=0)
    return {
        "cell_count": len(cells),
        "component_count": len(roots),
        "component_size_histogram": {str(size): count for size, count in sorted(component_sizes.items())},
        "largest_component_size": largest,
        "non_singleton_component_count": sum(count for size, count in component_sizes.items() if size > 1),
        "camera_support_cardinality_histogram": {
            str(size): count for size, count in sorted(support_cardinality.items())
        },
        "local_operator": "exact_bilinear_align_corners_false_support_overlap",
        "backbone_coupling": "OUTSIDE_LOCAL_PARTITION_REQUIRES_REALIZED_SECANT_CUSTODY",
    }


def fisher_trace_from_margin(margin: float) -> float:
    """Categorical top-two Fisher trace ``0.5 sech^2(m/2)``."""

    value = float(margin)
    if not math.isfinite(value) or value < 0.0:
        raise R1B5FisherEVError("Fisher margin must be finite and nonnegative")
    return 0.5 / math.cosh(0.5 * value) ** 2


def edge_stratum(labels: np.ndarray, pair: int, row: int, col: int) -> int:
    """Return 0=Road-Lane edge, 1=other edge, 2=non-edge."""

    center = int(labels[pair, row, col])
    neighbours: list[int] = []
    if row:
        neighbours.append(int(labels[pair, row - 1, col]))
    if row + 1 < labels.shape[1]:
        neighbours.append(int(labels[pair, row + 1, col]))
    if col:
        neighbours.append(int(labels[pair, row, col - 1]))
    if col + 1 < labels.shape[2]:
        neighbours.append(int(labels[pair, row, col + 1]))
    if center in (0, 1) and any({center, value} == {0, 1} for value in neighbours):
        return 0
    return 1 if any(value != center for value in neighbours) else 2


def rank_pdw1_candidates(
    *,
    labels: np.ndarray,
    hard_prediction: np.ndarray,
    sidecars: Mapping[int, Mapping[str, np.ndarray]],
    head_pair_norm_table: np.ndarray,
    enforce_counts: bool = True,
) -> tuple[list[list[object]], dict[str, object]]:
    """Rank every PDW1 mismatch in the registered Fisher/head-normal chart.

    Rows are compact arrays with a separately declared column schema.  The
    ordering is a deterministic lexicographic KKT precursor: necessity tier,
    exact local support cardinality, hyperplane flip distance, descending
    Fisher curvature, descending public head-normal gain, stable index.
    Marginal realized score-per-byte remains a later admission gate; it is not
    fabricated here.
    """

    target = np.asarray(labels)
    pred = np.asarray(hard_prediction)
    if target.shape != pred.shape or target.ndim != 3:
        raise R1B5FisherEVError("PDW1 labels/prediction geometry mismatch")
    if target.shape[1:] != SCORER_HW:
        raise R1B5FisherEVError("PDW1 scorer geometry is not 384x512")
    norms = np.asarray(head_pair_norm_table, dtype=np.float64)
    if norms.shape != (5, 5) or not np.isfinite(norms).all():
        raise R1B5FisherEVError("head-pair norm table must be finite 5x5")
    row_supports = exact_half_pixel_axis_supports(CAMERA_HW[0], SCORER_HW[0])
    col_supports = exact_half_pixel_axis_supports(CAMERA_HW[1], SCORER_HW[1])
    by_pair: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pair, row, col in zip(*np.nonzero(target != pred), strict=True):
        by_pair[int(pair)].append((int(row), int(col)))
    candidate_count = sum(map(len, by_pair.values()))
    if enforce_counts and candidate_count != EXPECTED_PDW1_CANDIDATES:
        raise R1B5FisherEVError(f"PDW1 candidate count {candidate_count} != {EXPECTED_PDW1_CANDIDATES}")
    records: list[tuple[tuple[object, ...], list[object]]] = []
    arrangement_match = 0
    edge_counts: Counter[int] = Counter()
    for pair, coordinates in sorted(by_pair.items()):
        custody = sidecars.get(pair)
        if custody is None:
            raise R1B5FisherEVError(f"VJP sidecar absent for PDW1 pair {pair}")
        required = {"winner", "rival", "cached_margin", "seg_q", "seg_local_lipschitz"}
        if not required.issubset(custody):
            raise R1B5FisherEVError(f"VJP sidecar fields absent for pair {pair}")
        winner = np.asarray(custody["winner"])
        rival = np.asarray(custody["rival"])
        margins = np.asarray(custody["cached_margin"], dtype=np.float64)
        q = np.asarray(custody["seg_q"], dtype=np.float64)
        lipschitz = np.asarray(custody["seg_local_lipschitz"], dtype=np.float64)
        expected_shapes = {
            "winner": SCORER_HW,
            "rival": SCORER_HW,
            "cached_margin": SCORER_HW,
            "seg_q": (*SCORER_HW, 3),
            "seg_local_lipschitz": SCORER_HW,
        }
        actual_shapes = {
            "winner": winner.shape,
            "rival": rival.shape,
            "cached_margin": margins.shape,
            "seg_q": q.shape,
            "seg_local_lipschitz": lipschitz.shape,
        }
        if actual_shapes != expected_shapes:
            raise R1B5FisherEVError(f"VJP sidecar geometry mismatch for pair {pair}")
        if not np.isfinite(margins).all() or not np.isfinite(q).all() or not np.isfinite(lipschitz).all():
            raise R1B5FisherEVError(f"VJP sidecar has nonfinite values for pair {pair}")
        for row, col in coordinates:
            desired = int(target[pair, row, col])
            observed = int(pred[pair, row, col])
            margin = float(margins[row, col])
            head_norm = float(norms[desired, observed])
            if head_norm <= 0.0:
                raise R1B5FisherEVError("mismatch candidate has zero class-pair norm")
            flip_distance = margin / head_norm
            fisher = fisher_trace_from_margin(margin)
            stratum = edge_stratum(target, pair, row, col)
            edge_counts[stratum] += 1
            support_taps = len(row_supports[row].indices) * len(col_supports[col].indices)
            matched = bool(winner[row, col] == desired and rival[row, col] == observed)
            arrangement_match += int(matched)
            linear = (pair * SCORER_HW[0] + row) * SCORER_HW[1] + col
            record: list[object] = [
                pair,
                row,
                col,
                linear,
                desired,
                observed,
                stratum,
                support_taps,
                margin,
                fisher,
                head_norm,
                flip_distance,
                matched,
                float(lipschitz[row, col]),
                [float(value) for value in q[row, col]],
            ]
            key: tuple[object, ...] = (
                stratum,
                support_taps,
                flip_distance,
                -fisher,
                -head_norm,
                linear,
            )
            records.append((key, record))
    records.sort(key=lambda item: item[0])
    rows = [record for _key, record in records]
    return rows, {
        "candidate_count": candidate_count,
        "edge_stratum_counts": {
            "road_lane_edge": edge_counts[0],
            "other_edge": edge_counts[1],
            "nonedge": edge_counts[2],
        },
        "vjp_arrangement_match_count": arrangement_match,
        "vjp_arrangement_mismatch_count": candidate_count - arrangement_match,
        "rank_columns": [
            "pair",
            "row",
            "col",
            "linear_index",
            "target_class",
            "realized_class",
            "necessity_edge_tier",
            "resize_support_taps",
            "top1_top2_margin",
            "fisher_trace",
            "target_realized_head_pair_norm",
            "flip_distance",
            "vjp_native_arrangement_match",
            "vjp_local_lipschitz",
            "vjp_unit_pullback_rgb",
        ],
        "metric": "fisher_top1_top2_margin",
        "policy": "measured_reverse_waterfill_highest_ev_first",
        "ordering_key": [
            "necessity_edge_tier_ascending",
            "resize_support_taps_ascending",
            "flip_distance_ascending",
            "fisher_trace_descending",
            "target_realized_head_pair_norm_descending",
            "linear_index_ascending",
        ],
        "marginal_rate_admission": "BLOCKED_PENDING_REALIZED_SECANT_AND_EXACT_PREFIX_BYTES",
    }


def head_pair_norm_table(weight: np.ndarray) -> np.ndarray:
    """Compute public frozen-head hyperplane norms without a scorer forward."""

    values = np.asarray(weight, dtype=np.float64)
    if values.shape[0] != 5 or values.ndim < 2 or not np.isfinite(values).all():
        raise R1B5FisherEVError("frozen SegNet head must have five finite rows")
    flat = values.reshape(5, -1)
    return np.linalg.norm(flat[:, None] - flat[None, :], axis=-1)


__all__ = [
    "EXPECTED_MODERATE_R2B",
    "EXPECTED_PDW1_CANDIDATES",
    "R1B5FisherEVError",
    "edge_stratum",
    "exact_half_pixel_axis_supports",
    "fisher_trace_from_margin",
    "head_pair_norm_table",
    "rank_pdw1_candidates",
    "support_overlap_component_histogram",
]
