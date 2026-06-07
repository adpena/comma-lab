# SPDX-License-Identifier: MIT
"""Path-coded support/action candidates for inverse-evaluate planning.

This module is deliberately small: it turns already-measured hard-region
surfaces into archive-executable support payloads and non-promotable
``ActionEffect`` rows.  It does not score an archive, invent SegNet wall
crossing, or bypass parse-back/inflate gates.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from tac.analysis.action_effect import ActionEffect
from tac.analysis.inverse_scorer_actions import build_candidate_queue

PATH_ACTION_PRODUCER_SCHEMA = "tac.path_action_producer.v1"
PATH_SUPPORT_SCHEMA = "tac.path_tube_support.v1"
PATH_ACTION_CANDIDATE_SCHEMA = "tac.path_action_candidate.v1"
PATH_SUPPORT_MAGIC = b"PTUB1"
_HEADER_FMT = "<5sHHBHHHII"
_POINT_FMT = "<HH"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

BLOCKER_PATH_SUPPORT_EMPTY = "path_action_support_empty"
BLOCKER_PATH_SUPPORT_NOT_BIRTH = "path_action_support_without_wrong_to_target_is_not_birth"
BLOCKER_PATH_ACTION_PARSEBACK_MISSING = "path_action_parseback_survival_missing"
BLOCKER_PATH_ACTION_INFLATE_MISSING = "path_action_inflate_survival_missing"


@dataclass(frozen=True)
class PathTubeSupport:
    """Exact support represented as path tube plus residual add/remove pixels."""

    height: int
    width: int
    frame_index: int
    target_class: int
    pair_index: int
    tube_width: int
    path_yx: tuple[tuple[int, int], ...]
    residual_add_yx: tuple[tuple[int, int], ...]
    residual_remove_yx: tuple[tuple[int, int], ...]

    @property
    def path_point_count(self) -> int:
        return len(self.path_yx)

    @property
    def residual_add_count(self) -> int:
        return len(self.residual_add_yx)

    @property
    def residual_remove_count(self) -> int:
        return len(self.residual_remove_yx)

    def base_tube_mask(self) -> np.ndarray:
        mask = np.zeros((int(self.height), int(self.width)), dtype=bool)
        if not self.path_yx:
            return mask
        radius = max(0, math.ceil((int(self.tube_width) - 1) / 2.0))
        points = list(self.path_yx)
        if len(points) == 1:
            _paint_point(mask, points[0], radius)
            return mask
        for start, end in pairwise(points):
            _paint_segment(mask, start, end, radius)
        return mask

    def decode_mask(self) -> np.ndarray:
        mask = self.base_tube_mask()
        for y, x in self.residual_add_yx:
            mask[int(y), int(x)] = True
        for y, x in self.residual_remove_yx:
            mask[int(y), int(x)] = False
        return mask

    def encode_payload(self) -> bytes:
        if not self.path_yx:
            raise ValueError("path_tube support requires at least one path point")
        chunks = [
            struct.pack(
                _HEADER_FMT,
                PATH_SUPPORT_MAGIC,
                int(self.height),
                int(self.width),
                int(self.frame_index),
                int(self.target_class),
                int(self.pair_index),
                int(self.tube_width),
                int(self.path_point_count),
                int(self.residual_add_count),
            ),
            struct.pack("<I", int(self.residual_remove_count)),
        ]
        for seq in (self.path_yx, self.residual_add_yx, self.residual_remove_yx):
            for y, x in seq:
                chunks.append(struct.pack(_POINT_FMT, int(y), int(x)))
        return zlib.compress(b"".join(chunks), level=9)

    def as_dict(self) -> dict[str, Any]:
        decoded = self.decode_mask()
        payload = self.encode_payload()
        return {
            "schema": PATH_SUPPORT_SCHEMA,
            "support_encoding": "path_tube",
            "height": int(self.height),
            "width": int(self.width),
            "frame_index": int(self.frame_index),
            "target_class": int(self.target_class),
            "pair_index": int(self.pair_index),
            "tube_width": int(self.tube_width),
            "path_point_count": int(self.path_point_count),
            "residual_add_count": int(self.residual_add_count),
            "residual_remove_count": int(self.residual_remove_count),
            "support_cardinality": int(np.count_nonzero(decoded)),
            "support_sha256": support_mask_sha256(decoded),
            "support_encoded_bytes": len(payload),
            "support_decode_sha256": hashlib.sha256(decoded.astype(np.uint8).tobytes(order="C")).hexdigest(),
            "archive_executable": True,
            "path_yx": [list(point) for point in self.path_yx],
            "residual_add_yx": [list(point) for point in self.residual_add_yx],
            "residual_remove_yx": [list(point) for point in self.residual_remove_yx],
        }


def decode_path_tube_payload(payload: bytes) -> PathTubeSupport:
    raw = zlib.decompress(payload)
    if len(raw) < _HEADER_SIZE + 4:
        raise ValueError("path_tube payload too short")
    magic, height, width, frame_index, target_class, pair_index, tube_width, point_count, add_count = struct.unpack(
        _HEADER_FMT,
        raw[:_HEADER_SIZE],
    )
    if magic != PATH_SUPPORT_MAGIC:
        raise ValueError(f"bad path_tube magic: {magic!r}")
    offset = _HEADER_SIZE
    (remove_count,) = struct.unpack("<I", raw[offset : offset + 4])
    offset += 4

    def read_points(count: int) -> tuple[tuple[int, int], ...]:
        nonlocal offset
        out: list[tuple[int, int]] = []
        for _ in range(int(count)):
            if offset + struct.calcsize(_POINT_FMT) > len(raw):
                raise ValueError("truncated path_tube point stream")
            y, x = struct.unpack(_POINT_FMT, raw[offset : offset + struct.calcsize(_POINT_FMT)])
            offset += struct.calcsize(_POINT_FMT)
            out.append((int(y), int(x)))
        return tuple(out)

    path = read_points(point_count)
    adds = read_points(add_count)
    removes = read_points(remove_count)
    if offset != len(raw):
        raise ValueError("path_tube payload has trailing bytes")
    return PathTubeSupport(
        height=int(height),
        width=int(width),
        frame_index=int(frame_index),
        target_class=int(target_class),
        pair_index=int(pair_index),
        tube_width=int(tube_width),
        path_yx=path,
        residual_add_yx=adds,
        residual_remove_yx=removes,
    )


def build_path_action_candidates_from_arrays(
    *,
    target_labels_bhw: np.ndarray,
    candidate_argmax_bhw: np.ndarray,
    target_margin_bhw: np.ndarray | None = None,
    pair_indices: np.ndarray | None = None,
    target_class: int | None = None,
    batch_index: int = 0,
    frame_index: int = 1,
    rdp_epsilon: float = 2.0,
    base_archive_bytes: int = 0,
    old_d_seg: float = 0.0,
    old_d_pose: float = 0.0,
    authority: str = "batch_local_path_support",
) -> dict[str, Any]:
    labels = np.asarray(target_labels_bhw)
    argmax = np.asarray(candidate_argmax_bhw)
    if labels.shape != argmax.shape or labels.ndim != 3:
        raise ValueError(
            "target_labels_bhw and candidate_argmax_bhw must share BHW shape; "
            f"got {labels.shape} and {argmax.shape}"
        )
    margins = None if target_margin_bhw is None else np.asarray(target_margin_bhw)
    if margins is not None and margins.shape != labels.shape:
        raise ValueError(f"target_margin_bhw must match labels; got {margins.shape} vs {labels.shape}")
    batch = int(batch_index)
    if batch < 0 or batch >= labels.shape[0]:
        raise ValueError(f"batch_index out of range: {batch}")
    pair_index = _pair_index(pair_indices, batch)
    klass = int(target_class) if target_class is not None else _worst_unsolved_class(labels[batch], argmax[batch])
    unsolved = (labels[batch] == klass) & (argmax[batch] != klass)
    component = largest_connected_component(unsolved)
    blockers: list[str] = []
    if not np.any(component):
        blockers.append(BLOCKER_PATH_SUPPORT_EMPTY)
        return {
            "schema": PATH_ACTION_PRODUCER_SCHEMA,
            "candidate_count": 0,
            "blockers": blockers,
            "action_effects": [],
            "candidate_queue": [],
            "comparison": {},
        }
    support = path_tube_support_from_mask(
        component,
        pair_index=pair_index,
        frame_index=frame_index,
        target_class=klass,
        epsilon=rdp_epsilon,
    )
    decoded = support.decode_mask()
    support_hash = support_mask_sha256(decoded)
    support_dict = support.as_dict()
    if support_hash != support_dict["support_sha256"]:
        raise AssertionError("path support hash mismatch after decode")
    cardinality = int(np.count_nonzero(decoded))
    margin_delta = _support_margin_stat(margins[batch] if margins is not None else None, decoded)
    payload_bytes = int(support_dict["support_encoded_bytes"])
    comparison = support_encoding_comparison(component, path_payload_bytes=payload_bytes)
    effect = ActionEffect.build(
        action_id=_path_action_id(pair_index=pair_index, target_class=klass, support_sha256=support_hash),
        family="hinerv",
        action_kind="frame1_seg_margin_frontier_path",
        inverse_source="path_tube_segnet_margin_frontier",
        frame_index=int(frame_index),
        frame_incidence="seg_pose_joint",
        candidate_status="rejected",
        authority=authority,
        normalization_scope="batch_local",
        producer="path_action_producer",
        consumer="inverse_evaluate_candidate_queue",
        pair_ids=[pair_index],
        class_ids=[klass],
        region_ids=[f"b{batch}/c{klass}/path0"],
        payload_sections=["path_tube_support"],
        old_d_seg=float(old_d_seg),
        new_d_seg=float(old_d_seg),
        old_d_pose=float(old_d_pose),
        new_d_pose=float(old_d_pose),
        old_bytes=int(base_archive_bytes),
        new_bytes=int(base_archive_bytes) + payload_bytes,
        receiver_surface={
            "uint8_changed_pixels": 0,
            "seg_argmax_changed_pixels": 0,
            "seg_wrong_to_target_count": 0,
            "seg_target_hard_lost_count": 0,
            "seg_wrong_to_wrong_count": 0,
        },
        exact_score_decision="reject",
        parseback_survived=False,
        inflate_survived=False,
        fakequant_survived=None,
        hard_won_count=0,
        wrong_to_target=0,
        target_to_wrong=0,
        wrong_to_wrong=0,
        net_target_support_delta=0,
        argmax_changed_count_region=0,
        uint8_changed_count_region=0,
        segnet_margin_delta=margin_delta,
        support_source="target_debt_component_path_tube",
        support_cardinality=cardinality,
        support_sha256=support_hash,
        support_encoding="path_tube",
        support_encoded_bytes=payload_bytes,
        support_research_only=False,
        blockers=[
            BLOCKER_PATH_SUPPORT_NOT_BIRTH,
            BLOCKER_PATH_ACTION_PARSEBACK_MISSING,
            BLOCKER_PATH_ACTION_INFLATE_MISSING,
        ],
    )
    queue = build_candidate_queue([effect])
    candidate = {
        "schema": PATH_ACTION_CANDIDATE_SCHEMA,
        "action_id": effect.action_id,
        "family": effect.family,
        "action_kind": effect.action_kind,
        "support": support_dict,
        "comparison": comparison,
        "blockers": list(effect.blockers),
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return {
        "schema": PATH_ACTION_PRODUCER_SCHEMA,
        "candidate_count": 1,
        "blockers": [],
        "action_effects": [effect],
        "candidate_queue": queue,
        "path_action_candidates": [candidate],
        "comparison": comparison,
        "policy": {
            "path_actions_do_not_clear_launch_gate": True,
            "support_decode_hash_required": True,
            "wrong_to_target_required_for_birth": True,
            "rate_term_included_in_action_effect": True,
        },
    }


def path_tube_support_from_mask(
    mask: np.ndarray,
    *,
    pair_index: int,
    frame_index: int,
    target_class: int,
    epsilon: float = 2.0,
) -> PathTubeSupport:
    support = np.asarray(mask, dtype=bool)
    if support.ndim != 2:
        raise ValueError(f"support mask must be HW; got {support.shape}")
    if not np.any(support):
        raise ValueError("support mask is empty")
    height, width = support.shape
    path, tube_width = _centerline_path(support, epsilon=epsilon)
    preliminary = PathTubeSupport(
        height=int(height),
        width=int(width),
        frame_index=int(frame_index),
        target_class=int(target_class),
        pair_index=int(pair_index),
        tube_width=int(tube_width),
        path_yx=tuple(path),
        residual_add_yx=(),
        residual_remove_yx=(),
    )
    base = preliminary.base_tube_mask()
    add = _coords(support & ~base)
    remove = _coords(base & ~support)
    return PathTubeSupport(
        height=int(height),
        width=int(width),
        frame_index=int(frame_index),
        target_class=int(target_class),
        pair_index=int(pair_index),
        tube_width=int(tube_width),
        path_yx=tuple(path),
        residual_add_yx=tuple(add),
        residual_remove_yx=tuple(remove),
    )


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    if src.ndim != 2:
        raise ValueError(f"mask must be HW; got {src.shape}")
    visited = np.zeros(src.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    ys, xs = np.nonzero(src)
    for start_y, start_x in zip(ys.tolist(), xs.tolist(), strict=False):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        comp: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= ny < src.shape[0]
                    and 0 <= nx < src.shape[1]
                    and src[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(comp) > len(best):
            best = comp
    out = np.zeros(src.shape, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def support_mask_sha256(mask: np.ndarray) -> str:
    src = np.asarray(mask, dtype=bool)
    coords = np.argwhere(src).astype("<u2", copy=False)
    h = hashlib.sha256()
    h.update(struct.pack("<HHI", int(src.shape[0]), int(src.shape[1]), int(coords.shape[0])))
    h.update(coords.tobytes(order="C"))
    return h.hexdigest()


def support_encoding_comparison(mask: np.ndarray, *, path_payload_bytes: int) -> dict[str, Any]:
    support = np.asarray(mask, dtype=bool)
    flat = support.reshape(-1)
    cardinality = int(np.count_nonzero(flat))
    bitmap_bytes = math.ceil(flat.size / 8.0)
    return {
        "schema": "tac.path_support_encoding_comparison.v1",
        "support_cardinality": cardinality,
        "bitmap_bytes": bitmap_bytes,
        "coordinate_list_bytes": int(cardinality * 4),
        "rle_bytes": _rle_bytes(flat),
        "path_tube_bytes": int(path_payload_bytes),
        "best_encoding": min(
            {
                "bitmap": bitmap_bytes,
                "coordinate_list": int(cardinality * 4),
                "rle": _rle_bytes(flat),
                "path_tube": int(path_payload_bytes),
            }.items(),
            key=lambda item: item[1],
        )[0],
    }


def _centerline_path(mask: np.ndarray, *, epsilon: float) -> tuple[list[tuple[int, int]], int]:
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    horizontal = (x1 - x0) >= (y1 - y0)
    points: list[tuple[int, int]] = []
    widths: list[int] = []
    if horizontal:
        for x in range(x0, x1 + 1):
            col_ys = ys[xs == x]
            if col_ys.size == 0:
                continue
            points.append((round(float(np.median(col_ys))), int(x)))
            widths.append(int(col_ys.max() - col_ys.min() + 1))
    else:
        for y in range(y0, y1 + 1):
            row_xs = xs[ys == y]
            if row_xs.size == 0:
                continue
            points.append((int(y), round(float(np.median(row_xs)))))
            widths.append(int(row_xs.max() - row_xs.min() + 1))
    if not points:
        points = [(int(ys[0]), int(xs[0]))]
        widths = [1]
    simplified = rdp_simplify(points, epsilon=epsilon)
    tube_width = max(1, round(float(np.median(widths))))
    return simplified, tube_width


def rdp_simplify(points: list[tuple[int, int]], *, epsilon: float) -> list[tuple[int, int]]:
    if len(points) <= 2:
        return list(points)
    start = np.array(points[0], dtype=np.float64)
    end = np.array(points[-1], dtype=np.float64)
    segment = end - start
    denom = float(np.dot(segment, segment))
    max_dist = -1.0
    max_index = 0
    for index, point in enumerate(points[1:-1], start=1):
        p = np.array(point, dtype=np.float64)
        if denom == 0.0:
            dist = float(np.linalg.norm(p - start))
        else:
            t = float(np.dot(p - start, segment) / denom)
            proj = start + max(0.0, min(1.0, t)) * segment
            dist = float(np.linalg.norm(p - proj))
        if dist > max_dist:
            max_dist = dist
            max_index = index
    if max_dist > float(epsilon):
        left = rdp_simplify(points[: max_index + 1], epsilon=epsilon)
        right = rdp_simplify(points[max_index:], epsilon=epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def _paint_point(mask: np.ndarray, point: tuple[int, int], radius: int) -> None:
    y, x = point
    y0, y1 = max(0, y - radius), min(mask.shape[0] - 1, y + radius)
    x0, x1 = max(0, x - radius), min(mask.shape[1] - 1, x + radius)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True


def _paint_segment(
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    radius: int,
) -> None:
    y0, x0 = start
    y1, x1 = end
    steps = max(abs(int(y1) - int(y0)), abs(int(x1) - int(x0)), 1)
    for step in range(steps + 1):
        t = step / steps
        y = round((1.0 - t) * y0 + t * y1)
        x = round((1.0 - t) * x0 + t * x1)
        _paint_point(mask, (y, x), radius)


def _coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(y), int(x)) for y, x in np.argwhere(np.asarray(mask, dtype=bool))]


def _rle_bytes(flat: np.ndarray) -> int:
    if flat.size == 0:
        return 0
    runs = 1
    prev = bool(flat[0])
    for value in flat[1:]:
        cur = bool(value)
        if cur != prev:
            runs += 1
            prev = cur
    return int(runs * 3)


def _worst_unsolved_class(labels: np.ndarray, argmax: np.ndarray) -> int:
    classes, counts = np.unique(labels[labels != argmax], return_counts=True)
    if classes.size == 0:
        return int(np.bincount(labels.reshape(-1).astype(np.int64)).argmax())
    return int(classes[int(np.argmax(counts))])


def _support_margin_stat(margins: np.ndarray | None, support: np.ndarray) -> float | None:
    if margins is None or not np.any(support):
        return None
    values = np.asarray(margins, dtype=np.float64)[support]
    if values.size == 0:
        return None
    return float(-np.median(values))


def _pair_index(pair_indices: np.ndarray | None, batch: int) -> int:
    if pair_indices is None:
        return batch
    arr = np.asarray(pair_indices).reshape(-1)
    if batch >= arr.size:
        return batch
    return int(arr[batch])


def _path_action_id(*, pair_index: int, target_class: int, support_sha256: str) -> str:
    return f"path_tube:p{int(pair_index)}:c{int(target_class)}:{support_sha256[:16]}"


__all__ = [
    "BLOCKER_PATH_ACTION_INFLATE_MISSING",
    "BLOCKER_PATH_ACTION_PARSEBACK_MISSING",
    "BLOCKER_PATH_SUPPORT_NOT_BIRTH",
    "PATH_ACTION_CANDIDATE_SCHEMA",
    "PATH_ACTION_PRODUCER_SCHEMA",
    "PATH_SUPPORT_SCHEMA",
    "PathTubeSupport",
    "build_path_action_candidates_from_arrays",
    "decode_path_tube_payload",
    "largest_connected_component",
    "path_tube_support_from_mask",
    "rdp_simplify",
    "support_encoding_comparison",
    "support_mask_sha256",
]
