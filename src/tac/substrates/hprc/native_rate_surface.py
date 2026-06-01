# SPDX-License-Identifier: MIT
"""Native HPRC residual-protection surfaces from P18/P19 scorer priors."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.scorer_region_waterfill import (
    P18_SEGNET_REGION_WATERFILL_SCHEMA,
    P19_POSENET_NULL_PAIRS_SCHEMA,
)

HPRC_NATIVE_RATE_RESIDUAL_PROTECTION_SURFACE_SCHEMA = (
    "hprc_native_rate_residual_protection_surface.v1"
)
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}


def build_hprc_native_rate_residual_protection_surface(
    *,
    p19_posenet_null_pairs: Mapping[str, Any],
    p18_segnet_region_waterfill: Mapping[str, Any] | None,
    frames: int,
    residual_grid_h: int,
    residual_grid_w: int,
    default_protection: float = 1.0,
    p19_null_protection: float = 0.15,
    p18_region_protection: float = 1.0,
    gop_size: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a residual-token protection map for native rate-aware HPRC training.

    Protection is in ``[0, 1]``.  ``1`` means "protect this residual token from
    train-time rate pressure"; ``0`` means "safe to shrink first."  P19-null
    pairs lower protection broadly, then P18 SegNet-sensitive regions raise it
    back inside vulnerable boxes.
    """

    if p19_posenet_null_pairs.get("schema") != P19_POSENET_NULL_PAIRS_SCHEMA:
        raise ValueError("P19 PoseNet-null artifact schema mismatch")
    if p18_segnet_region_waterfill is not None and (
        p18_segnet_region_waterfill.get("schema") != P18_SEGNET_REGION_WATERFILL_SCHEMA
    ):
        raise ValueError("P18 SegNet-region artifact schema mismatch")
    frame_count = _positive_int(frames, "frames")
    grid_h = _positive_int(residual_grid_h, "residual_grid_h")
    grid_w = _positive_int(residual_grid_w, "residual_grid_w")
    gop = _positive_int(gop_size, "gop_size")
    expected_pairs = (frame_count + gop - 1) // gop
    p19_pairs = _pair_ids(p19_posenet_null_pairs.get("selected_pair_ids"))
    p19_n_pairs = _positive_int(p19_posenet_null_pairs.get("n_pairs"), "p19.n_pairs")

    protection = np.full(
        (frame_count, grid_h, grid_w, 3),
        np.clip(float(default_protection), 0.0, 1.0),
        dtype=np.float32,
    )
    in_range_p19_pairs: list[int] = []
    for pair_id in p19_pairs:
        if pair_id < 0 or pair_id >= expected_pairs:
            continue
        in_range_p19_pairs.append(pair_id)
        for frame_index in _frames_for_pair(pair_id, frame_count=frame_count, gop_size=gop):
            protection[frame_index, :, :, :] = np.clip(float(p19_null_protection), 0.0, 1.0)

    p18_region_cell_count = 0
    p18_pair_ids: list[int] = []
    if p18_segnet_region_waterfill is not None:
        for row in p18_segnet_region_waterfill.get("rows") or []:
            if not isinstance(row, Mapping):
                continue
            pair_id = int(row.get("pair_id", -1))
            if pair_id < 0 or pair_id >= expected_pairs:
                continue
            p18_pair_ids.append(pair_id)
            for region in row.get("regions256") or []:
                if not isinstance(region, Mapping) or not isinstance(region.get("box"), Mapping):
                    continue
                y0 = _grid_start(float(region["box"]["y0"]), grid_h)
                y1 = _grid_end(float(region["box"]["y1"]), grid_h)
                x0 = _grid_start(float(region["box"]["x0"]), grid_w)
                x1 = _grid_end(float(region["box"]["x1"]), grid_w)
                for frame_index in _frames_for_pair(pair_id, frame_count=frame_count, gop_size=gop):
                    protection[frame_index, y0:y1, x0:x1, :] = np.clip(
                        float(p18_region_protection),
                        0.0,
                        1.0,
                    )
                p18_region_cell_count += max(0, y1 - y0) * max(0, x1 - x0)

    blockers: list[str] = []
    if not in_range_p19_pairs:
        blockers.append("no_p19_selected_pairs_in_hprc_training_span")
    if p19_n_pairs != expected_pairs:
        blockers.append(
            f"p19_pair_count_differs_from_hprc_training_span:p19={p19_n_pairs}:hprc={expected_pairs}"
        )
    p18_n_pairs = None
    if p18_segnet_region_waterfill is not None:
        p18_n_pairs = _positive_int(
            p18_segnet_region_waterfill.get("n_pairs_available"),
            "p18.n_pairs_available",
        )
        if p18_n_pairs != expected_pairs:
            blockers.append(
                f"p18_pair_count_differs_from_hprc_training_span:p18={p18_n_pairs}:hprc={expected_pairs}"
            )
    evidence_scope = (
        "full_video"
        if p19_n_pairs == expected_pairs and (p18_n_pairs is None or p18_n_pairs == expected_pairs)
        else "prefix_or_cross_archive_projection"
    )
    manifest = {
        "schema": HPRC_NATIVE_RATE_RESIDUAL_PROTECTION_SURFACE_SCHEMA,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frames": frame_count,
        "gop_size": gop,
        "expected_pairs": expected_pairs,
        "residual_grid_h": grid_h,
        "residual_grid_w": grid_w,
        "shape": [int(v) for v in protection.shape],
        "dtype": str(protection.dtype),
        "semantics": "1=protect_from_rate_pressure,0=safest_to_shrink",
        "default_protection": float(default_protection),
        "p19_null_protection": float(p19_null_protection),
        "p18_region_protection": float(p18_region_protection),
        "p19_selected_pair_count": len(p19_pairs),
        "p19_in_range_pair_count": len(in_range_p19_pairs),
        "p19_in_range_pair_ids": in_range_p19_pairs,
        "p18_in_range_pair_count": len(set(p18_pair_ids)),
        "p18_region_cell_count": int(p18_region_cell_count),
        "protection_min": float(np.min(protection)),
        "protection_max": float(np.max(protection)),
        "protection_mean": float(np.mean(protection)),
        "rate_pressure_mean": float(np.mean(1.0 - protection)),
        "evidence_scope": evidence_scope,
        "blockers": blockers,
        "array_sha256": _array_sha256(protection),
        **FALSE_AUTHORITY,
    }
    return protection, manifest


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _pair_ids(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _positive_int(value: object, label: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"{label} must be positive")
    return out


def _frames_for_pair(pair_id: int, *, frame_count: int, gop_size: int) -> range:
    start = int(pair_id) * int(gop_size)
    return range(start, min(start + int(gop_size), int(frame_count)))


def _grid_start(value: float, size: int) -> int:
    return max(0, min(int(size), int(np.floor(float(value) * int(size)))))


def _grid_end(value: float, size: int) -> int:
    return max(0, min(int(size), int(np.ceil(float(value) * int(size)))))


def _array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(json.dumps(list(arr.shape), sort_keys=True).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


__all__ = [
    "FALSE_AUTHORITY",
    "HPRC_NATIVE_RATE_RESIDUAL_PROTECTION_SURFACE_SCHEMA",
    "build_hprc_native_rate_residual_protection_surface",
    "load_json_object",
]
