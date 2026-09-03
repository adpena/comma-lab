# SPDX-License-Identifier: MIT
"""Instance-independent integer compensation search used by QS5 and FPC2."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import ClipConfig, PipelineBlocked, file_fact, require_device


@dataclasses.dataclass(frozen=True)
class CompensationRequest:
    archive: Path
    archive_sha256: str
    archive_bytes: int
    clip: ClipConfig
    device: str
    pair_ids: tuple[int, ...]

    def validate(self) -> dict[str, Any]:
        archive = file_fact(self.archive)
        if archive["sha256"] != self.archive_sha256 or archive["bytes"] != self.archive_bytes:
            raise PipelineBlocked("compensation archive declaration differs from bytes")
        if not self.pair_ids or len(set(self.pair_ids)) != len(self.pair_ids):
            raise ValueError("compensation pair scope must be nonempty and unique")
        if min(self.pair_ids) < 0 or max(self.pair_ids) >= self.clip.pair_count:
            raise ValueError("compensation pair scope exceeds clip geometry")
        device = require_device(self.device)
        return {"archive": archive, "clip": self.clip.as_dict(), "device": device.as_dict(), "pair_ids": list(self.pair_ids)}


def integer_coordinate_descent(
    initial: np.ndarray,
    objective: Callable[[np.ndarray], float],
    *,
    lower: int,
    upper: int,
    max_passes: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Strictly admit measured ±1 integer moves; ties preserve input bytes."""

    current = np.asarray(initial, dtype=np.int32).copy()
    if current.ndim != 1 or lower > upper or max_passes < 1:
        raise ValueError("invalid integer compensation search geometry")
    if np.any(current < lower) or np.any(current > upper):
        raise ValueError("initial compensation code is outside the lattice")
    current_value = float(objective(current.copy()))
    evaluations = 1
    accepted: list[dict[str, Any]] = []
    for pass_index in range(max_passes):
        changed = False
        for coordinate in range(current.size):
            best = current
            best_value = current_value
            for delta in (-1, 1):
                value = int(current[coordinate]) + delta
                if value < lower or value > upper:
                    continue
                candidate = current.copy()
                candidate[coordinate] = value
                measured = float(objective(candidate.copy()))
                evaluations += 1
                if measured < best_value:
                    best, best_value = candidate, measured
            if best is not current:
                accepted.append({"pass": pass_index, "coordinate": coordinate, "before": current_value, "after": best_value})
                current, current_value, changed = best, best_value, True
        if not changed:
            break
    return current, {
        "initial_objective": float(objective(np.asarray(initial, dtype=np.int32).copy())),
        "final_objective": current_value,
        "evaluations": evaluations + 1,
        "accepted_moves": accepted,
        "strict_improvement": bool(accepted),
        "stop": "complete pass with no strict improvement" if not changed else "max_passes",
    }


def restore_neutral_connective_support(
    site_rows: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, int]]:
    """QS5's exact selection kernel, lifted without its pinned filesystem."""

    sites: list[int] = []
    counts = {
        "strict_sites": 0,
        "neutral_restored_sites": 0,
        "negative_sites_excluded": 0,
        "model_B": 0,
        "model_H": 0,
        "model_W": 0,
    }
    for row in site_rows:
        strict = bool(row["strict_support_keep"])
        neutral = int(row["B"]) == int(row["H"]) and int(row["W"]) == 0
        if strict or neutral:
            sites.append(int(row["site_flat"]))
            counts["model_B"] += int(row["B"])
            counts["model_H"] += int(row["H"])
            counts["model_W"] += int(row["W"])
            counts["strict_sites"] += int(strict)
            counts["neutral_restored_sites"] += int(neutral and not strict)
        else:
            counts["negative_sites_excluded"] += 1
    return np.asarray(sorted(sites), dtype=np.int64), counts
