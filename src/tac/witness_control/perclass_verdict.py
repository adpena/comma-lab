# SPDX-License-Identifier: MIT
"""Per-class verdict decomposition + process-memory telemetry fields (2026-07-07).

WHY (two named gaps, both operator-directed "enhance all necessary telemetry now"):
1. PER-CLASS d_seg TRAJECTORIES — the §15 solver-pack agent measured an APPARATUS GAP: no run has
   ever logged per-class d_seg per verdict (rows are total-only), so the pre-registered
   α_lane < α_road weak-KAM check and every per-class exit criterion (§14 item 5,
   ``powerlaw_meat_exit``'s per-class input, the #315 per-class λ sensors) are starved. The
   decomposition below reuses the realized SegNet argmax the annulus-telemetry verdict ALREADY
   collects (same forward, zero extra scorer cost).
2. CONTINUOUS MEMORY TELEMETRY (#329) — the live RSS curve was unobservable (ep3-only probe);
   leak-vs-plateau was decided from ps snapshots. One row of process/MLX memory per verdict epoch
   closes it.

DISCIPLINE: score-neutral READ-ONLY observability => defaults ON per the orphaned-signal rule
(CLAUDE.md "'Off' is a tracked queue"). Byte-identity of the SCORE path is preserved by
construction: nothing here is read back into training/resume/parity; every function is fail-open
(exceptions => empty/error fields, never a crash into the verdict path).

Class order is the CANONICAL comma10k order (measured 2026-06-27, NEVER luma-sort re-derived):
0=Road · 1=Lane · 2=Undrivable · 3=Movable · 4=MyCar.
"""
from __future__ import annotations

from typing import Any

import numpy as np

N_CLASSES = 5
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

_GIB = float(1024**3)


def per_class_flip_stats(
    realized_maps: list[Any], gt_maps: list[Any], n_classes: int = N_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate (flips_by_gt_class, pixels_by_gt_class) over pairs.

    ``realized_maps[i]`` / ``gt_maps[i]``: (H, W) integer argmax maps for pair i (torch tensors or
    numpy arrays; converted via ``np.asarray``). A "flip" is attributed to the GT class of the
    pixel (matching how d_seg's per-pixel disagreement reads against the GT argmax); by
    construction ``sum(flips)/sum(pixels) == d_seg`` over the same pairs.
    """
    flips = np.zeros(n_classes, dtype=np.int64)
    pixels = np.zeros(n_classes, dtype=np.int64)
    for realized, gt in zip(realized_maps, gt_maps, strict=True):
        r = np.asarray(realized)
        g = np.asarray(gt)
        if r.shape != g.shape:
            raise ValueError(f"shape mismatch realized {r.shape} vs gt {g.shape}")
        neq = r != g
        for c in range(n_classes):
            mask_c = g == c
            pixels[c] += int(mask_c.sum())
            flips[c] += int((neq & mask_c).sum())
    return flips, pixels


def per_class_dseg_fields(flips: np.ndarray, pixels: np.ndarray) -> dict[str, list[float]]:
    """Row-ready fields from accumulated stats.

    * ``d_seg_by_class[c]`` = flips_c / pixels_c (the class-conditional flip RATE — the per-class
      trajectory the power-law exit fits consume; 0.0 where the class has no pixels).
    * ``flip_share_by_class[c]`` = flips_c / total_flips (which class CARRIES the residual; 0.0
      vector when there are no flips at all).
    """
    pixels_safe = np.maximum(pixels, 1)
    rate = flips / pixels_safe
    rate[pixels == 0] = 0.0
    total = int(flips.sum())
    share = (flips / total) if total > 0 else np.zeros_like(rate)
    return {
        "d_seg_by_class": [round(float(x), 8) for x in rate],
        "flip_share_by_class": [round(float(x), 6) for x in share],
    }


def memory_telemetry_fields() -> dict[str, float]:
    """Process + MLX memory snapshot for the verdict row (#329). Fail-open: any unavailable
    source is simply omitted; never raises."""
    out: dict[str, float] = {}
    try:
        import psutil

        out["rss_gib"] = round(psutil.Process().memory_info().rss / _GIB, 3)
        out["sys_avail_gib"] = round(psutil.virtual_memory().available / _GIB, 2)
    except Exception:
        pass
    try:
        import mlx.core as mx

        out["mlx_active_gib"] = round(float(mx.get_active_memory()) / _GIB, 3)
        out["mlx_cache_gib"] = round(float(mx.get_cache_memory()) / _GIB, 3)
        out["mlx_peak_gib"] = round(float(mx.get_peak_memory()) / _GIB, 3)
    except Exception:
        pass
    return out


__all__ = [
    "CLASS_NAMES",
    "N_CLASSES",
    "memory_telemetry_fields",
    "per_class_dseg_fields",
    "per_class_flip_stats",
]
