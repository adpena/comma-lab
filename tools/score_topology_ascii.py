#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ASCII visualization of contest score topology near an operating point.

Operator visceral map: render the (d_pose, d_seg) plane around a focal
point as a heatmap of contest score, with the importance-flip threshold
marked. Pure CPU + math.

The contest objective ``S = 100*d_seg + sqrt(10*d_pose) + 25*B/N`` has a
distinctive shape:
  - linear in d_seg (constant gradient 100)
  - concave in d_pose (sqrt rises fast near zero, flattens out)
  - linear in archive_bytes (constant gradient 25/N_REF)

This visualizer renders S(d_seg, d_pose) as a fixed-bytes 2D heatmap so
the operator can SEE:
  - where the focal point sits
  - the importance-flip line (d_pose = 2.5e-4) cutting the plane
  - which direction yields steepest descent in score
  - where the rate-only floor is (constant offset)

Each cell is a Unicode density block: ``  .,:-=+*#%@`` selected by
score quantile within the rendered window. The focal point is marked
with ``X``. The flip line is drawn as a vertical bar of ``|``.

Usage::

    .venv/bin/python tools/score_topology_ascii.py \\
        --d-seg-center 6.7e-4 --d-pose-center 3.4e-5 \\
        --archive-bytes 178258 --rows 24 --cols 60

CLAUDE.md compliance: pure terminal-text output; no scorer load; no
network. The plot is **planning intuition**, not score evidence.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.score_geometry import (  # noqa: E402
    contest_score,
    importance_flip_threshold,
    score_decomposition,
)

DENSITY_RAMP = " .,:-=+*#%@"


def _logspace(low: float, high: float, n: int) -> list[float]:
    """Geometric spacing from low to high (inclusive). low > 0."""
    if low <= 0 or high <= 0:
        raise ValueError("logspace bounds must be positive")
    if n < 2:
        return [low]
    log_low = math.log10(low)
    log_high = math.log10(high)
    step = (log_high - log_low) / (n - 1)
    return [10 ** (log_low + i * step) for i in range(n)]


def render(
    *,
    d_seg_center: float,
    d_pose_center: float,
    archive_bytes: int,
    rows: int = 20,
    cols: int = 60,
    span_decades: float = 1.5,
) -> str:
    """Render a heatmap of contest score on the (d_pose, d_seg) plane.

    The window is geometric, ``span_decades`` decades wide on each axis
    centered on ``(d_pose_center, d_seg_center)``.
    """
    if d_seg_center <= 0:
        d_seg_center = max(d_seg_center, 1e-9)
    if d_pose_center <= 0:
        d_pose_center = max(d_pose_center, 1e-9)
    half = span_decades / 2.0
    pose_axis = _logspace(
        d_pose_center / 10**half,
        d_pose_center * 10**half,
        cols,
    )
    seg_axis = _logspace(
        d_seg_center / 10**half,
        d_seg_center * 10**half,
        rows,
    )
    # Score grid: rows are d_seg (top = high d_seg = bad), cols are d_pose
    # (left = low d_pose = good)
    grid = [
        [
            contest_score(d_seg=s, d_pose=p, archive_bytes=archive_bytes)
            for p in pose_axis
        ]
        for s in seg_axis
    ]
    # Flatten and find quantile bounds for the density ramp
    flat = sorted(v for row in grid for v in row)
    lo = flat[0]
    hi = flat[-1]
    span = hi - lo if hi > lo else 1.0

    # Identify the column index closest to the importance-flip threshold
    flip = importance_flip_threshold()
    flip_col = None
    if pose_axis[0] <= flip <= pose_axis[-1]:
        flip_col = min(
            range(cols),
            key=lambda i: abs(math.log10(pose_axis[i]) - math.log10(flip)),
        )
    # Identify the focal cell
    focal_row = min(range(rows), key=lambda i: abs(math.log10(seg_axis[i]) - math.log10(d_seg_center)))
    focal_col = min(range(cols), key=lambda i: abs(math.log10(pose_axis[i]) - math.log10(d_pose_center)))

    out: list[str] = []
    out.append(f"Score topology near (d_seg={d_seg_center:.3e}, d_pose={d_pose_center:.3e}, B={archive_bytes:,})")
    decomp = score_decomposition(d_seg=d_seg_center, d_pose=d_pose_center, archive_bytes=archive_bytes)
    out.append(
        f"Focal score = {decomp.total:.5f}  "
        f"(seg={decomp.seg_term:.5f}, pose={decomp.pose_term:.5f}, rate={decomp.rate_term:.5f})"
    )
    out.append(
        f"Window: pose ∈ [{pose_axis[0]:.2e}, {pose_axis[-1]:.2e}], "
        f"seg ∈ [{seg_axis[0]:.2e}, {seg_axis[-1]:.2e}], "
        f"score range [{lo:.5f}, {hi:.5f}]"
    )
    if flip_col is not None:
        out.append(
            f"Importance-flip threshold (d_pose={flip:.2e}) marked with '|' at col {flip_col}"
        )
    out.append("")
    # Top axis label (avoid f-string backslash escape for Python 3.11 compat)
    seg_axis_label = "seg \\\\ pose"
    out.append(f"{seg_axis_label:>13} ↓pose-good                        pose-bad↓")
    out.append("")
    # Render rows (top of grid is highest d_seg = worst seg)
    for r in reversed(range(rows)):
        line_chars: list[str] = []
        for c in range(cols):
            v = grid[r][c]
            if r == focal_row and c == focal_col:
                line_chars.append("X")
            elif flip_col is not None and c == flip_col:
                line_chars.append("|")
            else:
                # Quantile -> density ramp index
                idx = min(
                    len(DENSITY_RAMP) - 1,
                    int((v - lo) / span * (len(DENSITY_RAMP) - 1)),
                )
                line_chars.append(DENSITY_RAMP[idx])
        seg_label = f"{seg_axis[r]:.2e}"
        out.append(f"  d_seg={seg_label}  {''.join(line_chars)}")
    out.append("")
    out.append(f"Density ramp (low→high score): '{DENSITY_RAMP}'")
    out.append("Marker key: X = focal point | = importance-flip threshold")
    out.append("")
    out.append("Reading guide:")
    out.append(f"  - Cells LEFT of '|' are pose-dominated (d_pose < {flip:.0e})")
    out.append(f"  - Cells RIGHT of '|' are seg-dominated  (d_pose > {flip:.0e})")
    out.append("  - DARKER cells = LOWER score = better candidate position")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-seg-center", type=float, required=True)
    parser.add_argument("--d-pose-center", type=float, required=True)
    parser.add_argument("--archive-bytes", type=int, required=True)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--cols", type=int, default=60)
    parser.add_argument("--span-decades", type=float, default=1.5,
                        help="Width of plot window in log10 decades (default 1.5)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional file to write the plot to")

    args = parser.parse_args(argv)
    text = render(
        d_seg_center=args.d_seg_center,
        d_pose_center=args.d_pose_center,
        archive_bytes=args.archive_bytes,
        rows=args.rows,
        cols=args.cols,
        span_decades=args.span_decades,
    )
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
