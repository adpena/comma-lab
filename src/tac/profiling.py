# SPDX-License-Identifier: MIT
"""Stage-level timer profiling for pipeline hot-spot analysis.

Usage::

    from tac.profiling import PipelineProfiler

    profiler = PipelineProfiler(budget_seconds=1800)

    with profiler.stage("load_frames"):
        frames = torch.load("frames.pt")

    with profiler.stage("upscale"):
        frames = F.interpolate(frames, ...)

    # Get results dict (JSON-serializable)
    report = profiler.report()

    # Save to disk alongside other results
    profiler.save("path/to/timings.json")

    # Human-readable summary
    profiler.print_report()
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class StageTimer:
    """Context manager for timing a single pipeline stage."""

    __slots__ = ("name", "start", "end", "elapsed")

    def __init__(self, name: str) -> None:
        self.name = name
        self.start: float | None = None
        self.end: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> "StageTimer":
        self.start = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end = time.monotonic()
        self.elapsed = self.end - self.start  # type: ignore[operator]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elapsed_s": round(self.elapsed, 4) if self.elapsed is not None else None,
        }


class PipelineProfiler:
    """Accumulates stage timings and produces budget-aware reports.

    Args:
        budget_seconds: Total time budget (e.g. 1800 for 30-min contest limit).
            If None, budget checks are skipped.
        name: Optional pipeline name for the report header.
    """

    def __init__(
        self,
        budget_seconds: float | None = 1800.0,
        name: str = "pipeline",
    ) -> None:
        self.budget_seconds = budget_seconds
        self.name = name
        self._stages: list[StageTimer] = []
        self._wall_start: float = time.monotonic()
        self._metadata: dict[str, Any] = {}

    # ── Public API ──────────────────────────────────────────────────────

    @contextmanager
    def stage(self, name: str):
        """Context manager that times a named stage.

        Yields the StageTimer so callers can inspect elapsed time inline::

            with profiler.stage("decode") as t:
                decode()
            print(f"decode took {t.elapsed:.1f}s")
        """
        timer = StageTimer(name)
        self._stages.append(timer)
        with timer:
            yield timer

    def add_metadata(self, key: str, value: Any) -> None:
        """Attach arbitrary metadata (device, frame count, etc.) to the report."""
        self._metadata[key] = value

    def total_elapsed(self) -> float:
        """Wall-clock time since profiler creation."""
        return time.monotonic() - self._wall_start

    def stage_elapsed(self) -> dict[str, float]:
        """Map of stage name -> elapsed seconds."""
        return {
            s.name: round(s.elapsed, 4) if s.elapsed is not None else 0.0
            for s in self._stages
        }

    def report(self) -> dict[str, Any]:
        """Generate a JSON-serializable report dict."""
        stages = self.stage_elapsed()
        total = self.total_elapsed()
        accounted = sum(stages.values())
        overhead = total - accounted

        result: dict[str, Any] = {
            "pipeline": self.name,
            "stages": stages,
            "total_wall_s": round(total, 4),
            "accounted_s": round(accounted, 4),
            "overhead_s": round(overhead, 4),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        if self.budget_seconds is not None:
            remaining = self.budget_seconds - total
            result["budget_s"] = self.budget_seconds
            result["remaining_s"] = round(remaining, 4)
            result["within_budget"] = remaining > 0
            result["budget_utilization_pct"] = round(
                100.0 * total / self.budget_seconds, 2
            )

        # Identify hot spots: stages sorted by time descending
        sorted_stages = sorted(stages.items(), key=lambda kv: kv[1], reverse=True)
        result["hot_spots"] = [
            {
                "stage": name,
                "elapsed_s": elapsed,
                "pct_of_total": round(100.0 * elapsed / total, 1) if total > 0 else 0.0,
            }
            for name, elapsed in sorted_stages
        ]

        if self._metadata:
            result["metadata"] = self._metadata

        return result

    def save(self, path: str | Path) -> Path:
        """Write the report to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.report(), f, indent=2)
        return path

    def print_report(self) -> None:
        """Print a human-readable timing summary to stdout."""
        r = self.report()
        total = r["total_wall_s"]

        print(f"\n{'=' * 60}")
        print(f"  Pipeline Profiling: {self.name}")
        print(f"{'=' * 60}")

        for hs in r["hot_spots"]:
            bar_len = int(hs["pct_of_total"] / 2.5)  # 40-char max bar
            bar = "#" * bar_len
            print(f"  {hs['stage']:30s} {hs['elapsed_s']:8.2f}s  ({hs['pct_of_total']:5.1f}%) {bar}")

        print(f"  {'---':30s} {'---':>8s}")
        print(f"  {'accounted':30s} {r['accounted_s']:8.2f}s")
        print(f"  {'overhead':30s} {r['overhead_s']:8.2f}s")
        print(f"  {'TOTAL':30s} {total:8.2f}s")

        if "budget_s" in r:
            status = "PASS" if r["within_budget"] else "FAIL"
            print(f"\n  Budget: {r['budget_s']:.0f}s | Used: {total:.1f}s | "
                  f"Remaining: {r['remaining_s']:.1f}s | {status} "
                  f"({r['budget_utilization_pct']:.1f}%)")

        print(f"{'=' * 60}\n")
