"""Compute cost tracking for all platforms.

Every experiment records: platform, GPU, cost/hr, runtime, total cost.
This prevents $22 surprises and enables ROI analysis per experiment.

Usage::

    from tac.cost_tracker import estimate_cost, record_cost, CostRecord

    # Pre-flight estimate
    est = estimate_cost("modal", "a10g", estimated_hours=2.0)
    print(f"Estimated cost: ${est.total_cost:.2f}")

    # Record after completion
    record = CostRecord.from_run(
        platform="local", gpu="mps", runtime_seconds=3600
    )
    record_cost(record, run_dir=Path("./eval_runs/my_run"))
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── Modal rates (Apr 2026) ────────────────────────────────────────────
MODAL_RATES: dict[str, float] = {
    "cpu_core": 0.0473,       # per core per hour
    "memory_gib": 0.0048,     # per GiB per hour
    "gpu_t4": 0.59,
    "gpu_l4": 0.56,
    "gpu_a10g": 1.04,
    "gpu_l40s": 1.85,
    "gpu_a100": 3.72,
    "gpu_h100": 3.95,
    "gpu_rtx_pro_6000": 3.03,
}

# ── Other platform rates ─────────────────────────────────────────────
PLATFORM_RATES: dict[str, float] = {
    "local_mps": 0.0,        # free (electricity only)
    "local_cpu": 0.0,
    "lightning_t4": 0.0,     # free tier (79hr/month)
    "kaggle_p100": 0.0,      # free tier (30hr/week)
    "kaggle_t4": 0.0,        # free tier
    "kaggle_t4x2": 0.0,      # free tier
}


def _rate_for(platform: str, gpu: str) -> tuple[float, bool]:
    """Return (rate_per_hour, is_free_tier) for a platform+gpu combo."""
    key = f"{platform}_{gpu}".lower()
    if key in PLATFORM_RATES:
        return PLATFORM_RATES[key], True

    # Modal GPU rate
    modal_key = f"gpu_{gpu}".lower()
    if platform.lower() == "modal" and modal_key in MODAL_RATES:
        return MODAL_RATES[modal_key], False

    # Unknown — assume free local
    return 0.0, True


@dataclass
class CostRecord:
    """A single cost record for one experiment or stage."""

    platform: str = "local"        # "local", "modal", "lightning", "kaggle"
    gpu: str = "cpu"               # "mps", "t4", "a10g", "p100", "cpu", etc.
    rate_per_hour: float = 0.0     # $/hr
    runtime_seconds: float = 0.0   # actual wall clock
    runtime_hours: float = 0.0     # runtime_seconds / 3600
    total_cost: float = 0.0        # rate * hours
    is_free_tier: bool = True
    estimated_hours: float | None = None  # pre-flight estimate (if any)
    estimated_cost: float | None = None   # pre-flight estimate (if any)

    @classmethod
    def from_run(
        cls,
        platform: str,
        gpu: str,
        runtime_seconds: float,
        estimated_hours: float | None = None,
    ) -> CostRecord:
        """Build a cost record from actual run data."""
        rate, is_free = _rate_for(platform, gpu)
        hours = runtime_seconds / 3600.0
        return cls(
            platform=platform,
            gpu=gpu,
            rate_per_hour=rate,
            runtime_seconds=runtime_seconds,
            runtime_hours=round(hours, 4),
            total_cost=round(rate * hours, 4),
            is_free_tier=is_free,
            estimated_hours=estimated_hours,
            estimated_cost=round(rate * estimated_hours, 4) if estimated_hours is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageTimer:
    """Tracks start/end/duration for a pipeline stage."""

    stage: str
    start_iso: str = ""
    end_iso: str = ""
    duration_seconds: float = 0.0
    _start_mono: float = field(default=0.0, repr=False)

    def start(self) -> StageTimer:
        """Mark stage start. Returns self for chaining."""
        import datetime
        self.start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._start_mono = time.monotonic()
        return self

    def stop(self) -> StageTimer:
        """Mark stage end. Returns self for chaining."""
        import datetime
        self.end_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.duration_seconds = round(time.monotonic() - self._start_mono, 2)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start_iso,
            "end": self.end_iso,
            "duration_seconds": self.duration_seconds,
        }


def estimate_cost(platform: str, gpu: str, estimated_hours: float) -> CostRecord:
    """Pre-flight cost estimate before launching an experiment."""
    rate, is_free = _rate_for(platform, gpu)
    return CostRecord(
        platform=platform,
        gpu=gpu,
        rate_per_hour=rate,
        runtime_seconds=0.0,
        runtime_hours=0.0,
        total_cost=0.0,
        is_free_tier=is_free,
        estimated_hours=estimated_hours,
        estimated_cost=round(rate * estimated_hours, 4),
    )


def record_cost(cost: CostRecord, run_dir: Path) -> None:
    """Write cost record to run_dir/cost.json."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cost_path = run_dir / "cost.json"
    cost_path.write_text(json.dumps(cost.to_dict(), indent=2) + "\n")


def print_cost_estimate(gpu: str, estimated_hours: float, platform: str = "modal") -> None:
    """Print a human-readable cost estimate. For pre-flight display."""
    est = estimate_cost(platform, gpu, estimated_hours)
    print(f"  Platform: {platform}")
    print(f"  GPU: {gpu} @ ${est.rate_per_hour:.2f}/hr")
    print(f"  Est. runtime: {estimated_hours:.1f} hours")
    print(f"  Est. cost: ${est.estimated_cost:.2f}")
    if est.is_free_tier:
        print("  (free tier — no charge)")


def collect_replicability_metadata(device: str = "cpu") -> dict[str, Any]:
    """Collect comprehensive replicability metadata for the current environment.

    Captures Python version, torch version, GPU info, library versions,
    git state, and relevant environment variables. Safe to call on any
    platform — missing libraries produce sensible defaults.
    """
    import os
    import platform
    import subprocess
    import sys

    meta: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "exact_command": " ".join(sys.argv),
        "environment_variables": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("INFLATE_", "TAC_", "COMMA_", "PYTHONPATH", "PACT_"))
        },
    }

    # Torch
    try:
        import torch
        meta["torch_version"] = torch.__version__
        meta["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        )
        meta["gpu_vram_mb"] = (
            torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
            if torch.cuda.is_available()
            else 0
        )
        meta["mps_available"] = (
            torch.backends.mps.is_available()
            if hasattr(torch.backends, "mps")
            else False
        )
    except ImportError:
        meta["torch_version"] = "not installed"
        meta["gpu_name"] = "none"
        meta["gpu_vram_mb"] = 0
        meta["mps_available"] = False

    # numpy
    try:
        import numpy as np
        meta["numpy_version"] = np.__version__
    except ImportError:
        meta["numpy_version"] = "not installed"

    # av (PyAV)
    try:
        import av
        meta["av_version"] = av.__version__
    except ImportError:
        meta["av_version"] = "not installed"

    # git state
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        meta["git_commit"] = "unknown"

    try:
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        meta["git_dirty"] = bool(porcelain)
    except (subprocess.CalledProcessError, FileNotFoundError):
        meta["git_dirty"] = False

    return meta
