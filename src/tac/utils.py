"""Shared training utilities for CPU and GPU lanes.

Provides signal handling, JSONL telemetry, and formatted epoch logging
so both training.py and train_renderer.py stay DRY.
"""
from __future__ import annotations

import atexit
import json
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path


def setup_signal_handlers(save_fn: Callable[[], None]) -> None:
    """Register SIGTERM/SIGINT/SIGHUP + atexit to call *save_fn* before exit.

    Args:
        save_fn: Zero-arg callable that persists the current training state.
    """
    def _signal_handler(signum, frame):
        try:
            print(f"\n[train] EMERGENCY SAVE (signal {signum})")
            save_fn()
            print("[train] Emergency save complete.")
        except Exception as e:
            print(f"[train] Emergency save FAILED: {e}")
        sys.exit(1)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            pass  # some signals unavailable in threads

    def _atexit_save():
        try:
            save_fn()
        except Exception:
            pass

    atexit.register(_atexit_save)


def write_telemetry(path: str | Path, data: dict) -> None:
    """Append a single JSON object as one line to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


def log_epoch(
    epoch: int,
    total: int,
    loss: float,
    metrics: dict[str, float],
    lr: float,
    elapsed: float,
    tag: str = "",
    extra: str = "",
) -> None:
    """Print a standardised one-line epoch summary.

    Args:
        epoch: Current epoch (0-indexed).
        total: Total number of epochs.
        loss: Average loss this epoch.
        metrics: Dict of metric_name -> value (e.g. {"pose": 0.001, "seg": 0.02}).
        lr: Current learning rate.
        elapsed: Seconds this epoch took.
        tag: Optional run tag for prefix.
        extra: Optional suffix (e.g. " *BEST*").
    """
    eta_hours = elapsed * (total - epoch - 1) / 3600 if elapsed > 0 else 0.0
    parts = [f"[ep {epoch:4d}/{total}]", f"loss={loss:.4f}"]
    for k, v in metrics.items():
        parts.append(f"{k}={v:.6f}")
    parts.append(f"lr={lr:.6f}")
    parts.append(f"{elapsed:.1f}s/ep ETA={eta_hours:.1f}h")
    if extra:
        parts.append(extra)
    print(" ".join(parts))
