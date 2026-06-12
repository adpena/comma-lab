# SPDX-License-Identifier: MIT
"""Durable per-epoch telemetry for the P2 torch-vehicle curriculum driver.

A multi-week Modal run is unobservable unless every epoch's trajectory is
written to durable, queryable storage (the "Max observability" non-negotiable:
inspectable per layer, decomposable per signal, diff-able across runs, queryable
post-hoc). This module is the torch-vehicle's observability surface — the
parity of what the MLX capstone's per-epoch logging gives.

Two durable artifacts, both under the run's ``out_dir``:

* ``torch_vehicle_trajectory.jsonl`` — one APPEND-ONLY row per recorded epoch
  (loss / pose_mse / lr / grad-clip / per-eval d_seg / d_pose / rate / score /
  archive_bytes / wall-clock). The append is atomic-ish (a single ``write`` of
  a newline-terminated JSON line under an ``"a"`` handle); a death never
  corrupts prior rows, and a resumed run keeps appending (the resume position is
  re-derivable from the last row).
* ``torch_vehicle_summary.json`` — a small machine + human readable rollup
  (best score / best epoch / best stage / current position / last-eval row),
  rewritten atomically (tmp + ``os.replace``) on each update.

Authority: telemetry only. The in-loop d_seg/d_pose are
``[macOS-CPU advisory]`` / ``[contest-CPU advisory]`` (NON-PROMOTABLE) until the
byte-closed archive is run through ``upstream/evaluate.py``; this module does
NOT make a score claim, it RECORDS the advisory trajectory. NO MPS.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_TRAJECTORY_NAME = "torch_vehicle_trajectory.jsonl"
_SUMMARY_NAME = "torch_vehicle_summary.json"


@dataclass
class EpochRecord:
    """One recorded epoch's telemetry (loss/lr always; eval fields when evaluated)."""

    stage_index: int
    stage_name: str
    epoch_in_stage: int  # 1-based (the epoch just completed)
    global_epoch: int
    loss: float
    pose_mse: float
    adamw_lr: float
    muon_lr: float | None
    grad_norm_adamw: float | None = None
    grad_norm_muon: float | None = None
    # Eval fields (only present on eval epochs).
    evaluated: bool = False
    d_seg: float | None = None
    d_pose: float | None = None
    rate: float | None = None
    score: float | None = None
    archive_bytes: int | None = None
    is_best: bool = False
    # Bookkeeping.
    wall_clock_s: float = 0.0
    authority_tag: str = "[contest-CPU advisory]"
    promotable: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class TelemetryWriter:
    """Append-only JSONL trajectory + atomically-rewritten summary rollup.

    Construct ONCE per run (resumed runs re-open the same ``out_dir`` in append
    mode — the trajectory continues, the summary is recomputed from the running
    best). ``record(...)`` appends a row and updates the summary.
    """

    def __init__(self, out_dir: str | os.PathLike[str], *, run_meta: dict[str, Any] | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_path = self.out_dir / _TRAJECTORY_NAME
        self.summary_path = self.out_dir / _SUMMARY_NAME
        self._run_meta = dict(run_meta or {})
        self._t0 = time.time()
        self._best_score = float("inf")
        self._best_ep = 0
        self._best_stage = -1
        self._n_records = 0
        self._last_eval: dict[str, Any] | None = None
        # If resuming, seed the running best from any prior summary.
        if self.summary_path.exists():
            try:
                prior = json.loads(self.summary_path.read_text())
                self._best_score = float(prior.get("best_score", float("inf")))
                self._best_ep = int(prior.get("best_ep", 0))
                self._best_stage = int(prior.get("best_stage", -1))
                self._n_records = int(prior.get("n_records", 0))
                self._last_eval = prior.get("last_eval")
            except (json.JSONDecodeError, ValueError, OSError):
                pass

    def record(self, rec: EpochRecord) -> EpochRecord:
        """Append one epoch row; update the running best + summary. Returns ``rec``
        with ``is_best`` set if this eval beat the prior best."""
        rec.wall_clock_s = time.time() - self._t0
        if rec.evaluated and rec.score is not None and rec.score < self._best_score:
            self._best_score = float(rec.score)
            self._best_ep = int(rec.global_epoch)
            self._best_stage = int(rec.stage_index)
            rec.is_best = True
        # Append the trajectory row (one line, flushed).
        with open(self.trajectory_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._n_records += 1
        if rec.evaluated:
            self._last_eval = {
                "global_epoch": rec.global_epoch,
                "stage_index": rec.stage_index,
                "stage_name": rec.stage_name,
                "d_seg": rec.d_seg,
                "d_pose": rec.d_pose,
                "rate": rec.rate,
                "score": rec.score,
                "archive_bytes": rec.archive_bytes,
            }
        self._write_summary()
        return rec

    def _write_summary(self) -> None:
        summary = {
            "run_meta": self._run_meta,
            "best_score": self._best_score,
            "best_ep": self._best_ep,
            "best_stage": self._best_stage,
            "n_records": self._n_records,
            "last_eval": self._last_eval,
            "wall_clock_s": time.time() - self._t0,
            "authority": "[contest-CPU advisory] / [macOS-CPU advisory] — NON-PROMOTABLE; "
            "score is authoritative ONLY after upstream/evaluate.py on the byte-closed archive",
            "promotable": False,
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = self.summary_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
        os.replace(tmp, self.summary_path)

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_ep(self) -> int:
        return self._best_ep

    @property
    def best_stage(self) -> int:
        return self._best_stage


def read_trajectory(out_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read all telemetry rows from the JSONL trajectory (for post-hoc query)."""
    p = Path(out_dir) / _TRAJECTORY_NAME
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_summary(out_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read the summary rollup (None if absent)."""
    p = Path(out_dir) / _SUMMARY_NAME
    if not p.exists():
        return None
    return json.loads(p.read_text())


def render_dashboard(out_dir: str | os.PathLike[str]) -> str:
    """Render a compact human-readable dashboard from the durable artifacts.

    A simple text dashboard (no heavy deps) showing the descent trajectory of
    the eval rows + the running best. Suitable for ``watch`` / log tail during a
    multi-week run.
    """
    summary = read_summary(out_dir)
    rows = read_trajectory(out_dir)
    evals = [r for r in rows if r.get("evaluated")]
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("TORCH-VEHICLE (vendored PR95) — advisory trajectory dashboard")
    lines.append("  authority: [contest-CPU advisory] NON-PROMOTABLE (exact = upstream/evaluate.py)")
    lines.append("=" * 78)
    if summary:
        meta = summary.get("run_meta", {})
        lines.append(
            f"  base_channels={meta.get('base_channels','?')} "
            f"latent_dim={meta.get('latent_dim','?')} n_pairs={meta.get('n_pairs','?')} "
            f"budget_epochs={meta.get('total_epoch_budget','full')}"
        )
        lines.append(
            f"  BEST score={summary.get('best_score'):.5f} "
            f"@ global_ep={summary.get('best_ep')} (stage {summary.get('best_stage')})"
            if summary.get("best_score", float("inf")) != float("inf")
            else "  BEST: (no eval yet)"
        )
        lines.append(f"  records={summary.get('n_records',0)}  wall={summary.get('wall_clock_s',0):.0f}s")
    lines.append("-" * 78)
    lines.append("  global_ep | stage              | d_seg    | d_pose   | rate   | score   | bytes")
    lines.append("-" * 78)
    for r in evals[-20:]:
        marker = " *" if r.get("is_best") else "  "
        lines.append(
            f"{marker}{r.get('global_epoch',0):>9} | {str(r.get('stage_name',''))[:18]:<18} | "
            f"{_fmt(r.get('d_seg')):>8} | {_fmt(r.get('d_pose')):>8} | "
            f"{_fmt(r.get('rate')):>6} | {_fmt(r.get('score')):>7} | {r.get('archive_bytes','?')}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.5f}"
    except (TypeError, ValueError):
        return str(x)


__all__ = [
    "EpochRecord",
    "TelemetryWriter",
    "read_summary",
    "read_trajectory",
    "render_dashboard",
]
