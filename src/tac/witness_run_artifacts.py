# SPDX-License-Identifier: MIT
"""Canonical witness run-artifact CONTRACT — the single source of truth for the
filenames + roles every level-set witness run emits into its ``--out-dir``.

**Why this module exists (the drift class it dissolves).** The witness trainer
(``experiments/train_levelset_witness_realized_through_R_mlx.py``) writes a known
set of artifacts; ~30 consumers (``tools/witness_checkin``, ``tools/costate_digest``,
the dashboard, the launcher, the observers, the byte-close path) each hardcoded
those filenames *independently*. That is a drift generator, and it bit us twice at
once (2026-07-10):

  1. the run-dir naming drifted ``levelset_n600_witness_*`` -> ``levelset_v752_baseline_*``
     so ``witness_checkin``'s fallback glob went blind; and
  2. the "signal file" set omitted the streaming progress log, so a genuinely-live
     run — emitting a ``loss_terms`` JSONL row every epoch but *before* its first
     checkpoint — was reported ``STALE: NO signal files``. A false-RED liveness
     confound (the inverse of the spike-guard frozen-run false-GREEN).

Every surface that discovers a run dir or judges liveness/freshness MUST reference
THIS contract instead of re-spelling the names. A drift test
(``src/tac/tests/test_witness_run_artifacts_contract.py``) parses the trainer
source and asserts its emitted filenames are all declared here, so adding/renaming
a trainer artifact fails closed until the contract is updated.

Roles (verified against the trainer source 2026-07-10):
  * ``BEST_JSON``     — tiny best-checkpoint POINTER (keys d_seg/epoch/path/ts)
  * ``RESUME_NPZ``    — crash-resume state sidecar (written per ``--ckpt-every``)
  * ``EMA_NPZ``       — EMA SHADOW weights = the inference / byte-close checkpoint
  * ``POLYAK_NPZ``    — Polyak-finisher deploy npz (only when that arm is on)
  * ``TRAIN_RESULT_JSON`` — final train-result summary (end of run)
  * ``COSTATE_JSONL`` — costate shadow-observer telemetry (observer-written; optional)
  * progress logs     — the daemon's stdout capture (a ``loss_terms`` JSONL row per
    epoch). The FILENAME is caller-chosen (``daemon.log`` under the durable daemon,
    ``run.log`` under the launcher), so liveness globs ``*.log`` — but EXCLUDES
    ``observer.log`` (the costate observer's continuous heartbeat), which would
    otherwise mask a hung trainer.
"""
from __future__ import annotations

from pathlib import Path

# ---- run-dir discovery -------------------------------------------------------
# The witness-run family glob. The historical ``levelset_n600_witness_*`` pattern
# and the drifted ``levelset_v752_baseline_*`` are both siblings under this family.
RUN_DIR_GLOB = "levelset_*"

# ---- trainer-written artifacts (the run's PRODUCED outputs) -------------------
BEST_JSON = "levelset_best.json"                       # tiny best-checkpoint pointer
RESUME_NPZ = "levelset_resume_state.npz"               # crash-resume state sidecar
EMA_NPZ = "levelset_witness_ema_mlx.npz"               # EMA shadow = deploy/byte-close ckpt
EMA_BEST_NPZ = "levelset_witness_ema_BEST.npz"         # best-scoring EMA ckpt (retention)
LIVE_NPZ = "levelset_witness_live_mlx.npz"             # live (non-EMA) weights
POLYAK_NPZ = "levelset_witness_polyak_mlx.npz"         # Polyak-finisher deploy npz
TRAIN_RESULT_JSON = "levelset_train_result.json"       # end-of-run summary

#: Every filename the trainer / checkpoint-retention writes. The drift test asserts
#: the producer sources emit no ``levelset_*.{npz,json}`` literal outside this set.
TRAINER_ARTIFACTS: tuple[str, ...] = (
    BEST_JSON,
    RESUME_NPZ,
    EMA_NPZ,
    EMA_BEST_NPZ,
    LIVE_NPZ,
    POLYAK_NPZ,
    TRAIN_RESULT_JSON,
)

# ---- observer-written telemetry (optional; not produced by the trainer) ------
COSTATE_JSONL = "costate_shadow.jsonl"

#: All NAMED signal files a consumer may look for (produced + observer telemetry).
SIGNAL_FILES: tuple[str, ...] = (*TRAINER_ARTIFACTS, COSTATE_JSONL)

# ---- streaming progress logs (the pre-first-checkpoint liveness signal) -------
PROGRESS_LOG_GLOB = "*.log"
#: Log names that are NOT the trainer's stdout and must not count as trainer
#: liveness (they can stay fresh while the trainer is hung -> false GREEN).
_NON_TRAINER_LOG_SUFFIXES: tuple[str, ...] = ("observer.log",)


def is_run_dir(path: Path | str) -> bool:
    """True if ``path`` is a directory matching the witness-run family glob."""
    p = Path(path)
    return p.is_dir() and p.match(RUN_DIR_GLOB)


def signal_paths(run_dir: Path | str) -> list[Path]:
    """Existing NAMED signal files in ``run_dir`` (checkpoint + telemetry)."""
    rd = Path(run_dir)
    return [rd / name for name in SIGNAL_FILES if (rd / name).exists()]


def progress_log_paths(run_dir: Path | str) -> list[Path]:
    """Existing streaming trainer-stdout logs in ``run_dir``.

    Globs ``*.log`` (the daemon/launcher name the trainer's stdout) but excludes
    ``observer.log`` so the continuous costate-observer heartbeat cannot mask a
    hung trainer.
    """
    return [
        p
        for p in sorted(Path(run_dir).glob(PROGRESS_LOG_GLOB))
        if not any(p.name.endswith(sfx) for sfx in _NON_TRAINER_LOG_SUFFIXES)
    ]


def liveness_paths(run_dir: Path | str) -> list[Path]:
    """Every on-disk file that evidences a live/progressing run.

    Named signals PLUS the streaming trainer log. A STALE detector MUST consult
    this set; consulting only :func:`signal_paths` false-flags a live run that has
    not yet reached its first checkpoint (the 2026-07-10 false-RED bug).
    """
    return signal_paths(run_dir) + progress_log_paths(run_dir)


def freshest_signal_age_s(run_dir: Path | str, now: float) -> float | None:
    """Age in seconds of the newest liveness file, or ``None`` if none exist.

    ``now`` and file mtimes are epoch seconds on the same clock (no timezone
    parsing). A fresh trainer log with no checkpoint yet is a live pre-first-
    checkpoint run, NOT stale.
    """
    ages = [
        max(0.0, now - p.stat().st_mtime)
        for p in liveness_paths(run_dir)
        if p.exists()
    ]
    return min(ages) if ages else None


def newest_run_dir(results_dir: Path | str) -> Path | None:
    """Newest witness-run dir under ``results_dir`` by freshest-signal mtime.

    Falls back to the dir's own mtime when a run dir has no signal files yet, so a
    just-created run still ranks. ``None`` if no run dir matches the family glob.
    """
    rd = Path(results_dir)
    candidates = [d for d in rd.glob(RUN_DIR_GLOB) if d.is_dir()]
    if not candidates:
        return None

    def _key(d: Path) -> float:
        paths = liveness_paths(d)
        if paths:
            return max(p.stat().st_mtime for p in paths if p.exists())
        return d.stat().st_mtime

    return max(candidates, key=_key)
