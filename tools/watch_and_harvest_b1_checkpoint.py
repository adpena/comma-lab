#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Durable, detached B1-pilot checkpoint harvester -> first exact HiNeRV score.

WHAT THIS DOES
==============
Polls a B1 HiNeRV-MLX pilot run directory for the next periodic checkpoint
(default epoch 250, with a fallback to the final checkpoint), then:

  1. exports that checkpoint's EMA shadow state to a **backend-only**
     ``archive.zip`` (single member ``x``, magic ``HIV1``, no sidecar) via the
     SAME export path the trainer uses (``export_hi_nerv_mlx_archive``), with
     the model architecture rebuilt from the trainer's own
     ``_config_from_args`` (ZERO config drift);
  2. runs the canonical B2 exact-eval bridge
     (``tools/run_hi_nerv_backend_only_b2_exact_eval.py``) on that archive
     (``--device cpu`` by default) to produce the first real exact contest
     score; and
  3. writes a durable result JSON + a status JSON into the run directory.

WHY DETACHED + DURABLE
======================
Claude-session watchers die to SIGURG at ~3 minutes; only nohup-detached
processes survive (like the pilot itself). This tool is a plain polling loop
designed to be launched detached (``--launch-detached`` does the launch for
you, using the canonical CLAUDE.md ``nohup + bash -c + </dev/null + disown``
pattern) so the first exact score is harvested the moment the checkpoint lands,
independent of any Claude session.

EVIDENCE DISCIPLINE (NON-NEGOTIABLE)
====================================
The B2 score produced here on local macOS CPU is ``[macOS-CPU advisory]`` —
it validates the pipeline + gives the first real number + the trend. It is
NOT authoritative. The authoritative score requires Linux-x86_64-CPU
(``[contest-CPU]``, public leaderboard axis) AND NVIDIA T4 CUDA
(``[contest-CUDA]``, promotion axis) on the SAME archive bytes (paid + deferred
to the operator). The B2 bridge stamps the axis automatically.

DISK HYGIENE (NON-NEGOTIABLE)
=============================
Work happens on the SSD tier (``/Volumes/VertigoDataTier/pact/...``), never
``/tmp``. The ~3.6 GB inflated raw frames the B2 inflate stage produces are
certified rebuildable (deterministic via inflate.sh on the archive sha) and are
auto-deleted by the B2 bridge after eval (this tool does NOT pass
``--keep-work-dir``). The exported archive itself is small (<300 KB) and kept.

MEMORY AWARENESS (do not OOM the pilot)
=======================================
Before the heavy B2 inflate (600 pairs -> ~3.6 GB raw frames decoded into RAM),
the tool checks free memory; if below ``--min-free-gb`` (default 15 GB) it
waits (the pilot uses ~11 GB GPU; we must not starve it). The export step is
cheap (a single forward-free state load + repack).

THIS TOOL IS READ-ONLY W.R.T. THE PILOT
=======================================
It only READS the pilot's ``checkpoints/`` + ``telemetry.jsonl`` + heartbeat. It
never edits the trainer, the runner, the adapter, or the pilot's files, and it
never restarts the pilot.

USAGE
=====
Foreground (blocks; for testing / direct run)::

    .venv/bin/python tools/watch_and_harvest_b1_checkpoint.py \
        --run-dir /Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z \
        --target-epoch 250

Detached (self-relaunch via nohup; returns immediately)::

    .venv/bin/python tools/watch_and_harvest_b1_checkpoint.py \
        --run-dir /Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z \
        --target-epoch 250 --launch-detached

Idempotency: if the result JSON already exists, the tool exits 0 immediately
(no re-harvest, no re-export, no re-eval). Safe to re-launch.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure the in-process ``import experiments.train_substrate_hi_nerv_mlx_local``
# (the standalone export path) + ``import tac.*`` + ``import upstream.*`` resolve
# regardless of cwd / PYTHONPATH. ROOT-CAUSE FIX (2026-06-09): the diverging
# pilot's detached harvester (call_id pid 14354) failed with
# ``ModuleNotFoundError: No module named 'experiments'`` (harvest_status_ep249
# ``state=failed_harvest_exception``) because it ran without PYTHONPATH and this
# bootstrap did not exist — so the harvest NEVER produced an archive. Inserting
# REPO_ROOT + src + upstream here makes the detached harvest self-sufficient.
for _path_entry in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "upstream"):
    _entry_str = str(_path_entry)
    if _entry_str not in sys.path:
        sys.path.insert(0, _entry_str)

# ---------------------------------------------------------------------------
# Constants / canonical defaults (re-derived from the pilot launch contract).
# ---------------------------------------------------------------------------

DEFAULT_TARGET_EPOCH = 250
DEFAULT_POLL_SECONDS = 60.0
DEFAULT_HEARTBEAT_STALE_SECONDS = 7 * 60  # 7 min per the harvest spec
DEFAULT_MIN_FREE_GB = 15.0
DEFAULT_MAX_WAIT_SECONDS = 12 * 60 * 60  # 12h ceiling: checkpoint must appear
DEFAULT_DEVICE = "cpu"

# The pilot's architecture + export contract (verified empirically: model built
# from these reproduces param_count == 228903 and export emits single-member
# ``x`` HIV1, sidecar dropped). The trainer's ``_config_from_args`` owns the
# canonical mapping; we only pass the architecture knobs the pilot passed.
PILOT_ARCH_DEFAULTS: dict[str, Any] = {
    "modelsize_row": "hi_nerv_local_tiny",
    "modelsize_candidate_json": None,
    "num_pairs": 600,
    "output_height": 874,
    "output_width": 1164,
    "seed": 0,
    "latent_dim_coarse": 16,
    "latent_dim_mid": 20,
    "latent_dim_fine": 24,
    "embed_dim": 64,
    "sin_frequency": None,
    "decoder_channels": "36,30,23,17,14,11,8",
}
PILOT_DECODER_CODEC = "int8_mixed"
PILOT_HARD_BYTE_CEILING = 300_000
EXPECTED_PARAM_COUNT = 228_903

RESULT_SCHEMA = "b1_checkpoint_harvest_result.v1"
STATUS_SCHEMA = "b1_checkpoint_harvest_status.v1"

# Non-promotable false-authority markers per Catalog #192/#317/#341.
FALSE_AUTHORITY: dict[str, Any] = {
    "score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

_FORBIDDEN_TMP_PREFIXES = (
    "/tmp/",
    "/var/tmp/",
    "/private/tmp/",
    "/private/var/tmp/",
)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp_path(path: Path, field_name: str) -> None:
    """CLAUDE.md FORBIDDEN_PATTERNS: refuse /tmp paths in persisted artifacts."""
    resolved = str(path)
    if any(resolved.startswith(p) for p in _FORBIDDEN_TMP_PREFIXES):
        raise ValueError(
            f"{field_name} = {resolved!r} starts with a /tmp/-class transient "
            "prefix; use the SSD tier (/Volumes/VertigoDataTier/pact/...) or "
            ".omx/state per CLAUDE.md 'Forbidden /tmp paths in any persisted "
            "artifact' (the transient-evidence trap)."
        )


# ---------------------------------------------------------------------------
# Pilot run-directory inspection (read-only).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointRef:
    """A resolved checkpoint: its meta JSON + the EMA/live state paths."""

    meta_path: Path
    global_epoch: int
    role: str
    loss: float
    ema_state_path: Path
    live_state_path: Path
    selection_metric_key: str
    selection_metric_value: float
    is_final: bool

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "meta_path": str(self.meta_path),
            "global_epoch": int(self.global_epoch),
            "checkpoint_role": self.role,
            "loss": float(self.loss),
            "ema_state_path": str(self.ema_state_path),
            "live_state_path": str(self.live_state_path),
            "checkpoint_selection_metric_key": self.selection_metric_key,
            "checkpoint_selection_metric_value": float(self.selection_metric_value),
            "is_final": bool(self.is_final),
        }


def _load_checkpoint_meta(meta_path: Path) -> dict[str, Any] | None:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, Mapping):
        return None
    if meta.get("schema_version") != "long_training_canonical_checkpoint.v1":
        return None
    return dict(meta)


def _checkpoint_ref_from_meta(meta_path: Path, meta: Mapping[str, Any]) -> CheckpointRef | None:
    ema_raw = meta.get("ema_shadow_state_path")
    live_raw = meta.get("live_state_path")
    if not ema_raw or not live_raw:
        return None
    return CheckpointRef(
        meta_path=meta_path,
        global_epoch=int(meta.get("global_epoch", -1)),
        role=str(meta.get("checkpoint_role", "periodic")),
        loss=float(meta.get("loss", float("nan"))),
        ema_state_path=Path(str(ema_raw)),
        live_state_path=Path(str(live_raw)),
        selection_metric_key=str(meta.get("checkpoint_selection_metric_key", "total")),
        selection_metric_value=float(
            meta.get("checkpoint_selection_metric_value", meta.get("loss", float("nan")))
        ),
        is_final=bool(meta.get("is_final", False)),
    )


def list_checkpoints(checkpoint_dir: Path) -> list[CheckpointRef]:
    """Return all valid canonical checkpoints in ``checkpoint_dir`` (read-only)."""
    if not checkpoint_dir.is_dir():
        return []
    refs: list[CheckpointRef] = []
    for meta_path in sorted(checkpoint_dir.glob("*.meta.json")):
        meta = _load_checkpoint_meta(meta_path)
        if meta is None:
            continue
        ref = _checkpoint_ref_from_meta(meta_path, meta)
        if ref is not None:
            refs.append(ref)
    return refs


def _state_file_exists(state_path: Path) -> bool:
    """The adapter emits ``<path>`` or ``<path>.npsd``; accept either."""
    if state_path.is_file():
        return True
    return state_path.with_suffix(state_path.suffix + ".npsd").is_file()


def _resolve_state_file(state_path: Path) -> Path:
    if state_path.is_file():
        return state_path
    npsd = state_path.with_suffix(state_path.suffix + ".npsd")
    if npsd.is_file():
        return npsd
    return state_path


def select_target_checkpoint(
    checkpoints: Sequence[CheckpointRef],
    *,
    target_epoch: int,
    allow_final_fallback: bool = True,
) -> CheckpointRef | None:
    """Pick the harvest target.

    Preference order:
      1. the earliest periodic/best checkpoint with
         ``global_epoch >= target_epoch - 1`` whose EMA state file actually
         exists on disk;
      2. if ``allow_final_fallback``: the final checkpoint (export only works on
         the end-of-run archive in some configs; the final checkpoint is the
         honest fallback).

    OFF-BY-ONE FIX (2026-06-09): the canonical ``run_long_training`` epoch loop
    writes a periodic checkpoint when ``(epoch + 1) % checkpoint_interval == 0``
    and stamps it with ``global_epoch == epoch`` (0-indexed). So the checkpoint
    at the Nth-epoch cadence boundary (interval N) is named ``epoch{N-1:06d}``
    and carries ``global_epoch = N - 1``. The diverging pilot's harvester
    targeted ``--target-epoch 250`` with a strict ``>= 250`` match, so the
    ``global_epoch=249`` checkpoint (the actual 250th-epoch interval boundary)
    NEVER matched and the harvest never fired. Matching ``>= target_epoch - 1``
    fires on the interval boundary as intended. ``target_epoch <= 0`` clamps the
    threshold floor to 0 so it never goes negative.
    """
    threshold = max(0, int(target_epoch) - 1)
    eligible = [
        c
        for c in checkpoints
        if c.global_epoch >= threshold and _state_file_exists(c.ema_state_path)
    ]
    if eligible:
        return min(eligible, key=lambda c: c.global_epoch)
    if allow_final_fallback:
        finals = [c for c in checkpoints if c.is_final and _state_file_exists(c.ema_state_path)]
        if finals:
            return max(finals, key=lambda c: c.global_epoch)
    return None


# ---------------------------------------------------------------------------
# Heartbeat / liveness (read-only).
# ---------------------------------------------------------------------------


def heartbeat_age_seconds(heartbeat_path: Path, *, now: float | None = None) -> float | None:
    """Return seconds since the last heartbeat line, or None if unreadable.

    Parses the trailing line of the heartbeat log (``<iso> pid=.. run=..``).
    Falls back to file mtime if no parseable timestamp line is present. A
    ``TRAIN_EXIT`` line means the pilot finished -> treat as a special signal
    (age computed from its timestamp; the caller decides whether final-fallback
    harvest is appropriate).
    """
    if not heartbeat_path.is_file():
        return None
    now_ts = time.time() if now is None else now
    try:
        text = heartbeat_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    last_ts: float | None = None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        token = line.split(" ", 1)[0]
        try:
            dt = datetime.strptime(token, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        except ValueError:
            continue
        last_ts = dt.timestamp()
        break
    if last_ts is None:
        try:
            last_ts = heartbeat_path.stat().st_mtime
        except OSError:
            return None
    return max(0.0, now_ts - last_ts)


def heartbeat_reports_train_exit(heartbeat_path: Path) -> bool:
    """True if the heartbeat log contains a TRAIN_EXIT marker (pilot finished)."""
    if not heartbeat_path.is_file():
        return False
    try:
        text = heartbeat_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "TRAIN_EXIT" in text


# ---------------------------------------------------------------------------
# Memory awareness (do not OOM the pilot).
# ---------------------------------------------------------------------------


def free_memory_gb() -> float | None:
    """Best-effort available system memory in GiB (psutil; macOS-friendly)."""
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().available) / (1024.0**3)
    except Exception:
        # Fallback: macOS vm_stat page-based estimate.
        try:
            out = subprocess.run(
                ["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=10
            )
            if out.returncode != 0:
                return None
            page_size = 4096
            free_pages = 0
            for line in out.stdout.splitlines():
                low = line.lower()
                if "page size of" in low:
                    for tok in line.replace(".", " ").split():
                        if tok.isdigit():
                            page_size = int(tok)
                            break
                if low.startswith("pages free") or low.startswith("pages inactive") or low.startswith(
                    "pages speculative"
                ):
                    digits = "".join(ch for ch in line.split(":")[-1] if ch.isdigit())
                    if digits:
                        free_pages += int(digits)
            if free_pages <= 0:
                return None
            return float(free_pages * page_size) / (1024.0**3)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Standalone export: checkpoint EMA state -> backend-only archive.zip.
# ---------------------------------------------------------------------------


def _build_pilot_model_and_config(arch: Mapping[str, Any] | None = None) -> tuple[Any, Any, int]:
    """Rebuild the pilot's HiNeRV MLX model + config (ZERO config drift).

    Reuses the trainer's own ``_config_from_args`` so the architecture is
    canonical, then builds ``HinervSubstrateMLX(cfg)`` and verifies the param
    count matches the pilot's 228903 contract (fail-closed: a mismatch means the
    architecture would not round-trip the checkpoint).
    """
    arch_args = dict(PILOT_ARCH_DEFAULTS)
    if arch:
        arch_args.update(arch)
    # Import lazily so non-export code paths (tests of liveness/idempotency) do
    # not require MLX.
    import mlx.core as mx
    from mlx.utils import tree_flatten

    import experiments.train_substrate_hi_nerv_mlx_local as trainer
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    ns = argparse.Namespace(**arch_args)
    cfg = trainer._config_from_args(ns)
    model = HinervSubstrateMLX(cfg)
    mx.eval(model.parameters())
    param_count = sum(
        int(v.size) for _, v in tree_flatten(model.parameters()) if hasattr(v, "size")
    )
    return model, cfg, param_count


def _load_npsd_state_into_model(model: Any, state_path: Path) -> dict[str, Any]:
    """Load a ``.npsd`` checkpoint state into ``model`` (strict; NO FAKE).

    Mirrors ``adapter.import_state_dict`` exactly (key-set + per-tensor shape
    validation; refuses partial / mismatched resume), but is model-only so the
    harvester does not need to construct the full training adapter (scorer
    teachers, loss weights, etc.).
    """
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten, tree_unflatten

    from tac.substrates._shared.mlx_score_aware.adapter import (
        _tree_name_to_saliency_group,
    )
    from tac.substrates._shared.numpy_portable_inflate import (
        unpack_state_dict_numpy,
    )

    resolved = _resolve_state_file(state_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint state file not found: {state_path}")
    restored = unpack_state_dict_numpy(resolved.read_bytes())
    current_flat = {
        _tree_name_to_saliency_group(name): value
        for name, value in tree_flatten(model.parameters())
    }
    restored_keys = set(restored)
    current_keys = set(current_flat)
    if restored_keys != current_keys:
        missing = sorted(current_keys - restored_keys)[:10]
        extra = sorted(restored_keys - current_keys)[:10]
        raise ValueError(
            "checkpoint key-set mismatch; refusing fake/partial resume "
            f"(missing={missing}, extra={extra})"
        )
    flat_updates: list[tuple[str, Any]] = []
    for name, current in current_flat.items():
        arr = np.asarray(restored[name])
        if tuple(arr.shape) != tuple(current.shape):
            raise ValueError(
                f"checkpoint shape mismatch for {name!r}: "
                f"checkpoint={arr.shape} model={tuple(current.shape)}"
            )
        flat_updates.append((name, mx.array(arr.astype(arr.dtype, copy=False))))
    model.update(tree_unflatten(flat_updates))
    mx.eval(model.parameters())
    return {
        "param_tensors_restored": len(flat_updates),
        "state_file": str(resolved),
        "state_file_bytes": int(resolved.stat().st_size),
    }


@dataclass
class ExportResult:
    archive_path: Path
    archive_sha256: str
    archive_bytes: int
    param_count: int
    zip_members: list[str]
    member_magic: str
    load_info: dict[str, Any]


def export_backend_only_archive_from_checkpoint(
    checkpoint: CheckpointRef,
    out_dir: Path,
    *,
    arch: Mapping[str, Any] | None = None,
    decoder_codec: str = PILOT_DECODER_CODEC,
    hard_byte_ceiling: int = PILOT_HARD_BYTE_CEILING,
    use_ema: bool = True,
) -> ExportResult:
    """Export a checkpoint's EMA (or live) state to a backend-only archive.zip.

    The export reuses the trainer's ``export_hi_nerv_mlx_archive`` (same fn the
    pilot wires into ``export_archive_fn``), which already emits a single-member
    ``x`` (HIV1) and drops any unproven sidecar at the export boundary.
    """
    import zipfile

    _refuse_tmp_path(out_dir, "export out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = checkpoint.ema_state_path if use_ema else checkpoint.live_state_path

    model, _cfg, param_count = _build_pilot_model_and_config(arch)
    # Belt-and-suspenders param-count guard on the CANONICAL pilot path (no arch
    # override): a mismatch means architecture drift. With an explicit arch
    # override (only used for fast tests), the count legitimately differs; the
    # real correctness guard in BOTH cases is import_state_dict's strict key-set
    # + per-tensor shape validation below (which refuses any drift outright).
    if not arch and param_count != EXPECTED_PARAM_COUNT:
        raise ValueError(
            f"rebuilt model param_count={param_count} != expected "
            f"{EXPECTED_PARAM_COUNT}; refusing export (architecture drift would "
            "corrupt the checkpoint round-trip)"
        )
    load_info = _load_npsd_state_into_model(model, state_path)

    from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive

    path, sha, nbytes = export_hi_nerv_mlx_archive(
        model,
        out_dir,
        repo_root=REPO_ROOT,
        decoder_codec=decoder_codec,
        source_backend="mlx",
        hard_byte_ceiling=hard_byte_ceiling,
    )
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        members = zf.namelist()
        magic = b""
        if members:
            magic = zf.read(members[0])[:4]
    return ExportResult(
        archive_path=path,
        archive_sha256=sha,
        archive_bytes=int(nbytes),
        param_count=param_count,
        zip_members=list(members),
        member_magic=magic.decode("latin-1"),
        load_info=load_info,
    )


# ---------------------------------------------------------------------------
# B2 bridge invocation.
# ---------------------------------------------------------------------------


def build_b2_command(
    archive_path: Path,
    out_row: Path,
    work_root: Path,
    *,
    device: str = DEFAULT_DEVICE,
    python_exe: str | None = None,
) -> list[str]:
    """Build the canonical B2 exact-eval bridge command (turnkey)."""
    py = python_exe or sys.executable
    return [
        py,
        "-u",
        str(REPO_ROOT / "tools" / "run_hi_nerv_backend_only_b2_exact_eval.py"),
        "--archive",
        str(archive_path),
        "--device",
        device,
        "--work-root",
        str(work_root),
        "--out-row",
        str(out_row),
        # NB: NOT --keep-work-dir; the bridge auto-deletes the ~3.6GB inflated
        # frames after eval (disk hygiene).
    ]


def run_b2_eval(
    archive_path: Path,
    out_row: Path,
    work_root: Path,
    *,
    device: str = DEFAULT_DEVICE,
    timeout_seconds: int = 3 * 60 * 60,
    runner: Callable[[list[str]], int] | None = None,
) -> dict[str, Any]:
    """Run the B2 bridge; return a summary dict (reads the out_row it wrote)."""
    _refuse_tmp_path(work_root, "B2 work_root")
    _refuse_tmp_path(out_row, "B2 out_row")
    work_root.mkdir(parents=True, exist_ok=True)
    out_row.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_b2_command(archive_path, out_row, work_root, device=device)
    started = _utc_now()
    if runner is not None:
        rc = int(runner(cmd))
    else:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), timeout=timeout_seconds)
        rc = int(proc.returncode)
    summary: dict[str, Any] = {
        "b2_command": cmd,
        "b2_returncode": rc,
        "b2_started_utc": started,
        "b2_finished_utc": _utc_now(),
        "b2_out_row": str(out_row),
    }
    if out_row.is_file():
        try:
            summary["b2_result"] = json.loads(out_row.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            summary["b2_result_read_error"] = str(exc)
    return summary


# ---------------------------------------------------------------------------
# Harvest orchestration.
# ---------------------------------------------------------------------------


@dataclass
class HarvestPaths:
    run_dir: Path
    checkpoint_dir: Path
    heartbeat_path: Path
    result_json: Path
    status_json: Path
    export_dir: Path
    work_root: Path
    harvester_log: Path


def resolve_paths(
    run_dir: Path,
    *,
    target_epoch: int,
    heartbeat_path: Path | None = None,
    work_root: Path | None = None,
) -> HarvestPaths:
    run_dir = run_dir.resolve()
    _refuse_tmp_path(run_dir, "run_dir")
    run_id = run_dir.name
    checkpoint_dir = run_dir / "checkpoints"
    if heartbeat_path is None:
        heartbeat_path = REPO_ROOT / ".omx" / "tmp" / f"heartbeat_b1_{run_id}.log"
    result_json = run_dir / f"hi_nerv_backend_only_ep{target_epoch}_exact_eval.json"
    status_json = run_dir / f"harvest_status_ep{target_epoch}.json"
    export_dir = run_dir / f"harvest_export_ep{target_epoch}"
    if work_root is None:
        work_root = run_dir / f"harvest_b2_work_ep{target_epoch}"
    harvester_log = run_dir / f"harvest_watcher_ep{target_epoch}.log"
    return HarvestPaths(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        heartbeat_path=heartbeat_path,
        result_json=result_json,
        status_json=status_json,
        export_dir=export_dir,
        work_root=work_root,
        harvester_log=harvester_log,
    )


def write_status(
    status_json: Path,
    *,
    state: str,
    target_epoch: int,
    detail: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "state": state,
        "target_epoch": int(target_epoch),
        "updated_utc": _utc_now(),
        "pid": os.getpid(),
        **FALSE_AUTHORITY,
    }
    if detail:
        payload["detail"] = json.loads(json.dumps(detail, default=str))
    status_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = status_json.with_suffix(status_json.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, status_json)


def write_result(
    result_json: Path,
    *,
    target_epoch: int,
    checkpoint: CheckpointRef,
    export: ExportResult,
    b2_summary: Mapping[str, Any],
) -> dict[str, Any]:
    b2_result = b2_summary.get("b2_result") or {}
    final_score = b2_result.get("final_score") if isinstance(b2_result, Mapping) else None
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "harvested_utc": _utc_now(),
        "target_epoch": int(target_epoch),
        "evidence_grade": "[macOS-CPU advisory]",
        "authoritative": False,
        "authoritative_note": (
            "Local macOS CPU B2 score is [macOS-CPU advisory] — pipeline-works "
            "+ first real number + trend. Authoritative requires Linux-x86_64 "
            "CPU ([contest-CPU]) AND NVIDIA T4 CUDA ([contest-CUDA]) on the same "
            "archive bytes (paid + deferred to operator)."
        ),
        "checkpoint": checkpoint.to_jsonable(),
        "export": {
            "archive_path": str(export.archive_path),
            "archive_sha256": export.archive_sha256,
            "archive_bytes": export.archive_bytes,
            "param_count": export.param_count,
            "zip_members": export.zip_members,
            "member_magic": export.member_magic,
            "decoder_codec": PILOT_DECODER_CODEC,
            "hard_byte_ceiling": PILOT_HARD_BYTE_CEILING,
            "state_kind": "ema_shadow",
            "load_info": export.load_info,
        },
        "b2": dict(b2_summary),
        "first_exact_score_advisory": final_score,
        "frontier_pointer_note": (
            "Compare to canonical frontier via tools/refresh_canonical_frontier.py "
            "(contest-CPU axis). Do NOT hardcode the frontier literal here."
        ),
        **FALSE_AUTHORITY,
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = result_json.with_suffix(result_json.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, result_json)
    return payload


def harvest_once(
    paths: HarvestPaths,
    *,
    target_epoch: int,
    device: str = DEFAULT_DEVICE,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    allow_final_fallback: bool = True,
    arch: Mapping[str, Any] | None = None,
    b2_runner: Callable[[list[str]], int] | None = None,
    export_fn: Callable[..., ExportResult] | None = None,
    free_memory_fn: Callable[[], float | None] | None = None,
) -> dict[str, Any] | None:
    """Attempt one harvest. Returns the result payload, or None if not yet ready.

    Steps: select target checkpoint -> (memory gate) -> export backend-only
    archive -> run B2 -> write result + status. Idempotent at the caller level
    (the watch loop checks ``result_json`` first).
    """
    checkpoints = list_checkpoints(paths.checkpoint_dir)
    target = select_target_checkpoint(
        checkpoints, target_epoch=target_epoch, allow_final_fallback=allow_final_fallback
    )
    if target is None:
        return None

    # Memory gate (export is cheap; the B2 inflate is the ~3.6GB step).
    free_fn = free_memory_fn or free_memory_gb
    free_gb = free_fn()
    if free_gb is not None and free_gb < min_free_gb:
        write_status(
            paths.status_json,
            state="waiting_for_memory",
            target_epoch=target_epoch,
            detail={
                "free_gb": round(free_gb, 2),
                "min_free_gb": min_free_gb,
                "selected_checkpoint_epoch": target.global_epoch,
            },
        )
        return None

    write_status(
        paths.status_json,
        state="exporting",
        target_epoch=target_epoch,
        detail={"selected_checkpoint": target.to_jsonable(), "free_gb": free_gb},
    )

    exporter = export_fn or export_backend_only_archive_from_checkpoint
    export = exporter(target, paths.export_dir, arch=arch)

    write_status(
        paths.status_json,
        state="evaluating_b2",
        target_epoch=target_epoch,
        detail={
            "archive_path": str(export.archive_path),
            "archive_bytes": export.archive_bytes,
            "zip_members": export.zip_members,
            "member_magic": export.member_magic,
        },
    )

    b2_summary = run_b2_eval(
        export.archive_path,
        paths.result_json.with_name(paths.result_json.stem + "_b2row.json"),
        paths.work_root,
        device=device,
        runner=b2_runner,
    )

    payload = write_result(
        paths.result_json,
        target_epoch=target_epoch,
        checkpoint=target,
        export=export,
        b2_summary=b2_summary,
    )
    write_status(
        paths.status_json,
        state="complete",
        target_epoch=target_epoch,
        detail={
            "result_json": str(paths.result_json),
            "first_exact_score_advisory": payload.get("first_exact_score_advisory"),
            "b2_verdict": (b2_summary.get("b2_result") or {}).get("verdict"),
        },
    )
    return payload


def watch_loop(
    paths: HarvestPaths,
    *,
    target_epoch: int,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    heartbeat_stale_seconds: float = DEFAULT_HEARTBEAT_STALE_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    device: str = DEFAULT_DEVICE,
    allow_final_fallback: bool = True,
    arch: Mapping[str, Any] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    now_fn: Callable[[], float] | None = None,
    b2_runner: Callable[[list[str]], int] | None = None,
    export_fn: Callable[..., ExportResult] | None = None,
    free_memory_fn: Callable[[], float | None] | None = None,
) -> int:
    """Poll until the checkpoint lands, then harvest. Returns a process rc.

    rc=0  : harvest complete (or idempotent no-op because result already exists).
    rc=3  : fail-closed (heartbeat stale before checkpoint appeared, OR
            max_wait elapsed with no eligible checkpoint).
    rc=4  : harvest raised (export/B2 error) -> failure status written.
    """
    sleeper = sleep_fn or time.sleep
    clock = now_fn or time.monotonic

    # IDEMPOTENT: result already present -> nothing to do.
    if paths.result_json.is_file():
        write_status(
            paths.status_json,
            state="idempotent_result_exists",
            target_epoch=target_epoch,
            detail={"result_json": str(paths.result_json)},
        )
        return 0

    write_status(
        paths.status_json,
        state="watching",
        target_epoch=target_epoch,
        detail={
            "checkpoint_dir": str(paths.checkpoint_dir),
            "heartbeat_path": str(paths.heartbeat_path),
            "poll_seconds": poll_seconds,
        },
    )

    deadline = clock() + max_wait_seconds
    while True:
        # Re-check idempotency each tick (a sibling could have harvested).
        if paths.result_json.is_file():
            return 0

        checkpoints = list_checkpoints(paths.checkpoint_dir)
        target = select_target_checkpoint(
            checkpoints,
            target_epoch=target_epoch,
            allow_final_fallback=allow_final_fallback,
        )

        if target is not None:
            try:
                payload = harvest_once(
                    paths,
                    target_epoch=target_epoch,
                    device=device,
                    min_free_gb=min_free_gb,
                    allow_final_fallback=allow_final_fallback,
                    arch=arch,
                    b2_runner=b2_runner,
                    export_fn=export_fn,
                    free_memory_fn=free_memory_fn,
                )
            except Exception as exc:
                write_status(
                    paths.status_json,
                    state="failed_harvest_exception",
                    target_epoch=target_epoch,
                    detail={
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "selected_checkpoint_epoch": target.global_epoch,
                    },
                )
                return 4
            if payload is not None:
                return 0
            # payload None => memory gate held us off; fall through to sleep+retry.

        # FAIL-CLOSED: heartbeat stale (pilot likely died) before a checkpoint.
        # If a usable checkpoint exists (target is not None) we would already
        # have harvested above; reaching here with a stale heartbeat AND no
        # eligible checkpoint means the pilot died before producing one.
        age = heartbeat_age_seconds(paths.heartbeat_path, now=time.time())
        train_exited = heartbeat_reports_train_exit(paths.heartbeat_path)
        heartbeat_stale = age is not None and age > heartbeat_stale_seconds
        if heartbeat_stale and not train_exited and target is None:
            write_status(
                paths.status_json,
                state="failed_heartbeat_stale",
                target_epoch=target_epoch,
                detail={
                    "heartbeat_age_seconds": round(age, 1),
                    "heartbeat_stale_seconds": heartbeat_stale_seconds,
                    "checkpoints_present": len(checkpoints),
                    "note": (
                        "Pilot heartbeat went stale before an eligible "
                        "checkpoint appeared; refusing to wait indefinitely."
                    ),
                },
            )
            return 3

        # FAIL-CLOSED: max wait elapsed.
        if clock() >= deadline:
            write_status(
                paths.status_json,
                state="failed_max_wait_elapsed",
                target_epoch=target_epoch,
                detail={
                    "max_wait_seconds": max_wait_seconds,
                    "checkpoints_present": len(checkpoints),
                    "heartbeat_age_seconds": (round(age, 1) if age is not None else None),
                },
            )
            return 3

        sleeper(poll_seconds)


# ---------------------------------------------------------------------------
# Trajectory harvest: a SEQUENCE of checkpoints, one eval at a time.
# ---------------------------------------------------------------------------

TRAJECTORY_SUMMARY_SCHEMA = "b1_trajectory_harvest_summary.v1"


def wait_for_result_file(
    result_path: Path,
    heartbeat_path: Path,
    *,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    heartbeat_stale_seconds: float = DEFAULT_HEARTBEAT_STALE_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    sleep_fn: Callable[[float], None] | None = None,
    now_fn: Callable[[], float] | None = None,
) -> bool:
    """Block until ``result_path`` exists, then return True (or False on fail-close).

    Used to serialize a trajectory harvester BEHIND an in-flight single-target
    harvest (e.g. the ep250 watcher): two concurrent 600-pair CPU evals would
    contend on cores + ~3.6 GB inflated-frame RAM each, so the trajectory waits
    for the running harvest's result JSON before starting its first target.

    Fail-closed (returns False) if the pilot heartbeat goes stale before the
    result appears (the sibling harvest's run likely died) OR ``max_wait_seconds``
    elapses. A ``TRAIN_EXIT`` heartbeat does NOT count as stale (the run finished
    cleanly; its final harvest may still be writing the result).
    """
    sleeper = sleep_fn or time.sleep
    clock = now_fn or time.monotonic
    if result_path.is_file():
        return True
    deadline = clock() + max_wait_seconds
    while True:
        if result_path.is_file():
            return True
        age = heartbeat_age_seconds(heartbeat_path, now=time.time())
        train_exited = heartbeat_reports_train_exit(heartbeat_path)
        if age is not None and age > heartbeat_stale_seconds and not train_exited:
            return False
        if clock() >= deadline:
            return False
        sleeper(poll_seconds)


def _result_score(result_json: Path) -> float | None:
    if not result_json.is_file():
        return None
    try:
        payload = json.loads(result_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, Mapping):
        val = payload.get("first_exact_score_advisory")
        if isinstance(val, (int, float)):
            return float(val)
    return None


def watch_trajectory(
    run_dir: Path,
    targets: Sequence[int],
    *,
    heartbeat_path: Path | None = None,
    wait_for_result_path: Path | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    heartbeat_stale_seconds: float = DEFAULT_HEARTBEAT_STALE_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    device: str = DEFAULT_DEVICE,
    arch: Mapping[str, Any] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    now_fn: Callable[[], float] | None = None,
    b2_runner: Callable[[list[str]], int] | None = None,
    export_fn: Callable[..., ExportResult] | None = None,
    free_memory_fn: Callable[[], float | None] | None = None,
    watch_loop_fn: Callable[..., int] | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Harvest a TRAJECTORY of checkpoints (ep N1 < N2 < ...) ONE EVAL AT A TIME.

    The operator's burning question — "does the ep-250 exact-eval TREND justify
    continuing to ep-3000?" — needs >= 2 exact points. This harvests them
    sequentially so the trend is built without ever running concurrent 600-pair
    evals (a single process looping ``watch_loop`` per target physically cannot
    overlap evals).

    Per-target semantics:
      * reuses the single-target ``watch_loop`` (idempotent, memory-gated,
        fail-closed) for each target in ASCENDING order;
      * ``allow_final_fallback=False`` per target so each harvests its OWN
        checkpoint (no duplicate-final across targets);
      * STOPS early on a fail-closed target (rc=3 => run dead/stale => later
        targets unreachable);
      * CONTINUES past a per-target harvest exception (rc=4 => transient bridge
        error for that checkpoint; the next checkpoint may export fine).

    If ``wait_for_result_path`` is given, blocks until that result appears (an
    in-flight single-target harvest, e.g. ep250) BEFORE the first target, so the
    trajectory serializes behind the running harvest (no contention).
    """
    run_dir = run_dir.resolve()
    _refuse_tmp_path(run_dir, "run_dir")
    run_id = run_dir.name
    if heartbeat_path is None:
        heartbeat_path = REPO_ROOT / ".omx" / "tmp" / f"heartbeat_b1_{run_id}.log"
    looper = watch_loop_fn or watch_loop
    targets_sorted = sorted({int(t) for t in targets})
    per_target: list[dict[str, Any]] = []
    stopped_early = False
    stop_reason: str | None = None

    def _finalize() -> dict[str, Any]:
        summary: dict[str, Any] = {
            "schema": TRAJECTORY_SUMMARY_SCHEMA,
            "run_dir": str(run_dir),
            "targets": targets_sorted,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "per_target": per_target,
            "updated_utc": _utc_now(),
            "evidence_grade": "[macOS-CPU advisory]",
            "authoritative": False,
            **FALSE_AUTHORITY,
        }
        if summary_path is not None:
            _refuse_tmp_path(summary_path, "trajectory summary_path")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(tmp, summary_path)
        return summary

    if wait_for_result_path is not None:
        appeared = wait_for_result_file(
            wait_for_result_path,
            heartbeat_path,
            poll_seconds=poll_seconds,
            heartbeat_stale_seconds=heartbeat_stale_seconds,
            max_wait_seconds=max_wait_seconds,
            sleep_fn=sleep_fn,
            now_fn=now_fn,
        )
        per_target.append(
            {
                "prereq_result_path": str(wait_for_result_path),
                "prereq_appeared": appeared,
                "prereq_score": _result_score(wait_for_result_path),
            }
        )
        if not appeared:
            stopped_early = True
            stop_reason = "prereq_result_not_seen_run_dead_or_stale"
            return _finalize()

    for target in targets_sorted:
        paths_t = resolve_paths(
            run_dir, target_epoch=target, heartbeat_path=heartbeat_path
        )
        rc = looper(
            paths_t,
            target_epoch=target,
            poll_seconds=poll_seconds,
            heartbeat_stale_seconds=heartbeat_stale_seconds,
            max_wait_seconds=max_wait_seconds,
            min_free_gb=min_free_gb,
            device=device,
            allow_final_fallback=False,
            arch=arch,
            sleep_fn=sleep_fn,
            now_fn=now_fn,
            b2_runner=b2_runner,
            export_fn=export_fn,
            free_memory_fn=free_memory_fn,
        )
        per_target.append(
            {
                "target_epoch": target,
                "rc": int(rc),
                "result_json": str(paths_t.result_json),
                "first_exact_score_advisory": _result_score(paths_t.result_json),
            }
        )
        if rc == 3:
            stopped_early = True
            stop_reason = f"target_{target}_fail_closed_run_dead_or_stale"
            break

    return _finalize()


# ---------------------------------------------------------------------------
# Detached self-launch (canonical CLAUDE.md nohup pattern).
# ---------------------------------------------------------------------------


def build_detached_launch_command(child_argv: Sequence[str], log_path: Path) -> str:
    """Return the canonical nohup+bash -c+</dev/null+disown launch string."""
    _refuse_tmp_path(log_path, "detached log_path")
    inner = " ".join(shlex.quote(a) for a in child_argv)
    quoted_log = shlex.quote(str(log_path))
    outer_log = shlex.quote(str(log_path.with_suffix(log_path.suffix + ".outer")))
    # nohup + bash -c subshell so the pipe survives; </dev/null closes stdin;
    # 2>&1 | tee captures; disown removes from the job table.
    return (
        f"nohup bash -c {shlex.quote(inner + ' 2>&1 | tee ' + quoted_log + ' > /dev/null')} "
        f"< /dev/null > {outer_log} 2>&1 & disown"
    )


def launch_detached(child_argv: Sequence[str], log_path: Path) -> dict[str, Any]:
    """Launch the harvester detached so it survives SIGURG + session end."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    launch_cmd = build_detached_launch_command(child_argv, log_path)
    proc = subprocess.Popen(
        ["/bin/bash", "-c", launch_cmd],
        cwd=str(REPO_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {
        "launched": True,
        "launch_command": launch_cmd,
        "launcher_pid": proc.pid,
        "log_path": str(log_path),
        "launched_utc": _utc_now(),
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_arch_overrides(values: Sequence[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in values or ():
        if "=" not in item:
            raise SystemExit(f"--arch-override must be key=value; got {item!r}")
        key, _, raw = item.partition("=")
        out[key.strip()] = raw.strip()
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="B1 pilot run directory (SSD).")
    ap.add_argument("--target-epoch", type=int, default=DEFAULT_TARGET_EPOCH)
    ap.add_argument(
        "--target-epochs",
        default=None,
        help=(
            "Comma-list of checkpoint epochs to harvest as a TREND, sequentially "
            "(e.g. --target-epochs 500,750,1000,1250,1500). One eval at a time "
            "(no concurrent 600-pair evals). Overrides --target-epoch when given."
        ),
    )
    ap.add_argument(
        "--wait-for-result",
        default=None,
        help=(
            "Path to a result JSON to wait for BEFORE starting the trajectory "
            "(serializes behind an in-flight single-target harvest, e.g. the "
            "ep250 result, so the two never contend). Trajectory mode only."
        ),
    )
    ap.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "cuda"])
    ap.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    ap.add_argument(
        "--heartbeat-stale-seconds", type=float, default=DEFAULT_HEARTBEAT_STALE_SECONDS
    )
    ap.add_argument("--max-wait-seconds", type=float, default=DEFAULT_MAX_WAIT_SECONDS)
    ap.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    ap.add_argument(
        "--no-final-fallback",
        action="store_true",
        help="Do NOT fall back to the final checkpoint if no ep>=target exists.",
    )
    ap.add_argument(
        "--heartbeat-path",
        default=None,
        help="Override heartbeat path (default: .omx/tmp/heartbeat_b1_<run_id>.log).",
    )
    ap.add_argument(
        "--work-root",
        default=None,
        help="B2 scratch root (default: <run_dir>/harvest_b2_work_ep<N>). NEVER /tmp.",
    )
    ap.add_argument(
        "--arch-override",
        action="append",
        default=None,
        help="Override a pilot architecture arg, e.g. --arch-override num_pairs=8 (testing).",
    )
    ap.add_argument(
        "--launch-detached",
        action="store_true",
        help="Self-relaunch detached (nohup) and return immediately.",
    )
    ap.add_argument(
        "--print-launch-command",
        action="store_true",
        help="Print the canonical detached launch command and exit (no launch).",
    )
    return ap.parse_args(argv)


def _parse_target_epochs(raw: str | None) -> list[int]:
    """Parse ``--target-epochs`` comma-list into a sorted, de-duped int list."""
    if not raw:
        return []
    out: set[int] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.add(int(tok))
        except ValueError as exc:
            raise SystemExit(f"--target-epochs entry not an int: {tok!r}") from exc
    return sorted(out)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir).expanduser()
    arch = _parse_arch_overrides(args.arch_override)
    # Coerce known-numeric arch overrides.
    for k in ("num_pairs", "seed", "embed_dim", "latent_dim_coarse", "latent_dim_mid", "latent_dim_fine", "output_height", "output_width"):
        if k in arch:
            arch[k] = int(arch[k])

    trajectory_targets = _parse_target_epochs(args.target_epochs)
    heartbeat_path = Path(args.heartbeat_path) if args.heartbeat_path else None
    wait_for_result = Path(args.wait_for_result) if args.wait_for_result else None

    # Build the child argv (same as this invocation, minus the launch flags).
    child_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-dir",
        str(run_dir),
        "--device",
        args.device,
        "--poll-seconds",
        str(args.poll_seconds),
        "--heartbeat-stale-seconds",
        str(args.heartbeat_stale_seconds),
        "--max-wait-seconds",
        str(args.max_wait_seconds),
        "--min-free-gb",
        str(args.min_free_gb),
    ]
    if trajectory_targets:
        child_argv += ["--target-epochs", ",".join(str(t) for t in trajectory_targets)]
        if wait_for_result is not None:
            child_argv += ["--wait-for-result", str(wait_for_result)]
    else:
        child_argv += ["--target-epoch", str(args.target_epoch)]
    if args.no_final_fallback:
        child_argv.append("--no-final-fallback")
    if args.heartbeat_path:
        child_argv += ["--heartbeat-path", str(args.heartbeat_path)]
    if args.work_root:
        child_argv += ["--work-root", str(args.work_root)]
    for item in args.arch_override or ():
        child_argv += ["--arch-override", item]

    # The detached log: per-target for single mode, trajectory-wide otherwise.
    if trajectory_targets:
        detached_log = run_dir.resolve() / "harvest_trajectory.log"
    else:
        detached_log = resolve_paths(
            run_dir, target_epoch=args.target_epoch, heartbeat_path=heartbeat_path
        ).harvester_log

    if args.print_launch_command:
        print(build_detached_launch_command(child_argv, detached_log))
        return 0

    if args.launch_detached:
        info = launch_detached(child_argv, detached_log)
        print(json.dumps(info, indent=2, sort_keys=True))
        return 0

    if trajectory_targets:
        summary = watch_trajectory(
            run_dir,
            trajectory_targets,
            heartbeat_path=heartbeat_path,
            wait_for_result_path=wait_for_result,
            poll_seconds=args.poll_seconds,
            heartbeat_stale_seconds=args.heartbeat_stale_seconds,
            max_wait_seconds=args.max_wait_seconds,
            min_free_gb=args.min_free_gb,
            device=args.device,
            arch=arch or None,
            summary_path=run_dir.resolve() / "harvest_trajectory_summary.json",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        # rc=0 unless the trajectory stopped early without harvesting anything.
        harvested = any(
            row.get("rc") == 0 for row in summary["per_target"] if "rc" in row
        )
        return 0 if harvested or not summary["stopped_early"] else 3

    paths = resolve_paths(
        run_dir,
        target_epoch=args.target_epoch,
        heartbeat_path=heartbeat_path,
        work_root=(Path(args.work_root) if args.work_root else None),
    )
    return watch_loop(
        paths,
        target_epoch=args.target_epoch,
        poll_seconds=args.poll_seconds,
        heartbeat_stale_seconds=args.heartbeat_stale_seconds,
        max_wait_seconds=args.max_wait_seconds,
        min_free_gb=args.min_free_gb,
        device=args.device,
        allow_final_fallback=not args.no_final_fallback,
        arch=arch or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
