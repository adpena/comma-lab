#!/usr/bin/env python3
"""Governed admission for CONCURRENT local training cells (memory headroom + Metal contention).

WHY (operator 2026-09-04, verbatim: "Remember we can fully saturate cpu, gpu, and ane; we can
continue polishing the memory guard and governor and costate organ and controller as well"):
the one-Metal-fire rule was GOVERNANCE, not a hardware limit.  A second or third cell is admitted
by MEASURED headroom, never by a fixed count.  Two things must both hold:

  1. MEMORY.  Reclaimable-without-swap RAM must cover the candidate's projected peak PLUS the
     UNREALIZED growth every live cell still owes toward its own declared peak, plus a margin.
  2. THROUGHPUT.  Adding a cell must not make the machine slower.  Total steps/min under
     concurrency must stay at or above the measured serial baseline; otherwise the Metal is
     contended and the extra cell buys nothing.

THE BUG THIS CURES (MEASURED 2026-09-04).  MAIN's hand-written fire scripts computed admission
inline in shell::

    FREE=$(vm_stat | awk '/Pages free/…'); INACT=$(vm_stat | awk '/Pages inactive/…')
    RECLAIM=$((FREE+INACT)); [ "$RECLAIM" -ge $((PEAK+16)) ] || exit 2

That is ``psutil.virtual_memory().available`` recomputed by hand -- the reclaimable-BLIND basis
that ``src/tac/confound_gates.py:2336`` (``check_no_raw_psutil_memory_basis_outside_canonical_helper``,
STRICT since 2026-07-31) refuses in Python.  The shell spelling escaped the gate because the gate
scans Python.  Measured live on this box with both cells running, the three bases disagreed::

    shell free+inactive      13 GiB
    mem_basis canonical      18.12 GiB     <- the governor's queue decomposition
    psutil .available        22.53 GiB

Three bases, three numbers.  This module routes every decision through the ONE canonical helper
(``tools/mem_basis.py``) and additionally charges each live cell its unrealized growth, which the
shell rule ignored entirely (measured: ng3 declared a 41.5 GiB peak while holding 0.58 GiB, so
40.9 GiB of committed-but-unrealized growth was invisible to the shell rule).

Both legs FAIL CLOSED: a measurement that cannot be taken REFUSES rather than admitting blind.

Read-only with respect to live cells: this module only reads their launch manifests, their
authorized configs, their ``history.jsonl`` row counts, and ``/proc``-equivalent process state.
It never signals, edits, or claims anything belonging to a live run.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import mem_basis  # noqa: E402  (tools/ is on sys.path above)

ADMISSION_SCHEMA = "ddm_gv1_cell_admission_decision.v1"
LIVE_CELL_SCHEMA = "ddm_gv1_live_cell.v1"
THROUGHPUT_ROW_SCHEMA = "ddm_gv1_metal_contention_row.v1"

CONTENTION_LEDGER = _REPO / ".omx" / "state" / "metal_contention_ledger.jsonl"

#: Spike / model-error margin held back on top of every projection.  Matches the margin MAIN used
#: by hand in the ng3 fire script (``PEAK + 16``) so the apparatus reproduces the operator-blessed
#: decision rather than silently loosening it.
DEFAULT_MARGIN_GIB = 16.0

#: Absolute committed ceiling, mirroring ``tools/witness_memory_preflight.OPERATOR_CEILING_GIB_FALLBACK``
#: (operator ceiling policy 2026-07-21).  Read from the governor's knob when it is set.
OPERATOR_CEILING_ENV = "TAC_GOV_OPERATOR_CEILING_GIB"
OPERATOR_CEILING_GIB_FALLBACK = 116.0

#: Where launch manifests for local cells live.  Roots only -- the cell name is never encoded.
DEFAULT_MANIFEST_ROOTS: tuple[Path, ...] = (
    Path("/Volumes/APDataStore/pact"),
    Path("/Volumes/VertigoDataTier/pact"),
)

#: Depth of the manifest walk under each root.  The observed layout is
#: ``<root>/<campaign>/<cell>/launch/<run>/launch_manifest.json`` (depth 5).
_MANIFEST_MAX_DEPTH = 6

#: Registry of launcher-written manifests (``detached_launch_registry.v1`` rows appended by
#: ``tools/launch_detached_process.py`` at launch, and by the one-time seed walk).  Reading it
#: is O(launches); the SSD walk was MEASURED at >120 s per admission poll on 2026-09-04 (two
#: 1.8 TB roots at depth 6), which stalled the ng4 fire waiter for an hour after the Metal freed.
LAUNCH_REGISTRY = _REPO / ".omx" / "state" / "detached_launch_registry.jsonl"

#: Bulk directories the fallback walk must not descend into: they hold frames, payloads,
#: checkpoints and sealed source trees, never a ``launch_manifest.json``.
_WALK_PRUNE_NAMES = frozenset(
    {
        "retained",
        "runs",
        "milestones",
        "checkpoints",
        "frames",
        "inflated",
        "extracted",
        "eval_work",
        "cold_store",
        "harvested_artifacts",
        "payload",
        "payloads",
        "source",
        "repo",
        "upstream",
        "node_modules",
        ".git",
        ".venv",
    }
)
_WALK_PRUNE_PREFIXES = ("sealed_source_", "stage_", "step_", "shard_")

_GIB = 1024.0**3

# Exit codes (the fire scripts already branch on ``exit 2``).
RC_ADMIT = 0
RC_REFUSE = 2
RC_UNMEASURABLE = 3


def utc_text(value: dt.datetime | None = None) -> str:
    return (value or dt.datetime.now(dt.UTC)).astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def operator_ceiling_gib() -> float:
    """The absolute committed ceiling in GiB (governor knob, else the policy fallback)."""
    raw = os.environ.get(OPERATOR_CEILING_ENV)
    if raw is None:
        return OPERATOR_CEILING_GIB_FALLBACK
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return OPERATOR_CEILING_GIB_FALLBACK
    return value if value > 0.0 else OPERATOR_CEILING_GIB_FALLBACK


def pid_alive(pid: int) -> bool:
    """True when ``pid`` names a live process. Mirrors ``ddm_qbr1_cell_chain.pid_alive``."""
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def process_tree_rss_gib(pid: int) -> float | None:
    """Resident set size of ``pid`` AND all its descendants, in GiB; None when unreadable.

    The tree, not the single process: the canonical launcher detaches as
    ``launch_detached_process _supervise -> safe_run -> trainer``, so the manifest PID is a
    supervisor holding almost nothing while the real allocation sits two levels down.  MEASURED
    2026-09-04 on the live ng3 cell -- supervisor 51498 alone read 0.023 GiB while the tree
    (51499 + trainer 51510) read 0.608 GiB, a 26x under-read that would have made every live cell
    look free and inflated its unrealized growth.

    This is a PER-PROCESS reading, not a system memory basis, so the CLASS-1 raw-psutil gate
    (``src/tac/confound_gates.py:2336``) does not apply -- that gate scopes
    ``psutil.virtual_memory()``.  The system basis in this module comes from ``mem_basis``.
    """
    try:
        import psutil  # type: ignore

        parent = psutil.Process(pid)
        total = float(parent.memory_info().rss)
        for child in parent.children(recursive=True):
            try:
                total += float(child.memory_info().rss)
            except Exception:
                continue
        return total / _GIB
    except Exception:
        return None


# ── live-cell discovery ─────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class LiveCell:
    """One live governed local job, read from its launch manifest and (for cells) its config.

    Discovery is deliberately BROADER than "training cell": every live job launched through the
    canonical launcher holds real memory and must be charged by the admission arithmetic.  The
    ``is_cell`` flag marks the subset that are training cells (they carry a ``run-config`` with a
    ``total_steps`` budget and a ``history.jsonl``), which is what the queue driver and the digest
    treat as cells.  Charging only "cells" would under-protect against a 12 GiB non-cell job.
    """

    cell_id: str
    pid: int
    alive: bool
    declared_peak_gib: float
    current_rss_gib: float | None
    manifest_path: Path
    config_path: Path | None
    run_dir: Path | None
    total_steps: int | None
    completed_steps: int | None
    arm_name: str | None
    arm_role: str | None
    purpose: str | None
    # (ddm_gov2) provenance of the row itself.  ``discovery_source`` says WHERE the job was found
    # -- ``process_table`` is the complete-by-construction source, ``registry`` / ``walk`` are the
    # optional cross-checks.  ``declared_peak_source`` says which declaration the arithmetic
    # charged, because a manifest and a live ``safe_run --projected-gib`` can disagree and the
    # fail-closed answer is the LARGER of the two.
    discovery_source: str = "manifest"
    declared_peak_source: str = "manifest"
    safe_run_pid: int | None = None
    trainer_pid: int | None = None

    @property
    def is_cell(self) -> bool:
        """True when this job is a training cell (a sealed run-config with a step budget)."""
        return self.config_path is not None and bool(self.total_steps)

    @property
    def unrealized_growth_gib(self) -> float:
        """Memory the cell has declared but not yet allocated -- the growth a new cell must respect.

        Fail-closed: when the current RSS is unreadable the whole declared peak is charged.
        """
        if self.current_rss_gib is None:
            return max(0.0, self.declared_peak_gib)
        return max(0.0, self.declared_peak_gib - self.current_rss_gib)

    @property
    def progress_fraction(self) -> float | None:
        if not self.total_steps or self.completed_steps is None or self.total_steps <= 0:
            return None
        return min(1.0, self.completed_steps / float(self.total_steps))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LIVE_CELL_SCHEMA,
            "cell_id": self.cell_id,
            "is_cell": self.is_cell,
            "pid": self.pid,
            "alive": self.alive,
            "declared_peak_gib": round(self.declared_peak_gib, 4),
            "current_rss_gib": (None if self.current_rss_gib is None else round(self.current_rss_gib, 4)),
            "unrealized_growth_gib": round(self.unrealized_growth_gib, 4),
            "manifest_path": str(self.manifest_path),
            "config_path": None if self.config_path is None else str(self.config_path),
            "run_dir": None if self.run_dir is None else str(self.run_dir),
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "progress_fraction": (None if self.progress_fraction is None else round(self.progress_fraction, 6)),
            "arm_name": self.arm_name,
            "arm_role": self.arm_role,
            "purpose": self.purpose,
            "discovery_source": self.discovery_source,
            "declared_peak_source": self.declared_peak_source,
            "safe_run_pid": self.safe_run_pid,
            "trainer_pid": self.trainer_pid,
        }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _config_path_from_argv(argv: Sequence[Any]) -> Path | None:
    """The trailing ``run-config <path.json>`` argument of a cell launch, when present."""
    parts = [str(item) for item in argv]
    for index, item in enumerate(parts):
        if item == "run-config" and index + 1 < len(parts):
            return Path(parts[index + 1])
    # Fall back to a trailing .json argument (the cell contract puts the config last).
    if parts and parts[-1].endswith(".json"):
        return Path(parts[-1])
    return None


def count_history_steps(run_dir: Path | None) -> int | None:
    """Rows in ``<run_dir>/history.jsonl`` -- one row per completed step for the cell trainer."""
    if run_dir is None:
        return None
    history = run_dir / "history.jsonl"
    if not history.is_file():
        return None
    try:
        with history.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def live_cell_from_manifest(manifest_path: Path) -> LiveCell | None:
    """Build a :class:`LiveCell` from a launcher manifest, or None when it is not a live cell."""
    manifest = _read_json(manifest_path)
    if manifest is None or not str(manifest.get("schema", "")).startswith("detached_local_process_launch"):
        return None
    try:
        pid = int(manifest.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    alive = pid_alive(pid)
    if not alive:
        return None

    budget = manifest.get("resource_budget")
    budget = budget if isinstance(budget, Mapping) else {}
    try:
        declared_peak = float(budget.get("measured_peak_rss_gib") or 0.0)
    except (TypeError, ValueError):
        declared_peak = 0.0

    argv = manifest.get("argv")
    config_path = _config_path_from_argv(argv) if isinstance(argv, Sequence) else None
    config = _read_json(config_path) if config_path is not None else None

    run_dir: Path | None = None
    total_steps: int | None = None
    cell_id = manifest_path.parent.name
    arm_name = arm_role = None
    if config is not None:
        output = config.get("output")
        if isinstance(output, str) and output:
            run_dir = Path(output)
        try:
            total_steps = int(config["total_steps"]) if "total_steps" in config else None
        except (TypeError, ValueError):
            total_steps = None
        cell_id = str(config.get("cell_id") or cell_id)
        arm_name = config.get("arm_name")
        arm_role = config.get("arm_role")

    return LiveCell(
        cell_id=cell_id,
        pid=pid,
        alive=alive,
        declared_peak_gib=declared_peak,
        current_rss_gib=process_tree_rss_gib(pid),
        manifest_path=manifest_path,
        config_path=config_path,
        run_dir=run_dir,
        total_steps=total_steps,
        completed_steps=count_history_steps(run_dir),
        arm_name=None if arm_name is None else str(arm_name),
        arm_role=None if arm_role is None else str(arm_role),
        purpose=(str(manifest["purpose"]) if isinstance(manifest.get("purpose"), str) else None),
    )


def _walk_manifests(root: Path, max_depth: int) -> Iterable[Path]:
    """Yield ``launch_manifest.json`` paths under ``root`` without descending forever."""
    if not root.is_dir():
        return
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            name = entry.name
            if name.startswith("."):
                continue
            try:
                if entry.is_file() and name == "launch_manifest.json":
                    yield Path(entry.path)
                elif (
                    entry.is_dir()
                    and depth < max_depth
                    and name not in _WALK_PRUNE_NAMES
                    and not name.startswith(_WALK_PRUNE_PREFIXES)
                ):
                    stack.append((Path(entry.path), depth + 1))
            except OSError:
                continue


def registry_manifest_paths(registry_path: Path | None = None) -> list[Path]:
    """Manifest paths recorded in the launch registry (deduplicated, existing files only)."""
    path = LAUNCH_REGISTRY if registry_path is None else Path(registry_path)
    if not path.is_file():
        return []
    seen: dict[str, Path] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        manifest = row.get("manifest_path") if isinstance(row, dict) else None
        if isinstance(manifest, str) and manifest and manifest not in seen:
            candidate = Path(manifest)
            if candidate.is_file():
                seen[manifest] = candidate
    return list(seen.values())


# ── process-table discovery (ddm_gov2) ──────────────────────────────────────────────────────────
#
# THE LAW.  A governed job is a LIVE PROCESS, and its argv already carries its whole identity.  The
# process table is therefore COMPLETE BY CONSTRUCTION: a job that is running is in it, and a job
# that is not running cannot hold memory.  A registry can be forgotten (MEASURED 2026-09-04: the
# ng4 cell fired from a SEALED-SOURCE copy of the launcher that predates the registry, so a live
# 45 GiB cell was invisible until MAIN backfilled the row by hand) and a filesystem walk can be too
# slow to run (MEASURED: >120 s per admission poll over the two 1.8 TB SSD roots).  Neither failure
# mode exists here.
#
# The argv shape is the OLD launcher's too, which is why this works for sealed-source copies --
# MEASURED on the live ng4 cell, 2026-09-04T22:57Z:
#
#   33030  ppid 1      … launch_detached_process.py _supervise --start-gate <out>/.launch_start_gate
#                        --manifest <out>/launch_manifest.json … -- <safe_run argv>
#   33039  ppid 33030  … safe_run.py --rss-mb 118784 --projected-gib 45.0 --status-receipt <out>/…
#                        -- <trainer argv>
#   33374  ppid 33039  … ddm_qbr1_born_fairform_burn_prep.py run-config <authorized_config.json>
#
# supervisor -> output dir + manifest;  safe_run -> the DECLARED PEAK;  trainer -> the sealed config
# and its step budget.  Three facts, one `ps`, no disk walk, no registry.

_SUPERVISOR_SCRIPT = "launch_detached_process.py"
_SUPERVISOR_TOKEN = "_supervise"
_SAFE_RUN_SCRIPT = "safe_run.py"
_RUN_CONFIG_TOKEN = "run-config"


@dataclasses.dataclass(frozen=True)
class ProcessRow:
    """One row of the process table: pid, parent pid, and the real argv vector."""

    pid: int
    ppid: int
    argv: tuple[str, ...]

    def has_script(self, name: str) -> bool:
        return any(part.endswith(name) for part in self.argv)


def _argv_option(argv: Sequence[str], flag: str) -> str | None:
    """Value of ``--flag VALUE`` or ``--flag=VALUE`` in ``argv``; None when absent."""
    prefix = f"{flag}="
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def _argv_float(argv: Sequence[str], flag: str) -> float | None:
    raw = _argv_option(argv, flag)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def read_process_table() -> list[ProcessRow]:
    """Every visible process as (pid, ppid, argv).

    psutil first because it returns the REAL argv vector; ``ps`` is a fallback that joins argv with
    spaces and therefore cannot represent an argument containing a space.  The fallback is honest
    about that limitation rather than silently mis-splitting: it is only reached when psutil is
    unavailable, and the paths this repo launches from contain no spaces (MEASURED on the live
    fleet).  Both paths swallow per-process permission errors -- another user's process cannot be
    read, and a governed job of ours always can.
    """
    try:
        import psutil  # type: ignore

        rows: list[ProcessRow] = []
        for proc in psutil.process_iter(["pid", "ppid", "cmdline"]):
            try:
                info = proc.info
                argv = info.get("cmdline") or []
                if not argv:
                    continue
                rows.append(
                    ProcessRow(
                        pid=int(info["pid"]),
                        ppid=int(info.get("ppid") or 0),
                        argv=tuple(str(part) for part in argv),
                    )
                )
            except Exception:
                continue
        if rows:
            return rows
    except Exception:
        pass
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,command="],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        rows.append(ProcessRow(pid=pid, ppid=ppid, argv=tuple(parts[2:])))
    return rows


def _is_supervisor(row: ProcessRow) -> bool:
    return row.has_script(_SUPERVISOR_SCRIPT) and _SUPERVISOR_TOKEN in row.argv


def _is_safe_run(row: ProcessRow) -> bool:
    return row.has_script(_SAFE_RUN_SCRIPT)


def _is_trainer(row: ProcessRow) -> bool:
    return _RUN_CONFIG_TOKEN in row.argv


def live_cells_from_process_table(rows: Sequence[ProcessRow] | None = None) -> list[LiveCell]:
    """Every live governed job, read from the process table alone.

    A governed job is rooted at the OUTERMOST of {supervisor, safe_run, trainer} in one process
    lineage, so a supervisor + its safe_run + its trainer collapse to ONE cell (rooted at the
    supervisor pid, which is exactly what the launcher records as the manifest ``pid`` -- the two
    discovery paths therefore agree on the key).  A bare ``safe_run`` with no supervisor (the
    dashboard server is one, MEASURED) and a bare trainer are each their own root.
    """
    table = list(read_process_table()) if rows is None else list(rows)
    by_pid = {row.pid: row for row in table}
    governed = {row.pid: row for row in table if _is_supervisor(row) or _is_safe_run(row) or _is_trainer(row)}

    def _root_of(pid: int) -> int:
        """Outermost governed ancestor of ``pid`` (itself when it has none)."""
        root = pid
        seen = {pid}
        cursor = by_pid.get(pid)
        while cursor is not None and cursor.ppid > 1 and cursor.ppid not in seen:
            seen.add(cursor.ppid)
            if cursor.ppid in governed:
                root = cursor.ppid
            cursor = by_pid.get(cursor.ppid)
        return root

    # group every governed process under its outermost governed ancestor
    groups: dict[int, list[ProcessRow]] = {}
    for pid, row in governed.items():
        groups.setdefault(_root_of(pid), []).append(row)

    cells: list[LiveCell] = []
    for root_pid, members in groups.items():
        root = by_pid.get(root_pid)
        if root is None:
            continue
        supervisor = next((m for m in members if _is_supervisor(m)), None)
        # NOT `_is_safe_run(m)` alone: the supervisor's own argv CONTAINS the whole safe_run argv
        # after its `--`, so the naive predicate matches the supervisor and reports its pid as the
        # safe_run pid (MEASURED on live ng4 -- it returned 33030 where the real safe_run is 33039).
        safe_run = next((m for m in members if _is_safe_run(m) and not _is_supervisor(m)), None)
        trainer = next((m for m in members if _is_trainer(m) and not _is_safe_run(m) and not _is_supervisor(m)), None)

        manifest_path: Path | None = None
        if supervisor is not None:
            raw = _argv_option(supervisor.argv, "--manifest")
            if raw:
                manifest_path = Path(raw)
        if manifest_path is None and safe_run is not None:
            raw = _argv_option(safe_run.argv, "--status-receipt")
            if raw:
                sibling = Path(raw).parent / "launch_manifest.json"
                if sibling.is_file():
                    manifest_path = sibling
        manifest = _read_json(manifest_path) if manifest_path is not None else None

        # DECLARED PEAK, fail-closed: the larger of the manifest budget and the live safe_run
        # projection.  They can disagree (a hand-edited manifest, an older launcher), and charging
        # the smaller one is exactly the under-reservation that admitted two ~40 GiB Metal cells.
        manifest_peak = 0.0
        if isinstance(manifest, dict):
            budget = manifest.get("resource_budget")
            if isinstance(budget, Mapping):
                try:
                    manifest_peak = float(budget.get("measured_peak_rss_gib") or 0.0)
                except (TypeError, ValueError):
                    manifest_peak = 0.0
        # The supervisor's argv carries the whole wrapped safe_run command after its `--`, so the
        # projection is still readable when the safe_run process itself is not visible (it has not
        # spawned yet, or it already exited while the supervisor lingers).
        projected = None
        for source in (safe_run, supervisor):
            if source is None:
                continue
            projected = _argv_float(source.argv, "--projected-gib")
            if projected is not None:
                break
        rss_cap_gib = None
        if safe_run is not None or supervisor is not None:
            rss_mb = _argv_float((safe_run or supervisor).argv, "--rss-mb")
            # the cap is the operator ceiling, not a projection -- only a last-resort declaration
            rss_cap_gib = None if rss_mb is None else rss_mb / 1024.0
        candidates: list[tuple[float, str]] = []
        if manifest_peak > 0.0:
            candidates.append((manifest_peak, "manifest_resource_budget"))
        if projected is not None and projected > 0.0:
            candidates.append((projected, "safe_run_projected_gib"))
        if candidates:
            declared_peak, peak_source = max(candidates, key=lambda item: item[0])
        elif rss_cap_gib:
            declared_peak, peak_source = rss_cap_gib, "safe_run_rss_cap"
        else:
            declared_peak, peak_source = 0.0, "UNDECLARED"

        config_path = None
        for member in (trainer, safe_run, supervisor):
            if member is None:
                continue
            config_path = _config_path_from_argv(member.argv)
            if config_path is not None:
                break
        config = _read_json(config_path) if config_path is not None else None

        run_dir: Path | None = None
        total_steps: int | None = None
        arm_name = arm_role = None
        cell_id = manifest_path.parent.name if manifest_path is not None else f"pid_{root_pid}"
        if config is not None:
            output = config.get("output")
            if isinstance(output, str) and output:
                run_dir = Path(output)
            try:
                total_steps = int(config["total_steps"]) if "total_steps" in config else None
            except (TypeError, ValueError):
                total_steps = None
            cell_id = str(config.get("cell_id") or cell_id)
            arm_name = config.get("arm_name")
            arm_role = config.get("arm_role")

        purpose = None
        if isinstance(manifest, dict) and isinstance(manifest.get("purpose"), str):
            purpose = str(manifest["purpose"])

        current_rss = process_tree_rss_gib(root_pid)
        # An UNDECLARED job still holds memory: charge what it is measurably using so it can never
        # read as free.  ``unrealized_growth`` then contributes nothing extra, which is correct --
        # there is no declaration to grow into.
        if peak_source == "UNDECLARED" and current_rss is not None:
            declared_peak = current_rss

        cells.append(
            LiveCell(
                cell_id=cell_id,
                pid=root_pid,
                alive=True,
                declared_peak_gib=declared_peak,
                current_rss_gib=current_rss,
                manifest_path=manifest_path if manifest_path is not None else Path(f"<process:{root_pid}>"),
                config_path=config_path,
                run_dir=run_dir,
                total_steps=total_steps,
                completed_steps=count_history_steps(run_dir),
                arm_name=None if arm_name is None else str(arm_name),
                arm_role=None if arm_role is None else str(arm_role),
                purpose=purpose,
                discovery_source="process_table",
                declared_peak_source=peak_source,
                safe_run_pid=None if safe_run is None else safe_run.pid,
                trainer_pid=None if trainer is None else trainer.pid,
            )
        )
    return sorted(cells, key=lambda cell: (cell.cell_id, cell.pid))


def registry_cross_check(
    cells: Sequence[LiveCell],
    *,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Compare process-table discovery against the launch registry (OPTIONAL, never authoritative).

    The registry can only ever be a SUBSET of the truth (a launcher generation that does not write
    it, a fail-open append that lost a row), so a registry row the process table missed is the only
    interesting direction -- and it means the ps scan has a blind spot worth naming.  Reported,
    never acted on: the arithmetic charges the process table.
    """
    seen = {cell.pid for cell in cells}
    missed: list[dict[str, Any]] = []
    registry_live = 0
    for manifest_path in registry_manifest_paths(registry_path):
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict):
            continue
        try:
            pid = int(manifest.get("pid") or 0)
        except (TypeError, ValueError):
            continue
        if not pid_alive(pid):
            continue
        registry_live += 1
        if pid not in seen:
            missed.append({"pid": pid, "manifest_path": str(manifest_path)})
    return {
        "registry_live_rows": registry_live,
        "process_table_cells": len(cells),
        "missed_by_process_table": missed,
        "agrees": not missed,
    }


def discover_live_cells(
    roots: Sequence[Path] | None = None,
    *,
    max_depth: int = _MANIFEST_MAX_DEPTH,
    registry_path: Path | None = None,
    walk_roots: bool | None = None,
    process_table: bool = True,
) -> list[LiveCell]:
    """Live governed jobs, PROCESS-TABLE first (ddm_gov2).

    The default (``roots=None``, ``walk_roots`` falsy) reads the process table and nothing else --
    complete by construction, MEASURED < 1 s, and correct for sealed-source launcher copies that
    never write the registry.

    ``walk_roots=True`` (or an explicit ``roots``) additionally performs the pruned SSD walk as an
    explicit FORENSIC mode: it can only find manifests for pids the process table already covers,
    so it exists to answer "what manifests are on disk", not "what is running".  It is slow
    (MEASURED >120 s over both SSD roots) and is never the default again.  The walk is UNIONED with
    the process table, never substituted for it -- an admission decision must never lose a live job
    because the caller asked a narrower question.  ``process_table=False`` is the one way to get the
    disk view alone, for forensic queries and for unit tests that must not see the real fleet.
    """
    cells = live_cells_from_process_table() if process_table else []
    if not (walk_roots is True or roots is not None):
        return cells

    found: dict[int, LiveCell] = {cell.pid: cell for cell in cells}
    for root in roots if roots is not None else DEFAULT_MANIFEST_ROOTS:
        for manifest_path in _walk_manifests(Path(root), max_depth):
            walked = live_cell_from_manifest(manifest_path)
            if walked is None or walked.pid in found:
                continue
            found[walked.pid] = dataclasses.replace(walked, discovery_source="walk")
    return sorted(found.values(), key=lambda cell: (cell.cell_id, cell.pid))


# ── memory arithmetic ───────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class MemoryVerdict:
    """The reclaimable-aware memory leg of an admission decision, with its full arithmetic."""

    admits: bool
    reclaimable_gib: float
    committed_gib: float
    candidate_peak_gib: float
    live_unrealized_gib: float
    margin_gib: float
    required_gib: float
    headroom_gib: float
    ceiling_gib: float
    ceiling_headroom_gib: float
    basis: str
    measurable: bool
    naive_shell_reclaimable_gib: float | None
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = dict(dataclasses.asdict(self))
        payload["reasons"] = list(self.reasons)
        for key, value in list(payload.items()):
            if isinstance(value, float):
                payload[key] = round(value, 4)
        return payload


def naive_shell_reclaimable_gib() -> float | None:
    """The fire scripts' ``vm_stat`` free+inactive figure -- reported ONLY for contrast.

    This is the reclaimable-BLIND basis (dirty anonymous pages parked in the inactive queue count
    as available even though evicting them needs swap).  It is never used for a decision; the
    admission arithmetic uses :mod:`mem_basis`.  Surfacing it makes the disagreement visible
    instead of leaving two validators to differ silently.
    """
    try:
        import subprocess

        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10, check=False).stdout
    except Exception:
        return None
    page_size = 4096
    free_pages = inactive_pages = None
    for line in out.splitlines():
        if "page size of" in line:
            for token in line.replace(".", " ").split():
                if token.isdigit():
                    page_size = int(token)
                    break
        elif line.startswith("Pages free:"):
            free_pages = int(line.split(":")[1].strip().rstrip("."))
        elif line.startswith("Pages inactive:"):
            inactive_pages = int(line.split(":")[1].strip().rstrip("."))
    if free_pages is None or inactive_pages is None:
        return None
    return (free_pages + inactive_pages) * page_size / _GIB


def memory_verdict(
    candidate_peak_gib: float,
    live_cells: Sequence[LiveCell],
    *,
    margin_gib: float = DEFAULT_MARGIN_GIB,
    ceiling_gib: float | None = None,
    include_naive_contrast: bool = True,
) -> MemoryVerdict:
    """Decide the MEMORY leg for a candidate cell against the live fleet.

    Two constraints, both must hold:

    * RELATIVE headroom -- ``reclaimable >= candidate_peak + sum(live unrealized growth) + margin``.
      The live term is what the hand-written shell rule omitted: a cell that declared a 41.5 GiB
      peak while holding 0.58 GiB still owes 40.9 GiB, and a second cell admitted against the
      raw reclaimable figure would be racing that growth.
    * ABSOLUTE ceiling -- ``true_committed + candidate_peak + live unrealized <= operator ceiling``.

    Fail-closed: an unmeasurable basis yields ``admits=False`` with ``measurable=False``.

    WHY THE LIVE TERM IS *UNREALIZED GROWTH* AND NOT *RESIDENT FOOTPRINT* -- read before "fixing"
    this.  The obvious-looking cure for the shell guard is "subtract every live cell's resident
    footprint", and on the SHELL basis (``free + ALL inactive``) that is right: a running cell's
    dirty anonymous pages sit on the inactive queue and the shell guard counts them as headroom.
    On the CANONICAL basis it would DOUBLE-COUNT.  MEASURED on this box 2026-09-04:

        available_reclaimable = free 3.166 + file_backed 13.670 + purgeable 1.170 = 18.006 GiB

    Anonymous memory -- which is what a live cell's resident footprint IS -- never enters that sum;
    it is already on the committed side.  Subtracting resident footprint again would refuse launches
    the box could actually take.  What the canonical basis genuinely does NOT know is the memory a
    live cell has DECLARED but not yet allocated, so that -- and only that -- is charged here.
    (Sister finding: ``.omx/research/ddm_bh1_fresh_eyes_bug_hunt_20260904.md`` §6, which reached the
    same defect from the shell side and measured the over-trust at 1.204x idle / 4.2x under load.)
    """
    ceiling = operator_ceiling_gib() if ceiling_gib is None else float(ceiling_gib)
    candidate = max(0.0, float(candidate_peak_gib))
    margin = max(0.0, float(margin_gib))

    # ``float('nan')`` sentinel default => the measurement was unavailable (fail-closed).
    reclaimable = mem_basis.conservative_free_gib(default=float("nan"))
    committed = mem_basis.true_committed_gib(default=float("nan"))
    measurable = reclaimable == reclaimable and committed == committed  # NaN-safe

    live_unrealized = sum(cell.unrealized_growth_gib for cell in live_cells)
    required = candidate + live_unrealized + margin
    headroom = (reclaimable - required) if measurable else float("-inf")
    ceiling_headroom = (ceiling - (committed + candidate + live_unrealized)) if measurable else float("-inf")

    reasons: list[str] = []
    if not measurable:
        reasons.append("memory basis unavailable (mem_basis returned no measurement) -- fail-closed REFUSE")
    else:
        if headroom < 0.0:
            reasons.append(
                f"reclaimable {reclaimable:.2f} GiB < required {required:.2f} GiB "
                f"(candidate {candidate:.2f} + live-unrealized {live_unrealized:.2f} "
                f"+ margin {margin:.2f})"
            )
        if ceiling_headroom < 0.0:
            reasons.append(
                f"committed {committed:.2f} + candidate {candidate:.2f} "
                f"+ live-unrealized {live_unrealized:.2f} GiB exceeds the operator ceiling "
                f"{ceiling:.2f} GiB"
            )
    admits = bool(measurable and headroom >= 0.0 and ceiling_headroom >= 0.0)
    if admits:
        reasons.append(
            f"reclaimable {reclaimable:.2f} GiB covers required {required:.2f} GiB "
            f"(headroom {headroom:.2f} GiB); ceiling headroom {ceiling_headroom:.2f} GiB"
        )

    return MemoryVerdict(
        admits=admits,
        reclaimable_gib=reclaimable if measurable else 0.0,
        committed_gib=committed if measurable else 0.0,
        candidate_peak_gib=candidate,
        live_unrealized_gib=live_unrealized,
        margin_gib=margin,
        required_gib=required,
        headroom_gib=headroom if measurable else 0.0,
        ceiling_gib=ceiling,
        ceiling_headroom_gib=ceiling_headroom if measurable else 0.0,
        basis="tools.mem_basis.conservative_free_gib (governor reclaimable-aware snapshot)",
        measurable=measurable,
        naive_shell_reclaimable_gib=(naive_shell_reclaimable_gib() if include_naive_contrast else None),
        reasons=tuple(reasons),
    )


# ── Metal-contention telemetry ──────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class CellRate:
    cell_id: str
    steps_per_min: float
    steps_delta: int
    window_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "steps_per_min": round(self.steps_per_min, 4),
            "steps_delta": self.steps_delta,
            "window_s": round(self.window_s, 3),
        }


def sample_cell_rates(cells: Sequence[LiveCell], *, window_s: float) -> list[CellRate]:
    """Measure steps/min for each cell by counting ``history.jsonl`` growth over ``window_s``.

    Purely observational: two reads of a row count separated by a wait.  Never touches the cells.
    """
    before = {cell.cell_id: count_history_steps(cell.run_dir) for cell in cells}
    t0 = time.monotonic()
    time.sleep(max(0.0, float(window_s)))
    elapsed = time.monotonic() - t0
    after = {cell.cell_id: count_history_steps(cell.run_dir) for cell in cells}

    rates: list[CellRate] = []
    for cell in cells:
        start, end = before.get(cell.cell_id), after.get(cell.cell_id)
        if start is None or end is None or elapsed <= 0.0:
            continue
        delta = end - start
        rates.append(
            CellRate(
                cell_id=cell.cell_id,
                steps_per_min=delta * 60.0 / elapsed,
                steps_delta=delta,
                window_s=elapsed,
            )
        )
    return rates


def cpu_load_context() -> dict[str, Any]:
    """CPU pressure alongside the Metal cells -- the co-factor a cell-count alone hides.

    A steps/min figure is only comparable against another taken under similar CPU load: the cells
    run a CPU-side scorer, so N concurrent CPU arms slow them even when the Metal is unchanged.
    Sister measurement (``ddm_bh1``, 2026-09-04) saw ng2 at ~15.5 steps/min with four CPU arms live.
    Recording ``load_avg_1m`` and ``logical_cpus`` beside the cell count is what makes two
    contention rows comparable at all; without it the ratio silently mixes two different machines.
    """
    context: dict[str, Any] = {"logical_cpus": os.cpu_count()}
    try:
        one, five, fifteen = os.getloadavg()
        context.update(load_avg_1m=round(one, 3), load_avg_5m=round(five, 3), load_avg_15m=round(fifteen, 3))
    except (OSError, AttributeError):
        context.update(load_avg_1m=None, load_avg_5m=None, load_avg_15m=None)
    return context


def append_contention_row(row: Mapping[str, Any], ledger_path: Path | None = None) -> dict[str, Any]:
    """Append one contention observation under an exclusive lock (canonical JSONL store pattern)."""
    path = CONTENTION_LEDGER if ledger_path is None else Path(ledger_path)
    payload = dict(row)
    payload.setdefault("schema", THROUGHPUT_ROW_SCHEMA)
    payload.setdefault("recorded_utc", utc_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return payload


def read_contention_rows(ledger_path: Path | None = None) -> list[dict[str, Any]]:
    path = CONTENTION_LEDGER if ledger_path is None else Path(ledger_path)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == THROUGHPUT_ROW_SCHEMA:
            rows.append(value)
    return rows


def serial_baseline_steps_per_min(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Best observed single-cell (concurrency == 1) total throughput, or None when never measured."""
    singles = [
        float(row["total_steps_per_min"])
        for row in rows
        if row.get("concurrency") == 1 and isinstance(row.get("total_steps_per_min"), (int, float))
    ]
    return max(singles) if singles else None


@dataclasses.dataclass(frozen=True)
class ThroughputVerdict:
    """The Metal-contention leg: does concurrency still pay?"""

    admits: bool
    concurrency_observed: int | None
    total_steps_per_min: float | None
    serial_baseline_steps_per_min: float | None
    ratio: float | None
    evidence: str
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = dict(dataclasses.asdict(self))
        payload["reasons"] = list(self.reasons)
        for key in ("total_steps_per_min", "serial_baseline_steps_per_min", "ratio"):
            value = payload.get(key)
            if isinstance(value, float):
                payload[key] = round(value, 4)
        return payload


def concurrency_resolution(
    rows: Sequence[Mapping[str, Any]],
    *,
    concurrency: int,
    baseline: float | None,
) -> dict[str, Any]:
    """Is the measured speedup at ``concurrency`` RESOLVED, and does it pay? (ddm_gov2)

    Two independent ways a set of rows fails to resolve, both MEASURED on this box 2026-09-04:

    * **They straddle the baseline.**  gv1 measured the same two cells twice, 30 min apart:
      31.2854 steps/min (ratio 1.117, PAYS) then 26.9996 (ratio 0.964, COSTS).
    * **The spread exceeds the effect.**  4.286 steps/min between those windows against a 3.285
      apparent effect -- the noise is bigger than the signal, so neither window is the answer.

    Either condition returns ``resolved=False``, and the standing rule below refuses on it.
    """
    at_n = [
        row
        for row in rows
        if isinstance(row.get("concurrency"), int)
        and row["concurrency"] >= concurrency
        and isinstance(row.get("total_steps_per_min"), (int, float))
    ]
    totals = [float(row["total_steps_per_min"]) for row in at_n]
    result: dict[str, Any] = {
        "concurrency_required": concurrency,
        "row_count": len(at_n),
        "totals_steps_per_min": [round(value, 4) for value in totals],
        "resolved": False,
        "pays": False,
        "reason": "",
    }
    if not totals:
        result["reason"] = f"no ledger row at concurrency >= {concurrency}"
        return result
    if baseline is None or baseline <= 0:
        result["reason"] = "no concurrency==1 serial baseline row; the ratio is undefined"
        return result
    ratios = [value / baseline for value in totals]
    result["ratios"] = [round(value, 4) for value in ratios]
    spread = max(totals) - min(totals)
    effect = abs((sum(totals) / len(totals)) - baseline)
    result["spread_steps_per_min"] = round(spread, 4)
    result["effect_steps_per_min"] = round(effect, 4)
    if len(totals) >= 2 and min(ratios) < 1.0 <= max(ratios):
        result["reason"] = (
            f"{len(totals)} rows STRADDLE the baseline (ratios {min(ratios):.3f}..{max(ratios):.3f}); "
            "concurrency neither pays nor costs on the evidence"
        )
        return result
    if len(totals) >= 2 and spread > effect:
        result["reason"] = (
            f"between-window spread {spread:.3f} steps/min exceeds the {effect:.3f} steps/min effect; "
            "the noise is larger than the signal"
        )
        return result
    result["resolved"] = True
    result["pays"] = bool(min(ratios) >= 1.0)
    result["reason"] = (
        f"{len(totals)} row(s) at concurrency >= {concurrency}, ratios {min(ratios):.3f}..{max(ratios):.3f} "
        f"vs serial baseline {baseline:.2f} steps/min"
    )
    return result


def throughput_verdict(rows: Sequence[Mapping[str, Any]], *, live_count: int) -> ThroughputVerdict:
    """The STANDING CONCURRENCY LAW of this governor (ddm_gov2 encoded it; do not patch it away).

    1. **A candidate that would run ALONE is never gated on contention.**  ``live_count <= 1``
       means no training cell is live, so concurrency-N evidence does not apply.  MEASURED defect
       2026-09-04: ``max(2, live_count)`` with contending=1 consulted the N=2 rows and refused ng4
       for 90 minutes with the Metal free.

    2. **A SECOND (or later) Metal cell is REFUSED by default.**  It is admitted only when the
       measured evidence at that concurrency is RESOLVED *and* pays (ratio >= 1.0).  Today's N=2
       evidence is UNRESOLVED -- 1.117 then 0.964 across two identical windows, spread 4.286 >
       effect 3.285 -- so two Metal cells REFUSE until an N>=3-window row resolves it.

       This REVERSES gv1's admit-on-no-evidence default, deliberately.  gv1's reason was that a
       governor which refuses everything can never collect data; the answer is not to admit blind
       but to make the measurement path EXPLICIT: ``decide_admission(..., concurrency_measurement_
       override="<rationale>")`` admits and stamps the row as an operator-authorized measurement.
       "Off" stays a tracked queue with a named way out, not a silent yes.
    """
    baseline = serial_baseline_steps_per_min(rows)
    if live_count <= 1:
        # The candidate would be the ONLY training cell on the Metal: there is no contention to
        # price, so concurrency-2 evidence must not gate it.  MEASURED defect (2026-09-04): with
        # the Metal free and five non-cell jobs live, ``max(2, 1)`` consulted the N=2 rows and
        # refused ng4 for 90 minutes — the guard read the wrong denominator (VACUITY's sibling).
        return ThroughputVerdict(
            admits=True,
            concurrency_observed=None,
            total_steps_per_min=None,
            serial_baseline_steps_per_min=baseline,
            ratio=None,
            evidence="SOLE_CELL_NO_CONTENTION",
            reasons=("no training cell is live; the candidate runs alone, so the throughput leg does not apply",),
        )
    required = max(2, live_count)
    resolution = concurrency_resolution(rows, concurrency=required, baseline=baseline)
    concurrent = [
        row
        for row in rows
        if isinstance(row.get("concurrency"), int)
        and row["concurrency"] >= required
        and isinstance(row.get("total_steps_per_min"), (int, float))
    ]
    latest = max(concurrent, key=lambda row: str(row.get("recorded_utc", ""))) if concurrent else None
    total = None if latest is None else float(latest["total_steps_per_min"])
    ratio = None if (total is None or not baseline) else total / baseline

    if not resolution["resolved"]:
        # RULE 2: refuse, and say exactly which of the two failure modes fired.
        evidence = (
            "NO_CONCURRENT_OBSERVATION"
            if not concurrent
            else ("NO_SERIAL_BASELINE" if baseline is None else "UNRESOLVED_AT_CONCURRENCY")
        )
        return ThroughputVerdict(
            admits=False,
            concurrency_observed=None if latest is None else int(latest["concurrency"]),
            total_steps_per_min=total,
            serial_baseline_steps_per_min=baseline,
            ratio=ratio,
            evidence=evidence,
            reasons=(
                f"a {live_count}-cell Metal configuration is REFUSED by the standing concurrency "
                f"law: {resolution['reason']}",
                "cure: resolve it with a measurement launch "
                "(decide_admission(..., concurrency_measurement_override=<rationale>)) or an "
                "N>=3-window contention row",
            ),
        )
    if not resolution["pays"]:
        return ThroughputVerdict(
            admits=False,
            concurrency_observed=None if latest is None else int(latest["concurrency"]),
            total_steps_per_min=total,
            serial_baseline_steps_per_min=baseline,
            ratio=ratio,
            evidence="MEASURED",
            reasons=(f"measured concurrency COSTS throughput: {resolution['reason']}",),
        )
    return ThroughputVerdict(
        admits=True,
        concurrency_observed=None if latest is None else int(latest["concurrency"]),
        total_steps_per_min=total,
        serial_baseline_steps_per_min=baseline,
        ratio=ratio,
        evidence="MEASURED",
        reasons=(f"measured concurrency PAYS: {resolution['reason']}",),
    )


# ── the composed decision ───────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class AdmissionDecision:
    verdict: str  # "ADMIT" | "REFUSE"
    memory: MemoryVerdict
    throughput: ThroughputVerdict
    live_cells: tuple[LiveCell, ...]
    decided_utc: str

    @property
    def admits(self) -> bool:
        return self.verdict == "ADMIT"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ADMISSION_SCHEMA,
            "verdict": self.verdict,
            "decided_utc": self.decided_utc,
            "memory": self.memory.as_dict(),
            "throughput": self.throughput.as_dict(),
            "live_cells": [cell.as_dict() for cell in self.live_cells],
            "live_cell_count": len(self.live_cells),
            "score_claim": False,
            "actuation": "ADVISORY_GATE",
        }

    def human_lines(self) -> list[str]:
        mem, thr = self.memory, self.throughput
        lines = [
            f"cell admission: {self.verdict} "
            f"(candidate peak {mem.candidate_peak_gib:.2f} GiB, {len(self.live_cells)} live cell(s))",
            f"  memory  : reclaimable {mem.reclaimable_gib:.2f} GiB - required {mem.required_gib:.2f} GiB "
            f"= headroom {mem.headroom_gib:+.2f} GiB "
            f"[candidate {mem.candidate_peak_gib:.2f} + live-unrealized {mem.live_unrealized_gib:.2f} "
            f"+ margin {mem.margin_gib:.2f}]",
            f"  ceiling : committed {mem.committed_gib:.2f} GiB, ceiling {mem.ceiling_gib:.1f} GiB, "
            f"headroom {mem.ceiling_headroom_gib:+.2f} GiB",
        ]
        if mem.naive_shell_reclaimable_gib is not None:
            lines.append(
                f"  basis   : canonical {mem.reclaimable_gib:.2f} GiB vs naive shell free+inactive "
                f"{mem.naive_shell_reclaimable_gib:.2f} GiB (decision uses the canonical basis)"
            )
        lines.append(f"  throughput: {thr.evidence} -- {thr.reasons[0] if thr.reasons else 'n/a'}")
        for cell in self.live_cells:
            progress = (
                f"{cell.completed_steps}/{cell.total_steps}"
                if cell.completed_steps is not None and cell.total_steps
                else "steps unknown"
            )
            lines.append(
                f"  live    : {cell.cell_id} pid={cell.pid} {progress} "
                f"rss={cell.current_rss_gib if cell.current_rss_gib is None else round(cell.current_rss_gib, 2)} GiB "
                f"declared={cell.declared_peak_gib:.2f} GiB "
                f"unrealized={cell.unrealized_growth_gib:.2f} GiB"
            )
        for reason in mem.reasons:
            lines.append(f"  why     : {reason}")
        return lines


def decide_admission(
    candidate_peak_gib: float,
    *,
    margin_gib: float = DEFAULT_MARGIN_GIB,
    roots: Sequence[Path] | None = None,
    live_cells: Sequence[LiveCell] | None = None,
    ledger_path: Path | None = None,
    ceiling_gib: float | None = None,
    include_naive_contrast: bool = True,
    walk_roots: bool | None = None,
    concurrency_measurement_override: str | None = None,
) -> AdmissionDecision:
    """Compose the memory and throughput legs into one ADMIT/REFUSE with full arithmetic.

    ``concurrency_measurement_override`` is the ONE named way past the standing concurrency law
    (see :func:`throughput_verdict`): it admits a second Metal cell for the express purpose of
    MEASURING the contention that the law is refusing on, and stamps the rationale into the
    decision.  It can never relax the MEMORY leg -- that one has no override at all.
    """
    cells = list(live_cells) if live_cells is not None else discover_live_cells(roots, walk_roots=walk_roots)
    mem = memory_verdict(
        candidate_peak_gib,
        cells,
        margin_gib=margin_gib,
        ceiling_gib=ceiling_gib,
        include_naive_contrast=include_naive_contrast,
    )
    # MEMORY charges every live governed job (they all hold RAM). THROUGHPUT counts only TRAINING
    # CELLS: contention is about cells competing for the Metal, and counting unrelated jobs would
    # inflate the required concurrency level until no ledger row ever matched, silently making the
    # throughput leg vacuous ("VACUITY==PASS").
    contending = sum(1 for cell in cells if cell.is_cell) + 1
    thr = throughput_verdict(read_contention_rows(ledger_path), live_count=contending)
    if (
        not thr.admits
        and thr.evidence != "MEASURED"
        and isinstance(concurrency_measurement_override, str)
        and concurrency_measurement_override.strip()
    ):
        # A measurement launch: the law refused for want of RESOLVED evidence, and this is the
        # authorized way to go and get it.  Never applied when the evidence is MEASURED-and-COSTS
        # -- that is a real negative, not a data gap.
        thr = dataclasses.replace(
            thr,
            admits=True,
            evidence="MEASUREMENT_OVERRIDE",
            reasons=(
                *thr.reasons,
                f"admitted as an authorized CONTENTION MEASUREMENT: {concurrency_measurement_override.strip()}",
            ),
        )
    return AdmissionDecision(
        verdict="ADMIT" if (mem.admits and thr.admits) else "REFUSE",
        memory=mem,
        throughput=thr,
        live_cells=tuple(cells),
        decided_utc=utc_text(),
    )


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────────


def _roots_from_args(values: Sequence[str] | None) -> Sequence[Path] | None:
    return [Path(value) for value in values] if values else None


def _warn_forensic_walk(args: argparse.Namespace) -> None:
    """Print the FORENSIC-MODE warning whenever the caller opts back into the SSD walk."""
    if getattr(args, "walk_roots", False) or getattr(args, "root", None):
        print(
            "[cell_admission] WARNING: --walk-roots/--root is FORENSIC MODE. Discovery is "
            "complete from the process table alone (MEASURED 0.045 s); the walk adds only "
            "manifests on disk, costs >120 s over both SSD roots, and is never required for "
            "admission.",
            file=sys.stderr,
        )


def _cmd_admit(args: argparse.Namespace) -> int:
    _warn_forensic_walk(args)
    decision = decide_admission(
        args.candidate_peak_gib,
        margin_gib=args.margin_gib,
        roots=_roots_from_args(args.root),
        ledger_path=args.ledger,
        ceiling_gib=args.ceiling_gib,
        walk_roots=True if getattr(args, "walk_roots", False) else None,
        concurrency_measurement_override=getattr(args, "concurrency_measurement_override", None),
    )
    if args.json:
        print(json.dumps(decision.as_dict(), indent=2, sort_keys=True))
    else:
        print("\n".join(decision.human_lines()))
    if decision.admits:
        return RC_ADMIT
    return RC_UNMEASURABLE if not decision.memory.measurable else RC_REFUSE


def _cmd_cells(args: argparse.Namespace) -> int:
    _warn_forensic_walk(args)
    cells = discover_live_cells(
        _roots_from_args(args.root),
        walk_roots=True if getattr(args, "walk_roots", False) else None,
        process_table=not getattr(args, "disk_only", False),
    )
    if args.json:
        print(
            json.dumps(
                {
                    "schema": "ddm_gv1_live_cells.v1",
                    "observed_utc": utc_text(),
                    "count": len(cells),
                    "cells": [cell.as_dict() for cell in cells],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not cells:
        print("live cells: NONE")
        return 0
    print(f"live governed jobs: {len(cells)} ({sum(1 for c in cells if c.is_cell)} training cells)")
    for cell in cells:
        kind = "cell" if cell.is_cell else "job "
        print(
            f"  [{kind}] {cell.cell_id} pid={cell.pid} "
            f"steps={cell.completed_steps}/{cell.total_steps} "
            f"declared_peak={cell.declared_peak_gib:.2f} GiB "
            f"tree_rss={cell.current_rss_gib if cell.current_rss_gib is None else round(cell.current_rss_gib, 2)} GiB "
            f"unrealized={cell.unrealized_growth_gib:.2f} GiB "
            f"[{cell.discovery_source}/{cell.declared_peak_source}]"
        )
    check = registry_cross_check(cells)
    if not check["agrees"]:
        print(
            f"  CROSS-CHECK DISAGREEMENT: the launch registry has {len(check['missed_by_process_table'])} "
            f"live row(s) the process table missed: {check['missed_by_process_table']}",
            file=sys.stderr,
        )
    else:
        print(f"  registry cross-check: agrees ({check['registry_live_rows']} live registry row(s))")
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    cells = discover_live_cells(
        _roots_from_args(args.root),
        walk_roots=True if getattr(args, "walk_roots", False) else None,
        process_table=not getattr(args, "disk_only", False),
    )
    rates = sample_cell_rates(cells, window_s=args.window_s)
    total = sum(rate.steps_per_min for rate in rates)
    row = {
        "schema": THROUGHPUT_ROW_SCHEMA,
        "recorded_utc": utc_text(),
        "concurrency": len(rates),
        "live_job_count": len(cells),
        "training_cell_count": sum(1 for cell in cells if cell.is_cell),
        "window_s": round(args.window_s, 3),
        "cells": [rate.as_dict() for rate in rates],
        "total_steps_per_min": round(total, 4),
        # CPU pressure is a co-factor on Metal-cell throughput; without it two rows are not
        # comparable and the speedup ratio silently mixes two different machines.
        "cpu_load_context": cpu_load_context(),
        "note": args.note or "",
        "score_claim": False,
    }
    if not args.no_write and rates:
        append_contention_row(row, args.ledger)
        row["written_to"] = str(args.ledger or CONTENTION_LEDGER)
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


def _cmd_contention(args: argparse.Namespace) -> int:
    rows = read_contention_rows(args.ledger)
    baseline = serial_baseline_steps_per_min(rows)
    verdict = throughput_verdict(rows, live_count=args.live_count)
    payload = {
        "schema": "ddm_gv1_metal_contention_summary.v1",
        "rows": len(rows),
        "serial_baseline_steps_per_min": baseline,
        "verdict": verdict.as_dict(),
        "observations": rows[-args.tail :] if args.tail > 0 else rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    common_roots: dict[str, Any] = {
        "action": "append",
        "metavar": "PATH",
        "help": (
            "FORENSIC ONLY: manifest root to scan on disk (repeatable). Discovery does NOT need "
            "this -- the process table is complete by construction."
        ),
    }
    walk_help = (
        "FORENSIC ONLY: additionally walk the SSD manifest roots. Costs >120 s (MEASURED) and "
        "cannot find a live job the process table missed; it only lists manifests on disk."
    )

    admit = sub.add_parser("admit", help="ADMIT/REFUSE a candidate cell (rc 0 admit / 2 refuse / 3 unmeasurable)")
    admit.add_argument("--candidate-peak-gib", type=float, required=True)
    admit.add_argument("--margin-gib", type=float, default=DEFAULT_MARGIN_GIB)
    admit.add_argument("--ceiling-gib", type=float, default=None)
    admit.add_argument("--ledger", type=Path, default=None)
    admit.add_argument("--root", **common_roots)
    admit.add_argument(
        "--walk-roots",
        action="store_true",
        help=walk_help,
    )
    admit.add_argument(
        "--concurrency-measurement-override",
        default=None,
        metavar="RATIONALE",
        help=(
            "admit a SECOND Metal cell that the standing concurrency law refuses for want of "
            "RESOLVED evidence, expressly to measure that contention. Never relaxes the memory leg."
        ),
    )
    admit.add_argument("--json", action="store_true")
    admit.set_defaults(func=_cmd_admit)

    cells = sub.add_parser("cells", help="list live cells with their declared peaks and progress")
    cells.add_argument("--root", **common_roots)
    cells.add_argument(
        "--walk-roots",
        action="store_true",
        help=walk_help,
    )
    cells.add_argument(
        "--disk-only",
        action="store_true",
        help="FORENSIC ONLY: report manifests on disk WITHOUT the process table (never for admission)",
    )
    cells.add_argument("--json", action="store_true")
    cells.set_defaults(func=_cmd_cells)

    sample = sub.add_parser("sample", help="measure live-cell step rates over a window and record them")
    sample.add_argument("--window-s", type=float, default=300.0)
    sample.add_argument("--ledger", type=Path, default=None)
    sample.add_argument("--note", default="")
    sample.add_argument("--no-write", action="store_true", help="measure without touching the ledger")
    sample.add_argument("--root", **common_roots)
    sample.add_argument(
        "--disk-only",
        action="store_true",
        help="FORENSIC ONLY: sample manifests on disk WITHOUT the process table",
    )
    sample.set_defaults(func=_cmd_sample)

    contention = sub.add_parser("contention", help="summarize the Metal-contention ledger")
    contention.add_argument("--ledger", type=Path, default=None)
    contention.add_argument("--live-count", type=int, default=2)
    contention.add_argument("--tail", type=int, default=5)
    contention.set_defaults(func=_cmd_contention)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
