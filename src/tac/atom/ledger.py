# SPDX-License-Identifier: MIT
"""Canonical fcntl-locked JSONL append-only ledger for ``tac.atom``.

Path: ``.omx/state/atom_ledger.jsonl``
Lock: ``.omx/state/.atom_ledger.lock``

Mirrors the canonical 4-layer pattern from
``tac.deploy.modal.call_id_ledger`` (Catalog #245):

  1. fcntl ``LOCK_EX`` on the lock file (separate path from ledger so
     readers can mmap the ledger without lock contention),
  2. atomic ``os.write`` with ``O_APPEND`` semantics for amortized O(1)
     append (POSIX-atomic for payloads <= PIPE_BUF),
  3. ``fsync`` for durability,
  4. quarantine on corrupt (``.corrupt.<utc>`` per Catalog #138 strict-load
     discipline) — the loader is fail-closed.

Per Catalog #131 sister discipline: ``ATOM_LEDGER_PATH`` is registered in
``src/tac/preflight.py::_SHARED_STATE_PATH_MARKERS`` once a Catalog #
follow-up wave lands. Until then this module is the SOLE writer and the
``_BARE_WRITE_CANONICAL_HELPER_CALL_TOKENS`` set covers
``append_atom`` / ``load_atoms_strict`` / ``_append_event_locked``.

Per CLAUDE.md HISTORICAL_PROVENANCE Catalog #110/#113: the ledger is
APPEND-ONLY; outcome updates (e.g. marking a probe-outcome atom as
SUPERSEDED) MUST be written as NEW rows referencing the same ``atom_id``
rather than mutating the original row in place.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atom import Atom
from .types import AtomKind, ResolutionPath


def _repo_root() -> Path:
    # Walk up until we find the .omx directory; fall back to module-relative.
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".omx").is_dir():
            return parent
    # Last-resort: project root by relative climb (src/tac/atom -> repo)
    return here.parent.parent.parent.parent


REPO_ROOT = _repo_root()
ATOM_LEDGER_PATH = REPO_ROOT / ".omx" / "state" / "atom_ledger.jsonl"
ATOM_LEDGER_LOCK = REPO_ROOT / ".omx" / "state" / ".atom_ledger.lock"
LOCK_TIMEOUT_SECONDS = 30.0
SCHEMA_VERSION = "atom_ledger_v1_20260518"


class AtomLedgerCorruptError(RuntimeError):
    """Raised by ``load_atoms_strict`` on any unparseable / malformed row.

    Mirrors ``tac.deploy.modal.call_id_ledger.CallIdLedgerCorruptError`` per
    Catalog #138 strict-load discipline. The fail-closed loader prevents
    silent state-loss when a torn write or corrupt row would otherwise be
    skipped by a lenient reader.
    """


def _ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _acquire_lock(timeout: float = LOCK_TIMEOUT_SECONDS):
    """Acquire exclusive fcntl lock on ATOM_LEDGER_LOCK with timeout.

    Returns the open file descriptor; caller is responsible for closing it
    via ``os.close(fd)``.
    """
    _ensure_dirs(ATOM_LEDGER_LOCK)
    fd = os.open(ATOM_LEDGER_LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError as exc:
            if time.monotonic() > deadline:
                os.close(fd)
                raise TimeoutError(
                    f"timed out acquiring fcntl lock on {ATOM_LEDGER_LOCK} "
                    f"after {timeout:.1f}s"
                ) from exc
            time.sleep(0.05)


def _release_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _append_event_locked(row: dict[str, Any]) -> dict[str, Any]:
    """Write one row to the ledger under fcntl lock; return the row.

    Per the canonical pattern, the row dict is serialized with
    ``sort_keys=True`` so the JSONL is byte-stable across processes.
    """
    _ensure_dirs(ATOM_LEDGER_PATH)
    payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    fd = _acquire_lock()
    try:
        # O_APPEND ensures atomic-at-the-kernel-level append for small
        # payloads (<= PIPE_BUF); fcntl additionally serializes us against
        # other process appenders.
        out_fd = os.open(ATOM_LEDGER_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(out_fd, payload)
            os.fsync(out_fd)
        finally:
            os.close(out_fd)
    finally:
        _release_lock(fd)
    return row


def append_atom(atom: Atom, *, event_type: str = "registered") -> dict[str, Any]:
    """Append one ``Atom`` to the canonical ledger.

    ``event_type`` is a top-level row metadata key (defaults to "registered";
    valid values include "registered" / "superseded" / "expired" /
    "operator_override" / "promoted") so downstream readers can filter
    lifecycle events without re-deserializing the Atom payload.
    """
    if not isinstance(atom, Atom):
        raise TypeError(f"append_atom expected Atom (got {type(atom).__name__})")
    row = {
        "schema": SCHEMA_VERSION,
        "event_type": event_type,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "written_pid": os.getpid(),
        "written_host": os.uname().nodename,
        "atom": atom.to_jsonl_row(),
    }
    return _append_event_locked(row)


def _load_rows_lenient(path: Path) -> list[dict[str, Any]]:
    """Load all rows; skip malformed lines silently (the lenient path).

    Use ``load_atoms_strict`` for the fail-closed path.
    """
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_atoms_strict(path: Path | None = None) -> list[dict[str, Any]]:
    """Load all rows with fail-closed semantics per Catalog #138.

    On ANY JSON parse error or non-dict row, the ledger file is quarantined
    to ``<path>.corrupt.<utc>`` and ``AtomLedgerCorruptError`` is raised.
    The caller MUST handle the corruption (operator review) before further
    appends.
    """
    target = path or ATOM_LEDGER_PATH
    if not target.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                _quarantine_corrupt(target)
                raise AtomLedgerCorruptError(
                    f"JSON parse error at line {lineno} of {target}: {exc}; "
                    f"ledger quarantined per Catalog #138 fail-closed discipline"
                ) from exc
            if not isinstance(row, dict):
                _quarantine_corrupt(target)
                raise AtomLedgerCorruptError(
                    f"non-dict row at line {lineno} of {target}: type={type(row).__name__}"
                )
            rows.append(row)
    return rows


def _quarantine_corrupt(target: Path) -> None:
    """Move corrupt ledger to ``<target>.corrupt.<utc>`` for operator review."""
    if not target.is_file():
        return
    suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    quarantine_path = target.with_suffix(f"{target.suffix}.corrupt.{suffix}")
    try:
        shutil.move(str(target), str(quarantine_path))
    except OSError:
        # Best-effort; if quarantine fails, the raise above still surfaces
        # the corruption to the caller.
        pass


def query_by_kind(kind: AtomKind, *, path: Path | None = None) -> list[dict[str, Any]]:
    """Return all atom rows whose ``atom.kind`` matches.

    Uses the lenient loader because filtering should tolerate legacy /
    corrupt rows; callers that need fail-closed semantics should call
    ``load_atoms_strict`` directly.
    """
    rows = _load_rows_lenient(path or ATOM_LEDGER_PATH)
    return [
        r for r in rows if isinstance(r.get("atom"), dict) and r["atom"].get("kind") == kind.value
    ]


def query_by_resolution_path(
    resolution_path: ResolutionPath, *, path: Path | None = None
) -> list[dict[str, Any]]:
    """Return all atom rows whose ``atom.resolution_path`` matches."""
    rows = _load_rows_lenient(path or ATOM_LEDGER_PATH)
    return [
        r
        for r in rows
        if isinstance(r.get("atom"), dict)
        and r["atom"].get("resolution_path") == resolution_path.value
    ]


def query_by_min_predicted_impact(
    *, lower_bound_geq: float, path: Path | None = None
) -> list[dict[str, Any]]:
    """Return atom rows whose ``predicted_impact_delta_s_lower`` >= threshold.

    Useful for ranking candidates by minimum guaranteed predicted impact;
    pair with ``query_by_kind`` for kind-filtered ranking.
    """
    rows = _load_rows_lenient(path or ATOM_LEDGER_PATH)
    filtered: list[dict[str, Any]] = []
    for r in rows:
        atom = r.get("atom")
        if not isinstance(atom, dict):
            continue
        lo = atom.get("predicted_impact_delta_s_lower")
        if isinstance(lo, (int, float)) and lo >= lower_bound_geq:
            filtered.append(r)
    return filtered


def append_atoms_batch(atoms: Iterable[Atom]) -> list[dict[str, Any]]:
    """Append multiple atoms in ONE locked transaction.

    Saves repeated lock-acquire/release overhead for bulk subsumption flows
    (e.g. ingesting all 52 rows of the sister arbitrariness-extinction
    audit JSONL at once).
    """
    atom_list = list(atoms)
    if not atom_list:
        return []
    _ensure_dirs(ATOM_LEDGER_PATH)
    written: list[dict[str, Any]] = []
    fd = _acquire_lock()
    try:
        out_fd = os.open(
            ATOM_LEDGER_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644
        )
        try:
            for atom in atom_list:
                if not isinstance(atom, Atom):
                    raise TypeError(
                        f"append_atoms_batch expected Atom (got {type(atom).__name__})"
                    )
                row = {
                    "schema": SCHEMA_VERSION,
                    "event_type": "registered",
                    "written_at_utc": datetime.now(UTC).isoformat(),
                    "written_pid": os.getpid(),
                    "written_host": os.uname().nodename,
                    "atom": atom.to_jsonl_row(),
                }
                payload = (
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    + b"\n"
                )
                os.write(out_fd, payload)
                written.append(row)
            os.fsync(out_fd)
        finally:
            os.close(out_fd)
    finally:
        _release_lock(fd)
    return written


__all__ = [
    "ATOM_LEDGER_LOCK",
    "ATOM_LEDGER_PATH",
    "SCHEMA_VERSION",
    "AtomLedgerCorruptError",
    "append_atom",
    "append_atoms_batch",
    "load_atoms_strict",
    "query_by_kind",
    "query_by_min_predicted_impact",
    "query_by_resolution_path",
]
