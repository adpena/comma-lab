# SPDX-License-Identifier: MIT
"""Fcntl-locked JSONL persistence for compress-time pass outcomes.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§5.5 production hardening: pass outcomes optionally persist to a canonical
fcntl-locked JSONL ledger at
``.omx/state/compress_time_optimization_pass_outcomes.jsonl``.

Persistence is OPT-IN — ``ComposableCompressPipeline.run()`` does NOT
auto-persist (the canonical contract treats run as a pure functional
operation). Callers that want the audit trail wrap the call::

    result = pipeline.run(seed_state)
    for outcome in result.per_pass_outcomes:
        append_pass_outcome_locked(outcome)

Mirrors the canonical pattern in ``tac.boosting.persistence`` /
``tac.deploy.modal.call_id_ledger`` (Catalog #245) — fcntl LOCK_EX +
STRICT-load + unique .tmp + os.replace + APPEND-ONLY per Catalog #128 /
#131 / #132 sister discipline.

PV-4 verified the canonical pattern. We use the SIMPLER full-rewrite path
here (sufficient for low-throughput design-time pass outcomes; the OP-10
O(1) amortized path is overkill at this cadence — documented decision,
not premature optimization).
"""

from __future__ import annotations

import datetime as _dt
import errno
import fcntl
import json
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from tac.compress_time_optimization.errors import (
    CompressTimeLedgerCorruptError,
)

__all__ = [
    "COMPRESS_TIME_OPT_PASS_OUTCOMES_LOCK",
    "COMPRESS_TIME_OPT_PASS_OUTCOMES_PATH",
    "COMPRESS_TIME_OPT_PASS_OUTCOMES_SCHEMA_VERSION",
    "append_pass_outcome_locked",
    "load_pass_outcomes",
    "load_pass_outcomes_strict",
]

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
COMPRESS_TIME_OPT_PASS_OUTCOMES_PATH = (
    _REPO_ROOT
    / ".omx"
    / "state"
    / "compress_time_optimization_pass_outcomes.jsonl"
)
COMPRESS_TIME_OPT_PASS_OUTCOMES_LOCK = (
    COMPRESS_TIME_OPT_PASS_OUTCOMES_PATH.with_suffix(
        COMPRESS_TIME_OPT_PASS_OUTCOMES_PATH.suffix + ".lock"
    )
)
COMPRESS_TIME_OPT_PASS_OUTCOMES_SCHEMA_VERSION = (
    "compress_time_optimization_pass_outcomes_v1"
)
LOCK_TIMEOUT_SEC = 30.0

# ---------------------------------------------------------------------------
# Per-process lock-state tracking (mirrors tac.boosting / call_id_ledger
# pattern)
# ---------------------------------------------------------------------------

_lock_lock = threading.Lock()
_lock_depth = 0


def _pass_outcomes_lock_held() -> bool:
    """Return True if the current process holds the fcntl lock."""
    with _lock_lock:
        return _lock_depth > 0


@contextmanager
def _pass_outcomes_lock(
    lock_path: Path | None = None,
) -> Iterator[None]:
    """Acquire fcntl LOCK_EX on the pass-outcomes lock file.

    Per-process advisory lock (multiple processes serialize against each
    other; a single process re-entering is tracked via depth counter so
    nested context managers do not deadlock).
    """
    global _lock_depth
    path = lock_path or COMPRESS_TIME_OPT_PASS_OUTCOMES_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + LOCK_TIMEOUT_SEC
    fd: int | None = None
    acquired_here = False
    try:
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired_here = True
                break
            except (OSError, BlockingIOError) as exc:
                if isinstance(exc, OSError) and exc.errno not in (
                    errno.EWOULDBLOCK,
                    errno.EAGAIN,
                ):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out acquiring compress-time-optimization "
                        f"pass-outcomes lock after {LOCK_TIMEOUT_SEC}s at "
                        f"{path}"
                    )
                time.sleep(0.05)
        with _lock_lock:
            _lock_depth += 1
        yield
    finally:
        if acquired_here:
            with _lock_lock:
                _lock_depth -= 1
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Record validation + atomic write
# ---------------------------------------------------------------------------


def _validate_outcome_record(record: dict[str, Any]) -> None:
    """Validate the structural shape of a pass-outcome row."""
    if not isinstance(record, dict):
        raise CompressTimeLedgerCorruptError(
            f"pass outcome record must be a dict, got "
            f"{type(record).__name__}"
        )
    schema_version = record.get("schema_version")
    if schema_version != COMPRESS_TIME_OPT_PASS_OUTCOMES_SCHEMA_VERSION:
        raise CompressTimeLedgerCorruptError(
            f"schema_version must be "
            f"{COMPRESS_TIME_OPT_PASS_OUTCOMES_SCHEMA_VERSION!r}; "
            f"got {schema_version!r}"
        )
    if not isinstance(record.get("pass_id"), str):
        raise CompressTimeLedgerCorruptError(
            f"pass_id must be a string; got {record.get('pass_id')!r}"
        )
    if not isinstance(record.get("status"), str):
        raise CompressTimeLedgerCorruptError(
            f"status must be a string; got {record.get('status')!r}"
        )


def _quarantine_corrupt_ledger(path: Path) -> Path:
    """Move a corrupt ledger file to .corrupt.<utc> per Catalog #138 pattern."""
    if not path.exists():
        return path
    stamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.with_suffix(path.suffix + f".corrupt.{stamp}")
    os.rename(path, quarantine)
    return quarantine


def _atomic_write_ledger(
    rows: list[dict[str, Any]],
    path: Path | None = None,
) -> None:
    """Atomic write — unique .tmp.<uuid12> + fsync + os.replace.

    Runtime-asserts the caller holds ``_pass_outcomes_lock``. Per Catalog
    #132 + the canonical call_id_ledger pattern: writes are append-only —
    every existing row is preserved verbatim; only new rows are added by
    the caller before this function runs.
    """
    if not _pass_outcomes_lock_held():
        raise RuntimeError(
            "_atomic_write_ledger called WITHOUT holding _pass_outcomes_lock. "
            "This is a CONCURRENCY BUG: concurrent appends can silently drop "
            "rows. Use append_pass_outcome_locked() which owns the full "
            "lock-load-append-save cycle."
        )
    p = path or COMPRESS_TIME_OPT_PASS_OUTCOMES_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    tmp = p.with_suffix(p.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    try:
        tmp.write_text(payload, encoding="utf-8")
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def append_pass_outcome_locked(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    quarantine_on_corrupt: bool = True,
) -> dict[str, Any]:
    """Append a single pass-outcome record under fcntl lock.

    Pattern (full-rewrite; sufficient for low-throughput design-time
    cadence):
      1. Acquire fcntl LOCK_EX
      2. STRICT-reload ledger from disk (raises CompressTimeLedgerCorruptError
         on ANY malformed row; quarantines if quarantine_on_corrupt=True)
      3. Append the new record (append-only per Catalog #132)
      4. Atomic write via unique .tmp.<uuid12> + fsync + os.replace

    Returns the record (augmented with timestamp / pid / host fields if
    they were absent in the input).
    """
    record = dict(record)
    record.setdefault(
        "schema_version", COMPRESS_TIME_OPT_PASS_OUTCOMES_SCHEMA_VERSION
    )
    record.setdefault(
        "written_at_utc",
        _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
    )
    record.setdefault("written_pid", os.getpid())
    record.setdefault("written_host", socket.gethostname())
    _validate_outcome_record(record)

    p = path or COMPRESS_TIME_OPT_PASS_OUTCOMES_PATH
    lp = lock_path or COMPRESS_TIME_OPT_PASS_OUTCOMES_LOCK

    with _pass_outcomes_lock(lp):
        try:
            rows = load_pass_outcomes_strict(p)
        except CompressTimeLedgerCorruptError as exc:
            if quarantine_on_corrupt:
                q = _quarantine_corrupt_ledger(p)
                raise CompressTimeLedgerCorruptError(
                    f"compress-time-optimization pass-outcomes ledger at {p} "
                    f"was corrupt; quarantined to {q}. Append refused; "
                    f"operator must repair (see "
                    f"CompressTimeLedgerCorruptError docstring)."
                ) from exc
            raise

        new_rows = [*rows, record]
        _atomic_write_ledger(new_rows, p)
        return record


def load_pass_outcomes(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load all pass-outcome rows (lenient — skips malformed lines).

    Use for read-only queries / aggregations where occasional malformed
    rows are acceptable. For strict validation use
    :func:`load_pass_outcomes_strict`.
    """
    p = path or COMPRESS_TIME_OPT_PASS_OUTCOMES_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def load_pass_outcomes_strict(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load all pass-outcome rows; raise CompressTimeLedgerCorruptError on
    ANY malformed line.

    Per Catalog #138 fail-closed discipline — sister of
    ``tac.boosting.persistence.load_stage_outcomes_strict``. Used by
    ``append_pass_outcome_locked`` to refuse a write on top of a corrupt
    ledger.
    """
    p = path or COMPRESS_TIME_OPT_PASS_OUTCOMES_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise CompressTimeLedgerCorruptError(
            f"failed to read compress-time-optimization pass-outcomes ledger "
            f"{p}: {exc}"
        ) from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompressTimeLedgerCorruptError(
                f"line {lineno} of {p} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise CompressTimeLedgerCorruptError(
                f"line {lineno} of {p} is not a dict: "
                f"type={type(obj).__name__}"
            )
        rows.append(obj)
    return rows
