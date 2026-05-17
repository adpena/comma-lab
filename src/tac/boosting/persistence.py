# SPDX-License-Identifier: MIT
"""Fcntl-locked JSONL persistence for boost stage outcomes.

Per `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
§5.5 production hardening: stage outcomes optionally persist to a
canonical fcntl-locked JSONL ledger at
``.omx/state/boosting_stage_outcomes.jsonl``.

Persistence is OPT-IN — ``ComposableBoostingPipeline.run()`` does NOT
auto-persist (the canonical contract treats run as a pure functional
operation). Callers that want the audit trail wrap the call::

    result = pipeline.run(seed_archive)
    for outcome in result.per_stage_outcomes:
        append_stage_outcome_locked(outcome)

Mirrors the canonical pattern in ``tac.deploy.modal.call_id_ledger``
(Catalog #245) — fcntl LOCK_EX + STRICT-load + unique .tmp + os.replace
+ APPEND-ONLY per Catalog #128/#131/#132 sister discipline.

PV-4 verified the canonical pattern. We use the SIMPLER full-rewrite
path here (sufficient for low-throughput design-time stage outcomes;
the OP-10 O(1) amortized path is overkill at this cadence — documented
decision, not premature optimization).
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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tac.boosting.errors import BoostingLedgerCorruptError

__all__ = [
    "BOOSTING_STAGE_OUTCOMES_LOCK",
    "BOOSTING_STAGE_OUTCOMES_PATH",
    "BOOSTING_STAGE_OUTCOMES_SCHEMA_VERSION",
    "append_stage_outcome_locked",
    "load_stage_outcomes",
    "load_stage_outcomes_strict",
]

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

# Repo-root anchored. Resolve from this file's location.
_REPO_ROOT = Path(__file__).resolve().parents[3]
BOOSTING_STAGE_OUTCOMES_PATH = (
    _REPO_ROOT / ".omx" / "state" / "boosting_stage_outcomes.jsonl"
)
BOOSTING_STAGE_OUTCOMES_LOCK = BOOSTING_STAGE_OUTCOMES_PATH.with_suffix(
    BOOSTING_STAGE_OUTCOMES_PATH.suffix + ".lock"
)
BOOSTING_STAGE_OUTCOMES_SCHEMA_VERSION = "boosting_stage_outcomes_v1"
LOCK_TIMEOUT_SEC = 30.0

# ---------------------------------------------------------------------------
# Per-process lock-state tracking (mirrors call_id_ledger pattern)
# ---------------------------------------------------------------------------

_lock_lock = threading.Lock()
_lock_depth = 0


def _stage_outcomes_lock_held() -> bool:
    """Return True if the current process holds the fcntl lock."""
    with _lock_lock:
        return _lock_depth > 0


@contextmanager
def _stage_outcomes_lock(
    lock_path: Path | None = None,
) -> Iterator[None]:
    """Acquire fcntl LOCK_EX on the stage-outcomes lock file.

    Per-process advisory lock (multiple processes serialize against each
    other; a single process re-entering is tracked via depth counter so
    nested context managers do not deadlock).
    """
    global _lock_depth
    path = lock_path or BOOSTING_STAGE_OUTCOMES_LOCK
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
                        f"Timed out acquiring boosting stage-outcomes lock "
                        f"after {LOCK_TIMEOUT_SEC}s at {path}"
                    ) from exc
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
    """Validate the structural shape of a stage-outcome row."""
    if not isinstance(record, dict):
        raise BoostingLedgerCorruptError(
            f"stage outcome record must be a dict, got "
            f"{type(record).__name__}"
        )
    schema_version = record.get("schema_version")
    if schema_version != BOOSTING_STAGE_OUTCOMES_SCHEMA_VERSION:
        raise BoostingLedgerCorruptError(
            f"schema_version must be {BOOSTING_STAGE_OUTCOMES_SCHEMA_VERSION!r}; "
            f"got {schema_version!r}"
        )
    if not isinstance(record.get("stage_id"), str):
        raise BoostingLedgerCorruptError(
            f"stage_id must be a string; got {record.get('stage_id')!r}"
        )
    if not isinstance(record.get("status"), str):
        raise BoostingLedgerCorruptError(
            f"status must be a string; got {record.get('status')!r}"
        )


def _quarantine_corrupt_ledger(path: Path) -> Path:
    """Move a corrupt ledger file to .corrupt.<utc> per Catalog #138 pattern."""
    if not path.exists():
        return path
    stamp = _dt.datetime.now(tz=_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.with_suffix(path.suffix + f".corrupt.{stamp}")
    os.rename(path, quarantine)
    return quarantine


def _atomic_write_ledger(
    rows: list[dict[str, Any]],
    path: Path | None = None,
) -> None:
    """Atomic write — unique .tmp.<uuid12> + fsync + os.replace.

    Runtime-asserts the caller holds ``_stage_outcomes_lock``. Per Catalog
    #132 + the canonical call_id_ledger pattern: writes are append-only —
    every existing row is preserved verbatim; only new rows are added by
    the caller before this function runs.
    """
    if not _stage_outcomes_lock_held():
        raise RuntimeError(
            "_atomic_write_ledger called WITHOUT holding _stage_outcomes_lock. "
            "This is a CONCURRENCY BUG: concurrent appends can silently drop "
            "rows. Use append_stage_outcome_locked() which owns the full "
            "lock-load-append-save cycle."
        )
    p = path or BOOSTING_STAGE_OUTCOMES_PATH
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


def append_stage_outcome_locked(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    quarantine_on_corrupt: bool = True,
) -> dict[str, Any]:
    """Append a single stage-outcome record under fcntl lock.

    Pattern (full-rewrite; sufficient for low-throughput design-time
    cadence):
      1. Acquire fcntl LOCK_EX
      2. STRICT-reload ledger from disk (raises BoostingLedgerCorruptError
         on ANY malformed row; quarantines if quarantine_on_corrupt=True)
      3. Append the new record (append-only per Catalog #132)
      4. Atomic write via unique .tmp.<uuid12> + fsync + os.replace

    Returns the record (augmented with timestamp / pid / host fields if
    they were absent in the input).
    """
    # Stamp record with provenance fields if not present
    record = dict(record)
    record.setdefault("schema_version", BOOSTING_STAGE_OUTCOMES_SCHEMA_VERSION)
    record.setdefault(
        "written_at_utc",
        _dt.datetime.now(tz=_dt.UTC).isoformat(),
    )
    record.setdefault("written_pid", os.getpid())
    record.setdefault("written_host", socket.gethostname())
    _validate_outcome_record(record)

    p = path or BOOSTING_STAGE_OUTCOMES_PATH
    lp = lock_path or BOOSTING_STAGE_OUTCOMES_LOCK

    with _stage_outcomes_lock(lp):
        try:
            rows = load_stage_outcomes_strict(p)
        except BoostingLedgerCorruptError as exc:
            if quarantine_on_corrupt:
                q = _quarantine_corrupt_ledger(p)
                raise BoostingLedgerCorruptError(
                    f"boosting stage-outcomes ledger at {p} was corrupt; "
                    f"quarantined to {q}. Append refused; operator must "
                    f"repair (see BoostingLedgerCorruptError docstring)."
                ) from exc
            raise

        new_rows = [*rows, record]
        _atomic_write_ledger(new_rows, p)
        return record


def load_stage_outcomes(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load all stage-outcome rows (lenient — skips malformed lines).

    Use for read-only queries / aggregations where occasional malformed
    rows are acceptable. For strict validation use
    :func:`load_stage_outcomes_strict`.
    """
    p = path or BOOSTING_STAGE_OUTCOMES_PATH
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


def load_stage_outcomes_strict(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load all stage-outcome rows; raise BoostingLedgerCorruptError on
    ANY malformed line.

    Per Catalog #138 fail-closed discipline — sister of
    ``call_id_ledger.load_call_ids_strict``. Used by
    ``append_stage_outcome_locked`` to refuse a write on top of a corrupt
    ledger.
    """
    p = path or BOOSTING_STAGE_OUTCOMES_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise BoostingLedgerCorruptError(
            f"failed to read boosting stage-outcomes ledger {p}: {exc}"
        ) from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BoostingLedgerCorruptError(
                f"line {lineno} of {p} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise BoostingLedgerCorruptError(
                f"line {lineno} of {p} is not a dict: type={type(obj).__name__}"
            )
        rows.append(obj)
    return rows
