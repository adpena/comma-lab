# SPDX-License-Identifier: MIT
"""Fcntl-locked JSONL persistence for search strategy outcomes.

Per ``.omx/research/tac_search_namespace_design_20260517.md`` §7
production hardening: search outcomes optionally persist to a canonical
fcntl-locked JSONL ledger at
``.omx/state/search_strategy_outcomes.jsonl``.

Persistence is OPT-IN — ``ComposableSearchPipeline.run()`` /
``run_search_over_pipeline`` do NOT auto-persist (the canonical contract
treats run as a pure functional operation). Callers that want the audit
trail wrap the call::

    result = run_search_over_pipeline(pipeline, objective_fn)
    append_search_outcome_locked(result.to_dict())

Mirrors the canonical pattern in ``tac.boosting.persistence`` /
``tac.compress_time_optimization.persistence`` /
``tac.deploy.modal.call_id_ledger`` (Catalog #245) — fcntl LOCK_EX +
STRICT-load + unique .tmp + os.replace + APPEND-ONLY per Catalog #128 /
#131 / #132 sister discipline.

We use the SIMPLER full-rewrite path here (sufficient for low-throughput
design-time search outcomes; the OP-10 O(1) amortized path is overkill at
this cadence — documented decision, not premature optimization).
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

from tac.search.errors import SearchLedgerCorruptError

__all__ = [
    "SEARCH_STRATEGY_OUTCOMES_LOCK",
    "SEARCH_STRATEGY_OUTCOMES_PATH",
    "SEARCH_STRATEGY_OUTCOMES_SCHEMA_VERSION",
    "append_search_outcome_locked",
    "latest_best_score_by_strategy",
    "load_search_outcomes",
    "load_search_outcomes_strict",
    "query_outcomes_by_objective_label",
    "query_outcomes_by_strategy_id",
]

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
SEARCH_STRATEGY_OUTCOMES_PATH = (
    _REPO_ROOT / ".omx" / "state" / "search_strategy_outcomes.jsonl"
)
SEARCH_STRATEGY_OUTCOMES_LOCK = SEARCH_STRATEGY_OUTCOMES_PATH.with_suffix(
    SEARCH_STRATEGY_OUTCOMES_PATH.suffix + ".lock"
)
SEARCH_STRATEGY_OUTCOMES_SCHEMA_VERSION = "search_strategy_outcomes_v1"
LOCK_TIMEOUT_SEC = 30.0

# ---------------------------------------------------------------------------
# Per-process lock-state tracking
# ---------------------------------------------------------------------------

_lock_lock = threading.Lock()
_lock_depth = 0


def _search_outcomes_lock_held() -> bool:
    """Return True if the current process holds the fcntl lock."""
    with _lock_lock:
        return _lock_depth > 0


@contextmanager
def _search_outcomes_lock(
    lock_path: Path | None = None,
) -> Iterator[None]:
    """Acquire fcntl LOCK_EX on the search-outcomes lock file."""
    global _lock_depth
    path = lock_path or SEARCH_STRATEGY_OUTCOMES_LOCK
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
                        f"Timed out acquiring search strategy outcomes lock "
                        f"after {LOCK_TIMEOUT_SEC}s at {path}"
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
    """Validate the structural shape of a search-outcome row."""
    if not isinstance(record, dict):
        raise SearchLedgerCorruptError(
            f"search outcome record must be a dict, got "
            f"{type(record).__name__}"
        )
    schema_version = record.get("schema_version")
    if schema_version != SEARCH_STRATEGY_OUTCOMES_SCHEMA_VERSION:
        raise SearchLedgerCorruptError(
            f"schema_version must be "
            f"{SEARCH_STRATEGY_OUTCOMES_SCHEMA_VERSION!r}; "
            f"got {schema_version!r}"
        )
    if not isinstance(record.get("strategy_id"), str):
        raise SearchLedgerCorruptError(
            f"strategy_id must be a string; got {record.get('strategy_id')!r}"
        )
    # best_score must be numeric (NaN is forbidden per
    # ObjectiveFunctionError discipline)
    score = record.get("best_score")
    if score is None or isinstance(score, bool) or not isinstance(
        score, (int, float)
    ):
        raise SearchLedgerCorruptError(
            f"best_score must be a finite number; got {score!r}"
        )
    if score != score:  # NaN check
        raise SearchLedgerCorruptError(
            f"best_score is NaN; refused per ObjectiveFunctionError "
            f"discipline"
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

    Runtime-asserts the caller holds ``_search_outcomes_lock``. Per
    Catalog #132 + the canonical call_id_ledger pattern: writes are
    append-only — every existing row is preserved verbatim; only new rows
    are added by the caller before this function runs.
    """
    if not _search_outcomes_lock_held():
        raise RuntimeError(
            "_atomic_write_ledger called WITHOUT holding "
            "_search_outcomes_lock. This is a CONCURRENCY BUG: concurrent "
            "appends can silently drop rows. Use "
            "append_search_outcome_locked() which owns the full "
            "lock-load-append-save cycle."
        )
    p = path or SEARCH_STRATEGY_OUTCOMES_PATH
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


def append_search_outcome_locked(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    quarantine_on_corrupt: bool = True,
) -> dict[str, Any]:
    """Append a single search-outcome record under fcntl lock.

    Pattern (full-rewrite):
      1. Acquire fcntl LOCK_EX
      2. STRICT-reload ledger from disk (raises SearchLedgerCorruptError
         on ANY malformed row; quarantines if quarantine_on_corrupt=True)
      3. Append the new record (append-only per Catalog #132)
      4. Atomic write via unique .tmp.<uuid12> + fsync + os.replace

    Returns the record (augmented with timestamp / pid / host fields if
    they were absent in the input).
    """
    record = dict(record)
    record.setdefault(
        "schema_version", SEARCH_STRATEGY_OUTCOMES_SCHEMA_VERSION
    )
    record.setdefault(
        "written_at_utc",
        _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
    )
    record.setdefault("written_pid", os.getpid())
    record.setdefault("written_host", socket.gethostname())
    _validate_outcome_record(record)

    p = path or SEARCH_STRATEGY_OUTCOMES_PATH
    lp = lock_path or SEARCH_STRATEGY_OUTCOMES_LOCK

    with _search_outcomes_lock(lp):
        try:
            rows = load_search_outcomes_strict(p)
        except SearchLedgerCorruptError as exc:
            if quarantine_on_corrupt:
                q = _quarantine_corrupt_ledger(p)
                raise SearchLedgerCorruptError(
                    f"search strategy outcomes ledger at {p} was corrupt; "
                    f"quarantined to {q}. Append refused; operator must "
                    f"repair (see SearchLedgerCorruptError docstring)."
                ) from exc
            raise

        new_rows = [*rows, record]
        _atomic_write_ledger(new_rows, p)
        return record


def load_search_outcomes(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load all search-outcome rows (lenient — skips malformed lines)."""
    p = path or SEARCH_STRATEGY_OUTCOMES_PATH
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


def load_search_outcomes_strict(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load all search-outcome rows; raise SearchLedgerCorruptError on
    ANY malformed line.

    Per Catalog #138 fail-closed discipline. Used by
    ``append_search_outcome_locked`` to refuse a write on top of a
    corrupt ledger.
    """
    p = path or SEARCH_STRATEGY_OUTCOMES_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise SearchLedgerCorruptError(
            f"failed to read search strategy outcomes ledger {p}: {exc}"
        ) from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SearchLedgerCorruptError(
                f"line {lineno} of {p} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise SearchLedgerCorruptError(
                f"line {lineno} of {p} is not a dict: "
                f"type={type(obj).__name__}"
            )
        rows.append(obj)
    return rows


# ---------------------------------------------------------------------------
# Canonical query helpers (per design memo §11 observability surface facet 4)
# ---------------------------------------------------------------------------


def query_outcomes_by_strategy_id(
    strategy_id: str, *, path: Path | None = None
) -> list[dict[str, Any]]:
    """Return all outcome rows with a given strategy_id, in append order."""
    return [
        row
        for row in load_search_outcomes(path)
        if row.get("strategy_id") == strategy_id
    ]


def query_outcomes_by_objective_label(
    objective_function_label: str, *, path: Path | None = None
) -> list[dict[str, Any]]:
    """Return all outcome rows with a given objective label, in append order."""
    return [
        row
        for row in load_search_outcomes(path)
        if row.get("objective_function_label") == objective_function_label
    ]


def latest_best_score_by_strategy(
    strategy_id: str, *, path: Path | None = None
) -> float | None:
    """Return the best_score from the most-recent outcome row for the
    given strategy_id, or None if no row exists."""
    rows = query_outcomes_by_strategy_id(strategy_id, path=path)
    if not rows:
        return None
    last = rows[-1]
    score = last.get("best_score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return float(score)
    return None
