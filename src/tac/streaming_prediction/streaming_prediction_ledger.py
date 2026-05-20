# SPDX-License-Identifier: MIT
"""Canonical streaming-prediction ledger — fcntl-locked JSONL append-only.

Per SLOT MG-5 of the 2026-05-19 master-gradient enhancement wave. Mirrors
the Catalog #245 Modal call_id ledger 4-layer pattern exactly:

- Layer 1 (this module): canonical fcntl-locked JSONL writer + readers
- Layer 2: cathedral_consumers/streaming_prediction_consumer (auto-discovered)
- Layer 3: Catalog #351 STRICT preflight gate refuses bare writes
- Layer 4: tools/realtime_prediction_dashboard.py (operator-facing tail)

Schema (one event per JSONL row)
────────────────────────────────
APPEND-ONLY per CLAUDE.md HISTORICAL_PROVENANCE (Catalog #110 / #113 /
#132). Rows are NEVER mutated; new events become NEW rows referencing
the same ``(subagent_id, substrate, epoch)`` triple. The full lifecycle
is reconstructable by ``query_by_substrate(substrate)``.

Required event_types::

    - "sampled"               — N-epoch streaming sample taken during training
    - "convergence_detected"  — Kalman posterior σ < user-tunable threshold
    - "stop_loss_triggered"   — Kalman posterior worsens > 3σ beyond initial

Per row::

    {
        "schema_version": 1,
        "event_type": "sampled" | "convergence_detected" | "stop_loss_triggered",
        "subagent_id": "claude_slot_X_substrate_dispatch_<utc>",
        "substrate": "substrate_z3_g1",
        "epoch": 100,
        "wall_clock_seconds": 1234.56,        # seconds since training start
        "m_sample_size": 8,                   # how many pairs sampled (for Taylor)
        "predicted_score": 0.198,             # Taylor-extrapolated score @ current weights
        "posterior_mean": 0.197,              # Kalman running mean
        "posterior_std": 0.008,               # Kalman running std (sqrt(variance))
        "posterior_n_observations": 5,        # samples folded into posterior
        "archive_sha256": null | "abc123...", # if archive built this epoch; null otherwise
        "predictor_model_id": "tac.master_gradient.predict_delta_s_taylor_v1",
        "provenance": {...},                  # canonical Provenance per Catalog #323
        "evidence_grade": "predicted",        # NEVER promotable; ALWAYS [predicted]
        "phantom_random_init": false,         # true if epoch==0 (Catalog #324 guard)
        "agent": "claude" | "codex" | "operator",
        "written_at_utc": "2026-05-19T...",
        "written_pid": 12345,
        "written_host": "primary",
    }

Path discipline
───────────────
- ``STREAMING_PREDICTION_LEDGER_PATH`` = ``.omx/state/streaming_predictions.jsonl``
  COMMITTED per HISTORICAL_PROVENANCE.
- Lock file ``.lock`` is gitignored.
- ``.tmp.<uuid12>`` files are gitignored.

Bare writes are FORBIDDEN
─────────────────────────
Per CLAUDE.md Catalog #131 + the new Catalog #351 STRICT gate. Every write
MUST acquire ``fcntl.flock(LOCK_EX)`` on the lock file + use a unique
``.tmp.<uuid12>`` OR the O_APPEND fast path.

Catalog #324 guard: streaming samples taken at epoch 0 carry
``phantom_random_init=true`` and the dashboard refuses to consume them
as score evidence; the field exists so the dashboard's stop-loss /
convergence detection can ignore the random-init sample without losing
the row from the audit trail.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
STREAMING_PREDICTION_LEDGER_PATH = REPO_ROOT / ".omx" / "state" / "streaming_predictions.jsonl"
STREAMING_PREDICTION_LEDGER_LOCK = STREAMING_PREDICTION_LEDGER_PATH.with_suffix(
    STREAMING_PREDICTION_LEDGER_PATH.suffix + ".lock"
)

SCHEMA_VERSION = 1

LOCK_TIMEOUT_SECONDS = 30

# Canonical event taxonomy.
EVENT_SAMPLED = "sampled"
EVENT_CONVERGENCE_DETECTED = "convergence_detected"
EVENT_STOP_LOSS_TRIGGERED = "stop_loss_triggered"

VALID_EVENT_TYPES = frozenset(
    {
        EVENT_SAMPLED,
        EVENT_CONVERGENCE_DETECTED,
        EVENT_STOP_LOSS_TRIGGERED,
    }
)


class StreamingPredictionLedgerCorruptError(RuntimeError):
    """Raised when the streaming-prediction ledger is corrupt and cannot
    be safely appended to.

    Sister of :class:`tac.deploy.modal.call_id_ledger.CallIdLedgerCorruptError`
    + Catalog #138 strict-load discipline. The append helpers raise this
    rather than silently overwriting the bad file, which would erase the
    historical audit trail of every streaming sample. The corrupt file is
    QUARANTINED to ``.corrupt.<utc>`` so the operator can inspect; the
    next ``register_streaming_sample`` will create a fresh empty ledger.
    """


# Thread-local lock-held depth (Catalog #140 sister pattern).
_ledger_lock_depth_tls = threading.local()


def _get_ledger_lock_depth() -> int:
    return int(getattr(_ledger_lock_depth_tls, "depth", 0))


def _set_ledger_lock_depth(value: int) -> None:
    _ledger_lock_depth_tls.depth = int(value)


def _ledger_lock_held() -> bool:
    return _get_ledger_lock_depth() > 0


@contextlib.contextmanager
def _ledger_lock(lock_path: Path | None = None):
    """Acquire fcntl exclusive lock on the ledger lock file.

    Re-entry within the same thread is counted (depth > 1); fcntl is only
    re-acquired on the 0->1 transition to avoid same-process deadlock.
    """
    p = lock_path or STREAMING_PREDICTION_LEDGER_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    depth = _get_ledger_lock_depth()
    if depth > 0:
        _set_ledger_lock_depth(depth + 1)
        try:
            yield None
        finally:
            _set_ledger_lock_depth(_get_ledger_lock_depth() - 1)
        return
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"could not acquire {p} within {LOCK_TIMEOUT_SECONDS}s"
                    ) from None
                time.sleep(0.05)
        _set_ledger_lock_depth(_get_ledger_lock_depth() + 1)
        try:
            yield fd
        finally:
            _set_ledger_lock_depth(_get_ledger_lock_depth() - 1)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 format."""
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _quarantine_corrupt_file(path: Path) -> Path:
    """Move ``path`` to ``path.corrupt.<utc>`` for forensic inspection."""
    if not path.exists():
        return path
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.with_suffix(path.suffix + f".corrupt.{ts}")
    counter = 0
    while quarantine.exists():
        counter += 1
        quarantine = path.with_suffix(path.suffix + f".corrupt.{ts}.{counter}")
    os.rename(path, quarantine)
    return quarantine


def load_streaming_samples(path: Path | None = None) -> list[dict[str, Any]]:
    """Lenient loader for dashboards / read-only callers.

    Malformed JSON lines are SKIPPED with no error.
    """
    p = path or STREAMING_PREDICTION_LEDGER_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
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


def load_streaming_samples_strict(path: Path | None = None) -> list[dict[str, Any]]:
    """Strict loader for mutating callers — raises on corrupt state.

    MUST be called from inside ``_ledger_lock`` by mutating callers.
    Returns ``[]`` if the path does not exist (bootstrap is normal).

    Raises:
        StreamingPredictionLedgerCorruptError: when the file exists and
            contains any malformed JSON line OR any non-dict root row.
    """
    p = path or STREAMING_PREDICTION_LEDGER_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise StreamingPredictionLedgerCorruptError(
            f"streaming-prediction ledger at {p} could not be read: {exc}"
        ) from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise StreamingPredictionLedgerCorruptError(
                f"streaming-prediction ledger at {p} line {lineno} contains "
                f"invalid JSON: {exc}. Mutating writes are refused to preserve "
                "the audit trail. Operator: inspect the file, then either fix "
                "it in place OR move it aside; the next register_streaming_sample "
                "will create a fresh empty file."
            ) from exc
        if not isinstance(row, dict):
            raise StreamingPredictionLedgerCorruptError(
                f"streaming-prediction ledger at {p} line {lineno} has non-dict "
                f"root (type={type(row).__name__}); expected JSON object."
            )
        rows.append(row)
    return rows


def _validate_record(record: dict[str, Any]) -> None:
    """Sanity-check a record before append. Raises ValueError on bad input."""
    schema_version = record.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}, got {schema_version!r}"
        )

    event_type = record.get("event_type")
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"event_type must be one of {sorted(VALID_EVENT_TYPES)!r}, got {event_type!r}"
        )

    substrate = record.get("substrate")
    if not isinstance(substrate, str) or not substrate.strip():
        raise ValueError("substrate must be a non-empty string")
    if any(c in substrate for c in ("\n", "\t", "\x1f")):
        raise ValueError("substrate must not contain newlines/tabs/0x1f")

    subagent_id = record.get("subagent_id")
    if not isinstance(subagent_id, str) or not subagent_id.strip():
        raise ValueError("subagent_id must be a non-empty string")

    epoch = record.get("epoch")
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")

    predicted_score = record.get("predicted_score")
    if predicted_score is not None and not isinstance(predicted_score, (int, float)):
        raise ValueError("predicted_score must be a number or null")

    evidence_grade = record.get("evidence_grade")
    if evidence_grade != "predicted":
        raise ValueError(
            f"evidence_grade must be 'predicted' for streaming samples "
            f"(non-promotable by construction per Catalog #323/#287); "
            f"got {evidence_grade!r}"
        )


def _truly_appending_write_locked(
    record: dict[str, Any],
    *,
    path: Path,
) -> tuple[int, int]:
    """Append a single row using POSIX O_APPEND under fcntl lock.

    Returns ``(byte_offset_of_new_row, total_file_size_after)``.

    Runtime-asserts the caller holds ``_ledger_lock``.
    """
    if not _ledger_lock_held():
        raise RuntimeError(
            "_truly_appending_write_locked called WITHOUT holding _ledger_lock"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        pre_size = os.fstat(fd).st_size
        n = os.write(fd, payload)
        if n != len(payload):
            raise OSError(
                f"short write ({n} of {len(payload)} bytes) to {path}; "
                "partial-write recovery is NOT supported because it would "
                "corrupt the audit trail."
            )
        os.fsync(fd)
        post_size = os.fstat(fd).st_size
    finally:
        os.close(fd)
    return pre_size, post_size


def _append_event_locked(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    quarantine_on_corrupt: bool = True,
) -> dict[str, Any]:
    """Append a single event under fcntl lock.

    Uses POSIX O_APPEND fast path. Pre-validates record + tail well-formedness.
    """
    _validate_record(record)
    p_path = path or STREAMING_PREDICTION_LEDGER_PATH
    l_path = lock_path or STREAMING_PREDICTION_LEDGER_LOCK

    with _ledger_lock(l_path):
        # Tail validation: if file exists and ends mid-line, refuse the append.
        if p_path.exists():
            try:
                # Quick sanity: load strict to detect any malformed row.
                load_streaming_samples_strict(p_path)
            except StreamingPredictionLedgerCorruptError as exc:
                if quarantine_on_corrupt:
                    quarantine = _quarantine_corrupt_file(p_path)
                    raise StreamingPredictionLedgerCorruptError(
                        f"streaming-prediction ledger at {p_path} was corrupt; "
                        f"quarantined to {quarantine}. Append refused."
                    ) from exc
                raise
        _truly_appending_write_locked(record, path=p_path)
    return record


def register_streaming_sample(
    *,
    subagent_id: str,
    substrate: str,
    epoch: int,
    wall_clock_seconds: float,
    m_sample_size: int,
    predicted_score: float,
    posterior_mean: float,
    posterior_std: float,
    posterior_n_observations: int,
    archive_sha256: str | None = None,
    predictor_model_id: str = "tac.master_gradient.predict_delta_s_taylor_v1",
    provenance: dict[str, Any] | None = None,
    agent: str = "claude",
    extra: dict[str, Any] | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Register a single streaming master-gradient sample.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog
    #323: the row carries ``evidence_grade="predicted"`` unconditionally and
    a canonical Provenance payload (built by the caller via
    ``tac.provenance.builders.build_provenance_for_predicted``).

    Catalog #324 guard: if ``epoch == 0`` the row is automatically tagged
    ``phantom_random_init=true`` so downstream dashboards can ignore the
    random-init sample without losing it from the audit trail.

    Returns the persisted record.
    """
    if epoch < 0:
        raise ValueError("epoch must be non-negative")

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_SAMPLED,
        "subagent_id": subagent_id,
        "substrate": substrate,
        "epoch": epoch,
        "wall_clock_seconds": float(wall_clock_seconds),
        "m_sample_size": int(m_sample_size),
        "predicted_score": float(predicted_score),
        "posterior_mean": float(posterior_mean),
        "posterior_std": float(posterior_std),
        "posterior_n_observations": int(posterior_n_observations),
        "archive_sha256": archive_sha256,
        "predictor_model_id": predictor_model_id,
        "provenance": provenance if provenance is not None else {},
        "evidence_grade": "predicted",
        "phantom_random_init": (epoch == 0),
        "agent": agent,
        "written_at_utc": _now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
    }
    if extra:
        for k, v in extra.items():
            if k in record:
                raise ValueError(f"extra field {k!r} cannot overwrite reserved key")
            record[k] = v

    return _append_event_locked(record, path=path, lock_path=lock_path)


def register_convergence_event(
    *,
    subagent_id: str,
    substrate: str,
    epoch: int,
    posterior_mean: float,
    posterior_std: float,
    convergence_threshold: float,
    rationale: str,
    agent: str = "claude",
    path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Append a convergence-detected event to the ledger.

    Emitted by the dashboard when Kalman posterior σ < threshold for the
    configured number of consecutive samples. NON-PROMOTABLE: this is a
    recommendation surface, not a score claim.
    """
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_CONVERGENCE_DETECTED,
        "subagent_id": subagent_id,
        "substrate": substrate,
        "epoch": int(epoch),
        "wall_clock_seconds": 0.0,  # event-only; not a sample
        "m_sample_size": 0,
        "predicted_score": float(posterior_mean),
        "posterior_mean": float(posterior_mean),
        "posterior_std": float(posterior_std),
        "posterior_n_observations": 0,
        "archive_sha256": None,
        "predictor_model_id": "tac.streaming_prediction.kalman_filter.detect_convergence",
        "provenance": {},
        "evidence_grade": "predicted",
        "phantom_random_init": False,
        "agent": agent,
        "convergence_threshold": float(convergence_threshold),
        "rationale": rationale,
        "written_at_utc": _now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
    }
    return _append_event_locked(record, path=path, lock_path=lock_path)


def register_stop_loss_event(
    *,
    subagent_id: str,
    substrate: str,
    epoch: int,
    posterior_mean: float,
    posterior_std: float,
    initial_estimate: float,
    deviation_sigma: float,
    rationale: str,
    agent: str = "claude",
    path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Append a stop-loss-triggered event to the ledger.

    Emitted by the dashboard when Kalman posterior_mean has worsened
    > ``deviation_sigma`` standard deviations beyond ``initial_estimate``.
    NON-PROMOTABLE: this is a recommendation surface; the operator
    decides whether to actually halt the dispatch.
    """
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_STOP_LOSS_TRIGGERED,
        "subagent_id": subagent_id,
        "substrate": substrate,
        "epoch": int(epoch),
        "wall_clock_seconds": 0.0,
        "m_sample_size": 0,
        "predicted_score": float(posterior_mean),
        "posterior_mean": float(posterior_mean),
        "posterior_std": float(posterior_std),
        "posterior_n_observations": 0,
        "archive_sha256": None,
        "predictor_model_id": "tac.streaming_prediction.kalman_filter.detect_stop_loss",
        "provenance": {},
        "evidence_grade": "predicted",
        "phantom_random_init": False,
        "agent": agent,
        "initial_estimate": float(initial_estimate),
        "deviation_sigma": float(deviation_sigma),
        "rationale": rationale,
        "written_at_utc": _now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
    }
    return _append_event_locked(record, path=path, lock_path=lock_path)


def latest_for_substrate(
    substrate: str, *, path: Path | None = None
) -> dict[str, Any] | None:
    """Return the most-recent ``sampled`` row for ``substrate``, or None.

    Uses the lenient loader; convergence/stop-loss events are skipped so
    callers always see the most-recent SAMPLE rather than a recommendation.
    """
    samples = load_streaming_samples(path=path)
    matched = [
        r for r in samples
        if r.get("substrate") == substrate and r.get("event_type") == EVENT_SAMPLED
    ]
    if not matched:
        return None
    # Sort by written_at_utc descending; fall back to epoch.
    def _sort_key(r: dict[str, Any]) -> tuple[str, int]:
        return (r.get("written_at_utc", ""), int(r.get("epoch", 0)))
    matched.sort(key=_sort_key, reverse=True)
    return matched[0]


def query_by_substrate(
    substrate: str, *, path: Path | None = None
) -> list[dict[str, Any]]:
    """Return all rows for ``substrate`` in chronological order (by epoch)."""
    rows = [r for r in load_streaming_samples(path=path) if r.get("substrate") == substrate]
    rows.sort(key=lambda r: (int(r.get("epoch", 0)), r.get("written_at_utc", "")))
    return rows


def query_all_post_utc(
    utc_str: str, *, path: Path | None = None
) -> list[dict[str, Any]]:
    """Return all rows whose ``written_at_utc`` >= ``utc_str``."""
    rows = [
        r for r in load_streaming_samples(path=path)
        if r.get("written_at_utc", "") >= utc_str
    ]
    return rows
