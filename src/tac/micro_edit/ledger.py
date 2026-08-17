# SPDX-License-Identifier: MIT
"""The edit-outcome ledger -- one append-only store for every measured micro-edit.

This is the "train" half's corpus (charter section 2). Every row is ONE edit object
measured against ONE base on ONE instrument. The row carries the base identity and the
CODER identity because both are regime variables: a delta measured under the old
range coder does not transfer to the HPAC-context stream (the cross-regime
constant-transfer genus), and a delta measured against CP135 does not transfer to rr4.

Rows are append-only JSONL under an ``fcntl`` exclusive lock (the canonical
``.omx/state`` store pattern). Nothing is ever mutated in place; a corrected row is a
NEW row with ``supersedes`` set, so the history stays auditable.

ALWAYS KEEP THE PAYLOAD: a row that claims a realized measurement must name a
persisted payload (``payload_path`` + ``payload_sha256``) or explicitly declare
``payload_status="none"`` with a reason. :func:`validate_row` refuses a realized row
that measured bytes it did not keep.

STORES CONSULTED
----------------
* ``.omx/research/ddm_eu4_fresh_eyes_fractal_composition_20260813.md`` -- qs2/re1
  banked numbers and the union-gating law.
* ``.omx/research/ddm_rr4_t4_verdict_pointer_move_20260817.md`` -- the live base and
  the standing recompile hazard for the banked offsets.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "LEDGER_PATH",
    "CoderRegime",
    "EditOutcomeRow",
    "LedgerValidationError",
    "append_row",
    "load_rows",
    "validate_row",
]

LEDGER_PATH: Path = Path(".omx/research/micro_edit_outcome_ledger.jsonl")
"""Canonical ledger location.

Deliberately under ``.omx/research`` and not ``.omx/state``: this is a small, durable,
COMMITTED corpus that the next arm inherits and the ranker trains on, which is exactly
what the repo policy says to track there. ``.omx/state/*.jsonl`` is gitignored live
machine state -- a corpus parked there would be invisible to every future session,
which is the orphaned-signal failure this ledger exists to prevent.
"""

CoderRegime = Literal["pre_rr4_range", "rr4_hpac_context", "unknown"]
"""Which entropy-coder regime the row's bytes were measured under.

``pre_rr4_range``    -- the coder in force before the rr4 HPAC-context recode.
``rr4_hpac_context`` -- the LIVE coder (archive sha 35ac2b9b...).
``unknown``          -- provenance could not establish it; such a row may seed the
                        ranker but may NEVER be summed into a byte projection.
"""


class LedgerValidationError(ValueError):
    """Raised when a row violates a ledger invariant."""


@dataclass(frozen=True)
class EditOutcomeRow:
    """One measured micro-edit outcome.

    Deltas are stored as strings so ``Decimal`` round-trips exactly through JSON;
    ``float`` would silently truncate the 1e-7 quantities the engine decides on.
    """

    row_id: str
    family: str
    arm_id: str
    support_desc: str
    support_size: int

    # --- realized measurement (strings -> Decimal) ---
    d_seg_delta: str
    d_pose_delta: str
    bytes_delta: int
    net_seg_flips: str

    # --- regime identity: WITHOUT these a row is untransferable ---
    base_label: str
    base_archive_sha256: str | None
    base_archive_bytes: int
    coder_regime: CoderRegime
    instrument: str

    # --- custody ---
    payload_path: str | None
    payload_sha256: str | None
    payload_status: Literal["retained", "none"]
    payload_absent_reason: str | None

    # --- provenance + lifecycle ---
    provenance: str
    realized: bool
    written_at_utc: str
    supersedes: str | None = None
    notes: str = ""
    features: dict[str, Any] = field(default_factory=dict)

    def as_decimals(self) -> dict[str, Decimal]:
        return {
            "d_seg_delta": Decimal(self.d_seg_delta),
            "d_pose_delta": Decimal(self.d_pose_delta),
            "net_seg_flips": Decimal(self.net_seg_flips),
        }


def validate_row(row: EditOutcomeRow) -> None:
    """Refuse rows that would poison the corpus. Raises :class:`LedgerValidationError`."""
    if not row.row_id or not row.family or not row.arm_id:
        raise LedgerValidationError("row_id, family, arm_id are all required")
    if row.support_size < 0:
        raise LedgerValidationError("support_size must be non-negative")
    try:
        Decimal(row.d_seg_delta)
        Decimal(row.d_pose_delta)
        Decimal(row.net_seg_flips)
    except Exception as exc:
        raise LedgerValidationError(f"delta fields must parse as Decimal: {exc}") from exc
    if row.coder_regime not in ("pre_rr4_range", "rr4_hpac_context", "unknown"):
        raise LedgerValidationError(f"unknown coder_regime {row.coder_regime!r}")
    if not row.instrument:
        raise LedgerValidationError("instrument label is required (axis honesty)")
    if row.payload_status == "retained":
        if not row.payload_path or not row.payload_sha256:
            raise LedgerValidationError(
                "payload_status='retained' requires payload_path AND payload_sha256"
            )
    elif row.payload_status == "none":
        if not row.payload_absent_reason:
            raise LedgerValidationError(
                "payload_status='none' requires payload_absent_reason "
                "(ALWAYS KEEP THE PAYLOAD -- an absent payload must be justified)"
            )
    else:
        raise LedgerValidationError(f"bad payload_status {row.payload_status!r}")
    if row.realized and row.bytes_delta != 0 and row.payload_status == "none":
        raise LedgerValidationError(
            "a realized row that moved bytes MUST retain the payload it measured "
            "(ALWAYS KEEP THE PAYLOAD, P0)"
        )


def append_row(row: EditOutcomeRow, path: Path | None = None) -> None:
    """Append one validated row under an exclusive lock."""
    validate_row(row)
    target = Path(path) if path is not None else LEDGER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(row), sort_keys=True, ensure_ascii=False)
    with open(target, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_rows(path: Path | None = None, *, skip_superseded: bool = True) -> list[EditOutcomeRow]:
    """Load rows, newest-wins on ``supersedes`` chains when ``skip_superseded``."""
    target = Path(path) if path is not None else LEDGER_PATH
    if not target.exists():
        return []
    rows: list[EditOutcomeRow] = []
    with open(target, encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(EditOutcomeRow(**json.loads(line)))
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    if not skip_superseded:
        return rows
    superseded = {r.supersedes for r in rows if r.supersedes}
    return [r for r in rows if r.row_id not in superseded]


def iter_by_family(rows: list[EditOutcomeRow]) -> Iterator[tuple[str, list[EditOutcomeRow]]]:
    """Group rows by family, families in first-appearance order."""
    order: list[str] = []
    buckets: dict[str, list[EditOutcomeRow]] = {}
    for row in rows:
        if row.family not in buckets:
            buckets[row.family] = []
            order.append(row.family)
        buckets[row.family].append(row)
    for family in order:
        yield family, buckets[family]


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically (tmp + os.replace) so a crash cannot truncate a store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Not a context manager on purpose: the file must survive close() so os.replace
    # can move it into place atomically. delete=False + explicit close is the pattern.
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    )
    try:
        json.dump(obj, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)
