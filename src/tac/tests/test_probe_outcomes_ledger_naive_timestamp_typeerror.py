# SPDX-License-Identifier: MIT
"""Pin the naive/aware timestamp TypeError class in the probe-outcomes ledger.

MEASURED BUG (2026-08-16, ddm_rd1g). ``query_expired_deferrals`` computed
``now - datetime.fromisoformat(expires_at)`` under an ``except ValueError``
guard. When either operand was naive the subtraction raised ``TypeError: can't
subtract offset-naive and offset-aware datetimes``, which that guard does not
catch, so the deferral nag query CRASHED instead of degrading to
``days_expired=None``.

Two independent reachable paths, both reproduced against real inputs:

1. A row whose ``expires_at_utc`` carries no offset (naive).
2. A caller passing a naive ``now_utc`` -- reproduced against the LIVE
   728-row ledger, where every stored row was already ``Z``-suffixed. The
   parameter is annotated ``datetime``, which does not exclude naive.

Path 2 is why the fix is a coercion and not a widened ``except``: the ledger
had no single point that guaranteed awareness, so the crash could arrive from
the caller even with a perfectly clean ledger.

The WRITE side manufactured the path-1 input: ``_compute_expires_at_utc``
returned ``'2026-07-01T00:00:00.000000'`` (no ``Z``) for a naive
``adjudicated_at_utc``. Writer and reader are pinned together here because
fixing only the reader would leave the ledger accumulating naive rows.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from tac.probe_outcomes_ledger import (
    VERDICT_DEFER,
    _as_utc,
    _compute_expires_at_utc,
    query_expired_deferrals,
)


def _write_rows(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "probe_outcomes.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_naive_expires_at_row_does_not_raise_typeerror(tmp_path: Path) -> None:
    """Path 1: a naive stored ``expires_at_utc`` must not crash the query."""
    path = _write_rows(
        tmp_path,
        [
            {
                "probe_id": "p_naive_row",
                "verdict": VERDICT_DEFER,
                # No 'Z', no offset -- exactly what the pre-fix writer emitted.
                "expires_at_utc": "2026-06-01T00:00:00",
            }
        ],
    )

    rows = query_expired_deferrals(
        now_utc=_dt.datetime(2026, 8, 16, tzinfo=_dt.UTC), path=path
    )

    assert [r["probe_id"] for r in rows] == ["p_naive_row"]
    # Real arithmetic, not a swallowed None: 2026-06-01 -> 2026-08-16 is 76 days.
    assert rows[0]["days_expired"] == 76


def test_naive_now_utc_argument_does_not_raise_typeerror(tmp_path: Path) -> None:
    """Path 2: a naive ``now_utc`` from the caller must not crash the query.

    This is the path that fired against the live ledger, whose rows were all
    already ``Z``-suffixed -- so a reader-only ledger cleanup would not have
    prevented it.
    """
    path = _write_rows(
        tmp_path,
        [
            {
                "probe_id": "p_aware_row",
                "verdict": VERDICT_DEFER,
                "expires_at_utc": "2026-06-01T00:00:00.000000Z",
            }
        ],
    )

    rows = query_expired_deferrals(
        now_utc=_dt.datetime(2026, 8, 16),  # naive on purpose
        path=path,
    )

    assert [r["probe_id"] for r in rows] == ["p_aware_row"]
    assert rows[0]["days_expired"] == 76


def test_unparseable_expiry_still_degrades_to_none(tmp_path: Path) -> None:
    """The pre-existing ValueError contract is preserved, not replaced."""
    path = _write_rows(
        tmp_path,
        [
            {
                "probe_id": "p_garbage",
                "verdict": VERDICT_DEFER,
                # Sorts before any real 'now', so it passes the expiry filter
                # and reaches the parse.
                "expires_at_utc": "0000-not-a-timestamp",
            }
        ],
    )

    rows = query_expired_deferrals(
        now_utc=_dt.datetime(2026, 8, 16, tzinfo=_dt.UTC), path=path
    )

    assert [r["probe_id"] for r in rows] == ["p_garbage"]
    assert rows[0]["days_expired"] is None


def test_writer_never_emits_a_naive_expires_at_utc() -> None:
    """The write side must not manufacture the poison the reader chokes on."""
    naive_in = _compute_expires_at_utc("2026-06-01T00:00:00")
    aware_in = _compute_expires_at_utc("2026-06-01T00:00:00Z")

    assert naive_in.endswith("Z"), f"writer emitted a naive expiry: {naive_in!r}"
    # A naive input is read as UTC, so both spellings land on the same instant.
    assert naive_in == aware_in


@pytest.mark.parametrize(
    "value",
    [
        _dt.datetime(2026, 6, 1),
        _dt.datetime(2026, 6, 1, tzinfo=_dt.UTC),
        _dt.datetime(2026, 6, 1, 2, tzinfo=_dt.timezone(_dt.timedelta(hours=2))),
    ],
)
def test_as_utc_always_returns_the_same_aware_instant(value: _dt.datetime) -> None:
    """The single coercion point is what makes mixed-awareness unreachable."""
    coerced = _as_utc(value)

    assert coerced.tzinfo is not None
    assert coerced.utcoffset() == _dt.timedelta(0)
    assert coerced == _dt.datetime(2026, 6, 1, tzinfo=_dt.UTC)
    # The property the callers actually depend on: subtraction never raises.
    assert isinstance(_dt.datetime.now(_dt.UTC) - coerced, _dt.timedelta)
