# SPDX-License-Identifier: MIT
"""Canonical Modal spend reader — chunked, fail-closed, receipt-backed.

WHY THIS EXISTS (measured 2026-08-16, arm ``ddm_mb1``, commit ``76f1d5de3f``):
the operator's binding Modal budget (task #381, hard cap ``$20``, set
2026-07-09) was answerable only from prose.  The three surfaces disagreed by
more than an order of magnitude:

===========================  ==============  =========================
surface                      reported        vs provider billing
===========================  ==============  =========================
session prose                ``~$6.9``       **2.7x low**
``modal_call_id_ledger``     ``~$1.74``      **10.7x low**
``modal billing report``     ``$18.6124``    authority
===========================  ==============  =========================

Three real defects produced the gap, none of them estimation slop (all MEASURED
by ``ddm_mb1`` at source):

1. ``cost_actual_usd`` mixes measured entries, rate-derived estimates, and hand
   entries under one name.  One row carries its own confession
   (``cost_estimate_source = modal_elapsed_seconds_x_configured_hourly_rate``),
   so deriving a rate from the ledger recovers the rate that was assumed.
2. ``modal_elapsed_seconds`` starts AFTER image pull and container start, so it
   is a lower bound on billable time.  Any ``elapsed x rate`` model under-prices
   by construction.
3. CPU and Memory bill separately and were **41.8%** of spend.  A GPU-seconds
   model sees none of it.

So the ledger cannot be repaired into a spend total: billing is app-by-day,
calls are not apps.  The cure is to ask the provider, and to make its answer a
durable record instead of a memo paragraph.

THE FAILURE MODE THIS MODULE EXISTS TO REFUSE
---------------------------------------------
``modal billing report`` is rate-limited at the workspace level.  ``ddm_mb1``
hit the limit, redirected stdout to a file without checking the return code,
and briefly computed a cross-check from a **silently empty file** -- i.e. a
spent cap read as ``$0.00``.  That is the most dangerous shape a spend reader
can have, because it reports an exhausted budget as untouched.

Therefore every read here is fail-closed.  An unreadable window is reported
``UNREADABLE`` and never as a number.  This mirrors ``tac.process_liveness``:
blindness and emptiness demand opposite responses, so they must be
distinguishable at the source.  ``[]`` (a real day with no usage) is READABLE
and totals ``0``; empty stdout is UNREADABLE and totals nothing at all.

Guards applied to every window, in order:

* non-zero ``returncode`` -> ``nonzero_returncode``
* rate-limit marker on stderr -> ``rate_limited``
* empty stdout -> ``empty_stdout``
* stdout that is not a JSON list -> ``rate_limited`` if it carries the marker,
  else ``unparseable_json`` / ``unexpected_payload``
* a row with a missing or non-numeric ``cost`` -> ``malformed_row``

CHUNKING
--------
``modal billing report`` refuses a daily range spanning more than 31 days
(MEASURED: ``rc=1``, stdout **empty**, stderr ``"Billing report range limit
exceeded. Daily reports cannot span more than 31 days."``).  It is not
truncated -- it is refused.  Any requested window is therefore split into
sub-31-day half-open intervals with no overlap and no gap, matching the
provider's own ``--start`` inclusive / ``--end`` exclusive semantics.

MONEY IS ``Decimal``
--------------------
Costs arrive as 8-decimal strings.  They are summed as :class:`decimal.Decimal`
so the total is exact; the float mirror in the receipt is a convenience, not the
authority.

Sisters: ``tac.process_liveness`` (the tri-state that refuses to encode an
absent measurement as a negative result) and the repo law that a measurement
which could not be TAKEN is never a RESULT.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.jsonl_store import append_locked_jsonl  # noqa: E402

__all__ = [
    "CAP_WINDOW_START",
    "DEFAULT_CAP_USD",
    "MAX_DAILY_SPAN_DAYS",
    "READABLE",
    "UNREADABLE",
    "BillingUnreadable",
    "SpendReading",
    "WindowReading",
    "chunk_window",
    "ledger_lower_bound_usd",
    "main",
    "read_billing_window",
    "read_spend",
]

READABLE = "readable"
UNREADABLE = "unreadable"

#: Provider limit, MEASURED 2026-08-16: a daily report spanning >31 days is
#: refused with rc=1 and empty stdout, not truncated.
MAX_DAILY_SPAN_DAYS = 31

#: Operator constraint, task #381 (2026-07-09): Modal spend <= $20 total.
DEFAULT_CAP_USD = Decimal("20.00")
CAP_WINDOW_START = date(2026, 7, 9)

DEFAULT_RECEIPT_PATH = REPO_ROOT / ".omx" / "state" / "modal_spend_receipts.jsonl"
DEFAULT_LEDGER_PATH = REPO_ROOT / ".omx" / "state" / "modal_call_id_ledger.jsonl"

RECEIPT_SCHEMA = "modal_spend_report_v1"

#: Substrings that mark a workspace rate-limit refusal.  Matched
#: case-insensitively against stderr always, and against stdout ONLY when
#: stdout is not a valid JSON list -- so a digit sequence inside a cost string
#: can never be mistaken for an HTTP status.
_RATE_LIMIT_MARKERS = ("rate limit", "ratelimit", "too many requests", "429")

_SUBPROCESS_TIMEOUT_S = 300.0

#: Seconds to wait before the first rate-limit retry; doubles each attempt.
RATE_LIMIT_BACKOFF_BASE_S = 20.0
DEFAULT_CLI_RETRIES = 3


class BillingUnreadable(RuntimeError):
    """A billing window could not be READ.

    Carrying the reason, the exact command, and the provider's own words is the
    point: the caller must be able to distinguish "the provider said zero" from
    "we could not ask", and to reproduce the failure.
    """

    def __init__(
        self,
        reason: str,
        *,
        command: Sequence[str],
        returncode: int | None = None,
        stderr_excerpt: str = "",
        stdout_excerpt: str = "",
    ) -> None:
        super().__init__(f"modal billing window UNREADABLE ({reason})")
        self.reason = reason
        self.command = list(command)
        self.returncode = returncode
        self.stderr_excerpt = stderr_excerpt
        self.stdout_excerpt = stdout_excerpt

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": UNREADABLE,
            "reason": self.reason,
            "command": self.command,
            "returncode": self.returncode,
            "stderr_excerpt": self.stderr_excerpt,
            "stdout_excerpt": self.stdout_excerpt,
        }


@dataclass(frozen=True)
class WindowReading:
    """One successfully READ sub-window.  Never constructed on failure."""

    start: date
    end: date
    total_usd: Decimal
    row_count: int
    command: tuple[str, ...]
    rows: tuple[dict[str, Any], ...] = field(default=(), repr=False)

    def as_dict(self, *, include_rows: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "start": self.start.isoformat(),
            "end_exclusive": self.end.isoformat(),
            "total_usd": str(self.total_usd),
            "row_count": self.row_count,
            "command": list(self.command),
        }
        if include_rows:
            out["rows"] = list(self.rows)
        return out


@dataclass(frozen=True)
class SpendReading:
    """A whole requested window, READ end to end.  Never partial.

    ``read_spend`` raises rather than returning this object when any sub-window
    is unreadable, so a ``SpendReading`` in hand always means every dollar in
    the range was actually observed.
    """

    start: date
    end: date
    windows: tuple[WindowReading, ...]
    cap_usd: Decimal

    @property
    def total_usd(self) -> Decimal:
        return sum((w.total_usd for w in self.windows), Decimal("0"))

    @property
    def headroom_usd(self) -> Decimal:
        return self.cap_usd - self.total_usd

    @property
    def over_cap(self) -> bool:
        return self.total_usd > self.cap_usd

    @property
    def cap_consumed_fraction(self) -> Decimal:
        if self.cap_usd == 0:
            return Decimal("0")
        return self.total_usd / self.cap_usd

    def as_dict(self, *, include_rows: bool = False) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "status": READABLE,
            "start": self.start.isoformat(),
            "end_exclusive": self.end.isoformat(),
            "total_usd": str(self.total_usd),
            "total_usd_float": float(self.total_usd),
            "cap_usd": str(self.cap_usd),
            "headroom_usd": str(self.headroom_usd),
            "headroom_usd_float": float(self.headroom_usd),
            "cap_consumed_fraction": float(self.cap_consumed_fraction),
            "over_cap": self.over_cap,
            "window_count": len(self.windows),
            "row_count": sum(w.row_count for w in self.windows),
            "windows": [w.as_dict(include_rows=include_rows) for w in self.windows],
            "commands": [list(w.command) for w in self.windows],
        }


def chunk_window(start: date, end: date, *, max_days: int = MAX_DAILY_SPAN_DAYS) -> list[tuple[date, date]]:
    """Split ``[start, end)`` into half-open sub-windows of at most ``max_days``.

    No overlap and no gap: each chunk's ``end`` is the next chunk's ``start``,
    matching the provider's inclusive-start / exclusive-end semantics.
    """
    if max_days <= 0:
        raise ValueError(f"max_days must be positive; got {max_days!r}")
    if end <= start:
        raise ValueError(f"end must be strictly after start (end is exclusive); got {start} .. {end}")
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + timedelta(days=max_days), end)
        chunks.append((cursor, stop))
        cursor = stop
    return chunks


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_S,
        cwd=str(REPO_ROOT),
    )


def _modal_executable() -> str:
    venv_modal = REPO_ROOT / ".venv" / "bin" / "modal"
    return str(venv_modal) if venv_modal.is_file() else "modal"


def _has_rate_limit_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


def _excerpt(text: str, limit: int = 600) -> str:
    stripped = text.strip()
    return stripped if len(stripped) <= limit else stripped[:limit] + "..."


def read_billing_window(
    start: date,
    end: date,
    *,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
    modal_executable: str | None = None,
    retries: int = 0,
    sleeper: Callable[[float], None] | None = None,
) -> WindowReading:
    """READ one billing window, or raise :class:`BillingUnreadable`.

    Never returns a total it did not observe.  ``[]`` from the provider is a
    real zero and returns a reading; an empty or unparseable response is a
    refusal.

    ``retries`` applies to ``rate_limited`` ONLY, with exponential backoff.  The
    workspace billing endpoint really does refuse a multi-chunk read (MEASURED
    2026-08-16: chunk 1 of a 39-day window succeeded, chunk 2 came back
    ``rc=1`` / ``Rate limit exceeded for workspace billing report requests``),
    so a reader without backoff cannot answer the cap question at all.  Every
    other refusal is deterministic and is NOT retried — hammering a malformed
    response would only burn the same rate limit.  The library default is ``0``
    so callers opt into waiting; the CLI defaults to ``3``.
    """
    if retries < 0:
        raise ValueError(f"retries must be non-negative; got {retries!r}")
    if end <= start:
        raise ValueError(f"end must be strictly after start (end is exclusive); got {start} .. {end}")
    span_days = (end - start).days
    if span_days > MAX_DAILY_SPAN_DAYS:
        raise ValueError(
            f"window {start}..{end} spans {span_days} days; the provider refuses "
            f">{MAX_DAILY_SPAN_DAYS}. Call chunk_window() first."
        )

    command = (
        modal_executable or _modal_executable(),
        "billing",
        "report",
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--resolution",
        "d",
        "--json",
    )
    run = runner or _default_runner
    sleep = sleeper if sleeper is not None else time.sleep
    for attempt in range(retries + 1):
        try:
            return _read_billing_window_once(start, end, command=command, run=run)
        except BillingUnreadable as exc:
            if exc.reason != "rate_limited" or attempt == retries:
                raise
            sleep(RATE_LIMIT_BACKOFF_BASE_S * (2**attempt))
    raise AssertionError("unreachable: the loop either returns or raises")  # pragma: no cover


def _read_billing_window_once(
    start: date,
    end: date,
    *,
    command: Sequence[str],
    run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]],
) -> WindowReading:
    """One shot at one window.  All five guards live here."""
    try:
        completed = run(command)
    except subprocess.TimeoutExpired as exc:
        raise BillingUnreadable("subprocess_timeout", command=command, stderr_excerpt=str(exc)) from exc
    except OSError as exc:
        raise BillingUnreadable("subprocess_oserror", command=command, stderr_excerpt=str(exc)) from exc

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    # Guard 1: the provider told us it failed.  Do not parse a failure.
    if completed.returncode != 0:
        reason = "rate_limited" if _has_rate_limit_marker(stderr) else "nonzero_returncode"
        raise BillingUnreadable(
            reason,
            command=command,
            returncode=completed.returncode,
            stderr_excerpt=_excerpt(stderr),
            stdout_excerpt=_excerpt(stdout),
        )

    # Guard 2: a rate-limit frame can arrive alongside rc=0.  stderr is scanned
    # unconditionally because no legitimate payload lives there.
    if _has_rate_limit_marker(stderr):
        raise BillingUnreadable(
            "rate_limited",
            command=command,
            returncode=completed.returncode,
            stderr_excerpt=_excerpt(stderr),
            stdout_excerpt=_excerpt(stdout),
        )

    # Guard 3: empty stdout is BLINDNESS, not zero spend.  This is the exact
    # shape that once reported a spent cap as untouched.
    if not stdout.strip():
        raise BillingUnreadable(
            "empty_stdout",
            command=command,
            returncode=completed.returncode,
            stderr_excerpt=_excerpt(stderr),
        )

    # Guard 4: anything that is not the expected JSON list is a refusal.  Only
    # here do we scan stdout for the marker, so cost digits cannot false-match.
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        reason = "rate_limited" if _has_rate_limit_marker(stdout) else "unparseable_json"
        raise BillingUnreadable(
            reason,
            command=command,
            returncode=completed.returncode,
            stderr_excerpt=_excerpt(stderr),
            stdout_excerpt=_excerpt(stdout),
        ) from exc
    if not isinstance(payload, list):
        reason = "rate_limited" if _has_rate_limit_marker(stdout) else "unexpected_payload"
        raise BillingUnreadable(
            reason,
            command=command,
            returncode=completed.returncode,
            stderr_excerpt=_excerpt(stderr),
            stdout_excerpt=_excerpt(stdout),
        )

    # Guard 5: a row we cannot price makes the whole window unreadable.  A
    # partial sum is worse than a refusal -- it looks like an answer.
    total = Decimal("0")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict) or "cost" not in raw:
            raise BillingUnreadable(
                "malformed_row",
                command=command,
                returncode=completed.returncode,
                stdout_excerpt=_excerpt(f"row {index}: {raw!r}"),
            )
        row: dict[str, Any] = {str(key): value for key, value in raw.items()}
        raw_cost = row["cost"]
        try:
            cost = Decimal(str(raw_cost))
        except (InvalidOperation, ValueError) as exc:
            raise BillingUnreadable(
                "malformed_row",
                command=command,
                returncode=completed.returncode,
                stdout_excerpt=_excerpt(f"row {index} cost={raw_cost!r}"),
            ) from exc
        if not cost.is_finite():
            raise BillingUnreadable(
                "malformed_row",
                command=command,
                returncode=completed.returncode,
                stdout_excerpt=_excerpt(f"row {index} cost={raw_cost!r} is not finite"),
            )
        total += cost
        rows.append(row)

    return WindowReading(
        start=start,
        end=end,
        total_usd=total,
        row_count=len(rows),
        command=tuple(command),
        rows=tuple(rows),
    )


def read_spend(
    start: date,
    end: date,
    *,
    cap_usd: Decimal = DEFAULT_CAP_USD,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
    modal_executable: str | None = None,
    retries: int = 0,
    sleeper: Callable[[float], None] | None = None,
) -> SpendReading:
    """READ ``[start, end)`` in provider-legal chunks, or raise.

    Fail-closed by construction: the first unreadable chunk aborts the whole
    reading, so a returned total is never a partial sum wearing a whole number's
    name.
    """
    windows = [
        read_billing_window(
            chunk_start,
            chunk_end,
            runner=runner,
            modal_executable=modal_executable,
            retries=retries,
            sleeper=sleeper,
        )
        for chunk_start, chunk_end in chunk_window(start, end)
    ]
    return SpendReading(start=start, end=end, windows=tuple(windows), cap_usd=cap_usd)


def ledger_lower_bound_usd(
    start: date,
    end: date,
    *,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> dict[str, Any]:
    """Sum the ledger's ``cost_actual_usd`` over the window as a LOWER BOUND.

    This exists so the ledger's number can never again be read as a total
    without its authority label beside it.  The ledger measured ``$1.74`` for a
    window the provider billed at ``$18.61`` -- 10.7x low -- because
    ``cost_actual_usd`` is mostly null, and where it is not null it mixes
    measurements with rate-derived estimates.  It is per-call attribution, not
    a spend total, and it is NOT authority.

    THE FOLD IS FIELD-WISE, and that is load-bearing (MEASURED 2026-08-16).
    The ledger is append-only and its rows are PARTIAL: the dispatch row carries
    ``dispatched_at_utc``, later outcome rows omit it.  Folding whole-row
    latest-wins therefore DESTROYS the dispatch timestamp on **136 of 275**
    call_ids, and the window filter then silently drops them -- 22 calls visible
    instead of 63, and ``$0.15`` instead of ``$1.74``.  A wrong fold on an
    append-only store reads as a smaller world, not as an error.  Taking the
    latest NON-NULL value per field reproduces the independently-derived counts
    exactly (63 calls, ``$1.7400``, arm ``ddm_mb1``).
    """
    if not ledger_path.is_file():
        return {
            "ledger_readable": False,
            "ledger_path": str(ledger_path),
            "note": "ledger file not found",
        }
    merged: dict[str, dict[str, Any]] = {}
    malformed = 0
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        call_id = row.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            continue
        accumulator = merged.setdefault(call_id, {})
        for key, value in row.items():
            if value is not None:
                accumulator[key] = value

    total = Decimal("0")
    priced = 0
    in_window = 0
    undated = 0
    for row in merged.values():
        dispatched = row.get("dispatched_at_utc")
        if not isinstance(dispatched, str):
            undated += 1
            continue
        try:
            stamp = datetime.fromisoformat(dispatched.replace("Z", "+00:00")).date()
        except ValueError:
            malformed += 1
            continue
        if not (start <= stamp < end):
            continue
        in_window += 1
        cost = row.get("cost_actual_usd")
        if cost is None:
            continue
        try:
            total += Decimal(str(cost))
        except (InvalidOperation, ValueError):
            malformed += 1
            continue
        priced += 1

    return {
        "ledger_readable": True,
        "ledger_path": str(ledger_path),
        "ledger_is_authority": False,
        "ledger_semantics": (
            "LOWER BOUND ONLY: cost_actual_usd is mostly null and where non-null "
            "mixes measured entries with rate-derived estimates; billing is the "
            "authority for any spend total."
        ),
        "ledger_fold": "latest_non_null_value_per_field",
        "ledger_sum_usd": str(total),
        "ledger_calls_total": len(merged),
        "ledger_calls_in_window": in_window,
        "ledger_calls_priced": priced,
        "ledger_calls_unpriced": in_window - priced,
        "ledger_calls_without_dispatch_timestamp": undated,
        "ledger_malformed_lines": malformed,
    }


def _parse_date(text: str) -> date:
    return date.fromisoformat(text)


def _parse_usd(text: str) -> Decimal:
    """Parse a dollar amount for argparse.

    ``Decimal`` cannot be used as an argparse ``type`` directly: it raises
    ``decimal.InvalidOperation``, which inherits from ``ArithmeticError`` and NOT
    from ``ValueError``, so argparse does not catch it and a typo produces a raw
    traceback instead of a usage error.
    """
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a valid dollar amount") from exc
    if not value.is_finite():
        raise argparse.ArgumentTypeError(f"{text!r} is not a finite dollar amount")
    return value


def _default_end() -> date:
    """Tomorrow (UTC).

    ``--end`` is EXCLUSIVE and the provider reports full intervals only, so
    today's usage is included exactly when the end bound is tomorrow.
    """
    return datetime.now(UTC).date() + timedelta(days=1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modal_spend_report",
        description=(
            "Canonical Modal spend reader: chunked to the provider's 31-day "
            "limit, fail-closed on any unreadable window, receipt-backed."
        ),
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=CAP_WINDOW_START,
        help=(f"Inclusive start date (YYYY-MM-DD). Default {CAP_WINDOW_START.isoformat()}, when the $20 cap was set."),
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=None,
        help="EXCLUSIVE end date (YYYY-MM-DD). Default tomorrow UTC, so today counts.",
    )
    parser.add_argument(
        "--cap-usd",
        type=_parse_usd,
        default=DEFAULT_CAP_USD,
        help=f"Spend cap to report headroom against (default {DEFAULT_CAP_USD}).",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=DEFAULT_RECEIPT_PATH,
        help=f"Append the typed receipt here (default {DEFAULT_RECEIPT_PATH}).",
    )
    parser.add_argument(
        "--no-receipt",
        action="store_true",
        help="Do not append a receipt (read-only probe).",
    )
    parser.add_argument(
        "--no-ledger-crosscheck",
        action="store_true",
        help="Skip the ledger lower-bound cross-check.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_CLI_RETRIES,
        help=(
            "Rate-limit retries per window, exponential backoff from "
            f"{RATE_LIMIT_BACKOFF_BASE_S:.0f}s (default {DEFAULT_CLI_RETRIES}). "
            "Applies to rate limits ONLY; deterministic failures never retry."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout.")
    parser.add_argument(
        "--fail-over-cap",
        action="store_true",
        help="Exit rc=3 when measured spend exceeds the cap.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Exit 0 READABLE / 2 UNREADABLE / 3 readable-but-over-cap.

    ``2`` is deliberately distinct from ``0``: a caller must never treat "we
    could not read the meter" as "the meter reads nothing".
    """
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    end = args.end if args.end is not None else _default_end()

    try:
        reading = read_spend(args.start, end, cap_usd=args.cap_usd, retries=args.retries)
    except BillingUnreadable as exc:
        payload = exc.as_dict()
        payload["schema"] = RECEIPT_SCHEMA
        payload["start"] = args.start.isoformat()
        payload["end_exclusive"] = end.isoformat()
        payload["read_at_utc"] = datetime.now(UTC).isoformat()
        if not args.no_receipt:
            append_locked_jsonl(args.receipt, payload)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            # Deliberately on STDOUT, not only stderr.  The defect this tool
            # exists to extinct was a caller redirecting stdout to a file,
            # ignoring the return code, and reading the empty file as $0.00.
            # A refusal must be LOUD in the same stream a total would occupy.
            print("MODAL SPEND: UNREADABLE — no number emitted.")
            print(f"  reason:     {exc.reason}")
            print(f"  window:     {args.start} .. {end} (end exclusive)")
            print(f"  command:    {' '.join(exc.command)}")
            print(f"  returncode: {exc.returncode}")
            if exc.stderr_excerpt:
                print(f"  provider:   {exc.stderr_excerpt}")
            print(
                "  This is blindness, NOT $0.00. Retry (the workspace billing "
                "endpoint is rate-limited) before any spend decision."
            )
            print(
                f"MODAL SPEND: UNREADABLE ({exc.reason}) — no number emitted, rc=2.",
                file=sys.stderr,
            )
        return 2
    except ValueError as exc:
        print(f"MODAL SPEND: bad request — {exc}", file=sys.stderr)
        return 2

    payload = reading.as_dict()
    payload["read_at_utc"] = datetime.now(UTC).isoformat()
    if not args.no_ledger_crosscheck:
        payload["ledger_crosscheck"] = ledger_lower_bound_usd(args.start, end)
    if not args.no_receipt:
        append_locked_jsonl(args.receipt, payload)
        payload["receipt_path"] = str(args.receipt)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"MODAL SPEND {reading.start} .. {reading.end} (end exclusive)")
        print(f"  measured spend:  ${reading.total_usd}")
        print(f"  cap:             ${reading.cap_usd}")
        print(f"  headroom:        ${reading.headroom_usd}")
        print(f"  cap consumed:    {float(reading.cap_consumed_fraction) * 100:.1f}%")
        print(f"  windows read:    {len(reading.windows)} ({payload['row_count']} rows)")
        crosscheck = payload.get("ledger_crosscheck")
        if crosscheck and crosscheck.get("ledger_readable"):
            print(
                f"  ledger sum:      ${crosscheck['ledger_sum_usd']} "
                f"(LOWER BOUND, not authority; "
                f"{crosscheck['ledger_calls_unpriced']} of "
                f"{crosscheck['ledger_calls_in_window']} calls unpriced)"
            )
        if not args.no_receipt:
            print(f"  receipt:         {args.receipt}")
        if reading.over_cap:
            print("  OVER CAP — every further dispatch breaches the operator constraint.")

    if args.fail_over_cap and reading.over_cap:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
