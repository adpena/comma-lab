# SPDX-License-Identifier: MIT
"""Controls for the canonical Modal spend reader (``tools/modal_spend_report``).

Per design philosophy P4 (no meter without a canary) this suite is built as a
matched pair, not as coverage:

* **positive control** — a window the provider bills reads NONZERO, so a reader
  that silently returned zero everywhere would fail here.
* **negative control** — every way a billing read can fail REFUSES.  Each of
  these asserts ``pytest.raises``, never a field value, because the exact bug
  being extincted is *returning ``0.00`` from a failed read*.  A test that
  asserted ``total == 0`` would PASS on the broken code; a test that demands an
  exception cannot.

The distinction that carries the most weight is the last pair: ``[]`` from the
provider is a real day with no usage and must READ as zero, while empty stdout
is blindness and must REFUSE.  Collapsing those two is precisely how a spent cap
gets reported as untouched.

The live provider control is opt-in (``TAC_MODAL_SPEND_LIVE_CONTROL=1``) because
the workspace billing endpoint is rate-limited and a test suite must not hammer
it.  It was executed by hand at landing; see the arm's memo.
"""

from __future__ import annotations

import importlib
import itertools
import json
import os
import subprocess
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

msr = importlib.import_module("tools.modal_spend_report")


def _completed(*, returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["modal", "billing", "report"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _runner(completed: subprocess.CompletedProcess[str]):
    def run(_command):
        return completed

    return run


# --------------------------------------------------------------------------
# POSITIVE CONTROL — a billed window reads nonzero
# --------------------------------------------------------------------------


def test_positive_control_billed_window_reads_nonzero() -> None:
    payload = json.dumps(
        [
            {
                "object_id": "ap-1",
                "description": "comma-ddm-x",
                "environment": "main",
                "interval_start": "2026-08-15T00:00:00",
                "cost": "0.02095760",
            },
            {
                "object_id": "ap-2",
                "description": "comma-auth-eval",
                "environment": "main",
                "interval_start": "2026-08-15T00:00:00",
                "cost": "0.08916814",
            },
        ]
    )
    reading = msr.read_billing_window(
        date(2026, 8, 15),
        date(2026, 8, 16),
        runner=_runner(_completed(stdout=payload)),
    )
    assert reading.total_usd == Decimal("0.11012574")
    assert reading.total_usd > 0
    assert reading.row_count == 2


def test_positive_control_totals_are_exact_decimal_not_float() -> None:
    """Eight-decimal costs must sum exactly; float accumulation drifts."""
    payload = json.dumps([{"cost": "0.10000001"}] * 3)
    reading = msr.read_billing_window(date(2026, 8, 15), date(2026, 8, 16), runner=_runner(_completed(stdout=payload)))
    assert reading.total_usd == Decimal("0.30000003")


# --------------------------------------------------------------------------
# NEGATIVE CONTROLS — every failure REFUSES, none returns a number
# --------------------------------------------------------------------------


def test_negative_control_rate_limit_with_nonzero_rc_refuses() -> None:
    with pytest.raises(msr.BillingUnreadable) as excinfo:
        msr.read_billing_window(
            date(2026, 8, 15),
            date(2026, 8, 16),
            runner=_runner(
                _completed(
                    returncode=1,
                    stdout="",
                    stderr="Rate limit exceeded for workspace billing report requests",
                )
            ),
        )
    assert excinfo.value.reason == "rate_limited"


def test_negative_control_rate_limit_wearing_rc_zero_refuses() -> None:
    """The dangerous shape: a rate-limit frame that LOOKS like a success."""
    with pytest.raises(msr.BillingUnreadable) as excinfo:
        msr.read_billing_window(
            date(2026, 8, 15),
            date(2026, 8, 16),
            runner=_runner(
                _completed(
                    returncode=0,
                    stdout="",
                    stderr="Error: Rate limit exceeded for workspace billing report requests",
                )
            ),
        )
    assert excinfo.value.reason == "rate_limited"


def test_negative_control_rate_limit_frame_on_stdout_refuses() -> None:
    with pytest.raises(msr.BillingUnreadable) as excinfo:
        msr.read_billing_window(
            date(2026, 8, 15),
            date(2026, 8, 16),
            runner=_runner(_completed(returncode=0, stdout="╭─ Error ─╮\n│ Rate limit exceeded │\n╰──╯")),
        )
    assert excinfo.value.reason == "rate_limited"


def test_negative_control_nonzero_rc_without_marker_refuses() -> None:
    """The measured >31-day refusal: rc=1, EMPTY stdout, range-limit stderr."""
    with pytest.raises(msr.BillingUnreadable) as excinfo:
        msr.read_billing_window(
            date(2026, 7, 1),
            date(2026, 7, 20),
            runner=_runner(
                _completed(
                    returncode=1,
                    stdout="",
                    stderr=("Billing report range limit exceeded. Daily reports cannot span more than 31 days."),
                )
            ),
        )
    assert excinfo.value.reason == "nonzero_returncode"
    assert excinfo.value.returncode == 1


def test_negative_control_empty_stdout_with_rc_zero_refuses_never_zero() -> None:
    """THE canonical bug: a silently empty response read as ``$0.00``."""
    with pytest.raises(msr.BillingUnreadable) as excinfo:
        msr.read_billing_window(date(2026, 8, 15), date(2026, 8, 16), runner=_runner(_completed(stdout="   \n")))
    assert excinfo.value.reason == "empty_stdout"


def test_negative_control_unparseable_stdout_refuses() -> None:
    with pytest.raises(msr.BillingUnreadable) as excinfo:
        msr.read_billing_window(date(2026, 8, 15), date(2026, 8, 16), runner=_runner(_completed(stdout="not json")))
    assert excinfo.value.reason == "unparseable_json"


def test_negative_control_non_list_payload_refuses() -> None:
    with pytest.raises(msr.BillingUnreadable) as excinfo:
        msr.read_billing_window(
            date(2026, 8, 15),
            date(2026, 8, 16),
            runner=_runner(_completed(stdout='{"error": "nope"}')),
        )
    assert excinfo.value.reason == "unexpected_payload"


@pytest.mark.parametrize(
    "row",
    [
        {"object_id": "ap-1"},  # no cost key at all
        {"cost": "not-a-number"},
        {"cost": None},
        {"cost": "NaN"},
        "a bare string, not a row",
    ],
)
def test_negative_control_malformed_row_refuses_whole_window(row) -> None:
    """A partial sum is worse than a refusal: it looks like an answer."""
    payload = json.dumps([{"cost": "1.00"}, row])
    with pytest.raises(msr.BillingUnreadable) as excinfo:
        msr.read_billing_window(date(2026, 8, 15), date(2026, 8, 16), runner=_runner(_completed(stdout=payload)))
    assert excinfo.value.reason == "malformed_row"


def test_negative_control_timeout_refuses() -> None:
    def run(_command):
        raise subprocess.TimeoutExpired(cmd="modal", timeout=1.0)

    with pytest.raises(msr.BillingUnreadable) as excinfo:
        msr.read_billing_window(date(2026, 8, 15), date(2026, 8, 16), runner=run)
    assert excinfo.value.reason == "subprocess_timeout"


def test_negative_control_missing_binary_refuses() -> None:
    def run(_command):
        raise FileNotFoundError("modal: command not found")

    with pytest.raises(msr.BillingUnreadable) as excinfo:
        msr.read_billing_window(date(2026, 8, 15), date(2026, 8, 16), runner=run)
    assert excinfo.value.reason == "subprocess_oserror"


# --------------------------------------------------------------------------
# THE PAIR THAT MATTERS — real zero READS, blindness REFUSES
# --------------------------------------------------------------------------


def test_empty_list_is_a_real_zero_and_reads() -> None:
    reading = msr.read_billing_window(date(2026, 8, 15), date(2026, 8, 16), runner=_runner(_completed(stdout="[]")))
    assert reading.total_usd == Decimal("0")
    assert reading.row_count == 0


def test_blindness_and_real_zero_are_distinguishable() -> None:
    """Same numeric outcome, opposite epistemic status. Never collapse them."""
    zero = msr.read_billing_window(date(2026, 8, 15), date(2026, 8, 16), runner=_runner(_completed(stdout="[]")))
    assert zero.total_usd == Decimal("0")
    with pytest.raises(msr.BillingUnreadable):
        msr.read_billing_window(date(2026, 8, 15), date(2026, 8, 16), runner=_runner(_completed(stdout="")))


# --------------------------------------------------------------------------
# CHUNKING — the provider's 31-day limit
# --------------------------------------------------------------------------


def test_chunk_window_respects_31_day_limit_with_no_gap_or_overlap() -> None:
    start, end = date(2026, 7, 9), date(2026, 8, 17)
    chunks = msr.chunk_window(start, end)
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    for span_start, span_end in chunks:
        assert (span_end - span_start).days <= msr.MAX_DAILY_SPAN_DAYS
    for earlier, later in itertools.pairwise(chunks):
        assert earlier[1] == later[0]  # no gap, no overlap


def test_chunk_window_single_chunk_when_short() -> None:
    assert msr.chunk_window(date(2026, 8, 1), date(2026, 8, 3)) == [(date(2026, 8, 1), date(2026, 8, 3))]


def test_chunk_window_rejects_empty_or_inverted_range() -> None:
    with pytest.raises(ValueError):
        msr.chunk_window(date(2026, 8, 3), date(2026, 8, 3))
    with pytest.raises(ValueError):
        msr.chunk_window(date(2026, 8, 4), date(2026, 8, 3))


def test_read_billing_window_refuses_an_over_limit_span_before_spending_a_call() -> None:
    """Guard the provider limit locally; do not burn a rate-limited request."""
    with pytest.raises(ValueError, match="31"):
        msr.read_billing_window(date(2026, 7, 1), date(2026, 8, 16), runner=_runner(_completed()))


def test_read_spend_chunks_and_sums_across_windows() -> None:
    calls: list[list[str]] = []

    def run(command):
        calls.append(list(command))
        return _completed(stdout=json.dumps([{"cost": "1.50"}]))

    reading = msr.read_spend(date(2026, 7, 9), date(2026, 8, 17), runner=run)
    assert len(calls) == 2
    assert reading.total_usd == Decimal("3.00")
    assert reading.cap_usd == msr.DEFAULT_CAP_USD
    assert reading.headroom_usd == Decimal("17.00")
    assert reading.over_cap is False


def test_read_spend_aborts_whole_reading_when_any_chunk_is_unreadable() -> None:
    """Fail-closed: no partial total may escape wearing a whole number's name."""
    seen: list[int] = []

    def run(_command):
        seen.append(1)
        if len(seen) == 1:
            return _completed(stdout=json.dumps([{"cost": "5.00"}]))
        return _completed(returncode=1, stderr="Rate limit exceeded")

    with pytest.raises(msr.BillingUnreadable):
        msr.read_spend(date(2026, 7, 9), date(2026, 8, 17), runner=run)


def test_over_cap_is_reported_not_clamped() -> None:
    reading = msr.read_spend(
        date(2026, 8, 1),
        date(2026, 8, 3),
        cap_usd=Decimal("20.00"),
        runner=_runner(_completed(stdout=json.dumps([{"cost": "25.00"}]))),
    )
    assert reading.over_cap is True
    assert reading.headroom_usd == Decimal("-5.00")


# --------------------------------------------------------------------------
# RETRY — backoff on rate limits ONLY, and it still refuses when they persist
# --------------------------------------------------------------------------


def test_retry_recovers_from_a_transient_rate_limit() -> None:
    attempts: list[int] = []
    slept: list[float] = []

    def run(_command):
        attempts.append(1)
        if len(attempts) < 3:
            return _completed(returncode=1, stderr="Rate limit exceeded")
        return _completed(stdout=json.dumps([{"cost": "2.50"}]))

    reading = msr.read_billing_window(
        date(2026, 8, 15),
        date(2026, 8, 16),
        runner=run,
        retries=3,
        sleeper=slept.append,
    )
    assert reading.total_usd == Decimal("2.50")
    assert len(attempts) == 3
    assert slept == [20.0, 40.0]  # exponential, only between failed attempts


def test_retry_exhausted_still_refuses_never_returns_zero() -> None:
    slept: list[float] = []
    with pytest.raises(msr.BillingUnreadable) as excinfo:
        msr.read_billing_window(
            date(2026, 8, 15),
            date(2026, 8, 16),
            runner=_runner(_completed(returncode=1, stderr="Rate limit exceeded")),
            retries=2,
            sleeper=slept.append,
        )
    assert excinfo.value.reason == "rate_limited"
    assert slept == [20.0, 40.0]


def test_deterministic_failures_are_not_retried() -> None:
    """Hammering a malformed response only burns the same rate limit."""
    attempts: list[int] = []

    def run(_command):
        attempts.append(1)
        return _completed(stdout="not json")

    with pytest.raises(msr.BillingUnreadable):
        msr.read_billing_window(
            date(2026, 8, 15),
            date(2026, 8, 16),
            runner=run,
            retries=5,
            sleeper=lambda _s: None,
        )
    assert len(attempts) == 1


def test_negative_retries_rejected() -> None:
    with pytest.raises(ValueError):
        msr.read_billing_window(date(2026, 8, 15), date(2026, 8, 16), runner=_runner(_completed()), retries=-1)


def test_benign_429_in_stderr_is_not_a_rate_limit() -> None:
    """A bare ``429`` substring matched app ids and log counts.

    A false rate-limit classification is not merely noisy: it discards a
    READABLE window and triggers backoff, and under the CLI's default retries it
    cascades into refusing the whole report.
    """
    reading = msr.read_billing_window(
        date(2026, 8, 15),
        date(2026, 8, 16),
        runner=_runner(
            _completed(
                returncode=0,
                stdout=json.dumps([{"cost": "1.00"}]),
                stderr="WARNING: app ap-429abc emitted 429 log lines",
            )
        ),
    )
    assert reading.total_usd == Decimal("1.00")


def test_real_rate_limit_phrase_still_refuses() -> None:
    """The paired control for the test above: the MEASURED frame still refuses."""
    with pytest.raises(msr.BillingUnreadable) as excinfo:
        msr.read_billing_window(
            date(2026, 8, 15),
            date(2026, 8, 16),
            runner=_runner(
                _completed(
                    returncode=0,
                    stdout=json.dumps([{"cost": "1.00"}]),
                    stderr="Rate limit exceeded for workspace billing report requests.",
                )
            ),
        )
    assert excinfo.value.reason == "rate_limited"


# --------------------------------------------------------------------------
# DEFAULT END — exclusive, so today must still be counted
# --------------------------------------------------------------------------


def test_default_end_is_tomorrow_so_today_is_counted() -> None:
    """``--end`` is EXCLUSIVE. Using today would silently drop today's spend."""
    from datetime import UTC, datetime, timedelta

    today = datetime.now(UTC).date()
    assert msr._default_end() == today + timedelta(days=1)
    assert msr._default_end() != today


# --------------------------------------------------------------------------
# LEDGER CROSS-CHECK — labelled a lower bound, never authority
# --------------------------------------------------------------------------


def test_ledger_window_end_is_exclusive(tmp_path: Path) -> None:
    """The cross-check window must match the billing window exactly."""
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "call_id": "fc-in",
                        "dispatched_at_utc": "2026-08-31T23:00:00Z",
                        "cost_actual_usd": 1.0,
                    }
                ),
                json.dumps(
                    {
                        "call_id": "fc-out",
                        "dispatched_at_utc": "2026-09-01T00:00:00Z",
                        "cost_actual_usd": 500.0,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    out = msr.ledger_lower_bound_usd(date(2026, 8, 1), date(2026, 9, 1), ledger_path=ledger)
    assert out["ledger_calls_in_window"] == 1, "the end bound must be EXCLUSIVE"
    assert Decimal(out["ledger_sum_usd"]) == Decimal("1.0")


# --------------------------------------------------------------------------
# CONTRADICTION — the ledger cross-examines the billing read for free
# --------------------------------------------------------------------------


def test_contradiction_fires_when_billing_says_zero_but_the_ledger_priced_calls() -> None:
    """The last path to a false $0.00: every window legitimately returning ``[]``."""
    crosscheck = {"ledger_readable": True, "ledger_sum_usd": "1.74", "ledger_calls_priced": 8}
    message = msr.ledger_contradiction(Decimal("0"), crosscheck)
    assert message is not None
    assert "cannot bill to nothing" in message


def test_contradiction_does_not_fire_when_the_ledger_merely_over_estimates() -> None:
    """Configured rates were MEASURED over-predicting 2.01x; that is not a refutation."""
    crosscheck = {"ledger_readable": True, "ledger_sum_usd": "5.00", "ledger_calls_priced": 3}
    assert msr.ledger_contradiction(Decimal("2.50"), crosscheck) is None


def test_contradiction_silent_when_the_ledger_prices_nothing() -> None:
    crosscheck = {"ledger_readable": True, "ledger_sum_usd": "0", "ledger_calls_priced": 0}
    assert msr.ledger_contradiction(Decimal("0"), crosscheck) is None
    assert msr.ledger_contradiction(Decimal("0"), {"ledger_readable": False}) is None


def test_cli_refuses_a_zero_total_that_the_ledger_contradicts(capsys, tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "call_id": "fc-a",
                "dispatched_at_utc": "2026-08-15T00:00:00Z",
                "cost_actual_usd": 1.25,
            }
        ),
        encoding="utf-8",
    )
    original_runner = msr._default_runner
    msr._default_runner = _runner(_completed(stdout="[]"))
    try:
        rc = msr.main(
            [
                "--start",
                "2026-08-15",
                "--end",
                "2026-08-16",
                "--no-receipt",
                "--retries",
                "0",
                "--ledger",
                str(ledger),
            ]
        )
    finally:
        msr._default_runner = original_runner
    captured = capsys.readouterr()
    assert rc == 2
    assert "UNREADABLE" in captured.out
    assert "contradicted_by_ledger" in captured.out


# --------------------------------------------------------------------------
# RECEIPT — re-derivability rests on the commands, and a write failure must
# never destroy an expensive rate-limited read
# --------------------------------------------------------------------------


def test_receipt_pins_the_commands_and_windows_that_make_it_re_derivable(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.jsonl"
    original = msr._default_runner
    msr._default_runner = _runner(_completed(stdout=json.dumps([{"cost": "1.00"}])))
    try:
        msr.main(
            [
                "--start",
                "2026-07-09",
                "--end",
                "2026-08-17",
                "--receipt",
                str(receipt),
                "--no-ledger-crosscheck",
                "--retries",
                "0",
            ]
        )
    finally:
        msr._default_runner = original
    row = json.loads(receipt.read_text(encoding="utf-8").strip())
    assert len(row["commands"]) == 2, "one command per chunk, so the read is reproducible"
    for command in row["commands"]:
        assert "billing" in command and "--json" in command
        assert "--start" in command and "--end" in command
    assert len(row["windows"]) == 2
    assert [w["start"] for w in row["windows"]] == ["2026-07-09", "2026-08-09"]
    assert [w["end_exclusive"] for w in row["windows"]] == ["2026-08-09", "2026-08-17"]
    assert row["receipt_path"] == str(receipt)


def test_a_receipt_write_failure_does_not_destroy_the_reading(capsys, tmp_path: Path) -> None:
    """The read is expensive and rate-limited; never lose it to a bad path."""
    unwritable = tmp_path / "a-file-not-a-dir"
    unwritable.write_text("blocking", encoding="utf-8")
    original = msr._default_runner
    msr._default_runner = _runner(_completed(stdout=json.dumps([{"cost": "7.00"}])))
    try:
        rc = msr.main(
            [
                "--start",
                "2026-08-15",
                "--end",
                "2026-08-16",
                "--receipt",
                str(unwritable / "receipt.jsonl"),
                "--no-ledger-crosscheck",
                "--retries",
                "0",
            ]
        )
    finally:
        msr._default_runner = original
    captured = capsys.readouterr()
    assert rc == 0
    assert "$7.00" in captured.out
    assert "NOT WRITTEN" in captured.out


def test_cli_rejects_a_non_positive_cap() -> None:
    for bad in ("0", "-5"):
        with pytest.raises(SystemExit) as excinfo:
            msr.main(["--cap-usd", bad, "--no-receipt"])
        assert excinfo.value.code == 2


def test_ledger_crosscheck_labels_itself_non_authoritative(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "call_id": "fc-a",
                        "dispatched_at_utc": "2026-08-10T00:00:00Z",
                        "cost_actual_usd": 1.25,
                    }
                ),
                json.dumps(
                    {
                        "call_id": "fc-b",
                        "dispatched_at_utc": "2026-08-11T00:00:00Z",
                        "cost_actual_usd": None,
                    }
                ),
                json.dumps(
                    {
                        "call_id": "fc-c",
                        "dispatched_at_utc": "2026-01-01T00:00:00Z",
                        "cost_actual_usd": 99.0,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    out = msr.ledger_lower_bound_usd(date(2026, 8, 1), date(2026, 9, 1), ledger_path=ledger)
    assert out["ledger_is_authority"] is False
    assert out["ledger_sum_usd"] == "1.25"
    assert out["ledger_calls_in_window"] == 2  # fc-c is outside the window
    assert out["ledger_calls_unpriced"] == 1


def test_ledger_fold_is_field_wise_not_whole_row(tmp_path: Path) -> None:
    """The fold that once hid 41 of 63 calls.

    Ledger rows are PARTIAL: the dispatch row carries ``dispatched_at_utc``,
    outcome rows omit it.  Whole-row latest-wins destroys the timestamp, the
    window filter drops the call, and the sum comes back SMALLER with no error
    anywhere.  MEASURED on the live ledger: 22 calls / ``$0.15`` under the wrong
    fold vs 63 calls / ``$1.74`` under the right one.
    """
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "\n".join(
            [
                # dispatch row: has the timestamp, no cost yet
                json.dumps(
                    {
                        "call_id": "fc-a",
                        "dispatched_at_utc": "2026-08-10T00:00:00Z",
                        "cost_actual_usd": None,
                    }
                ),
                # outcome row: has the cost, and NO timestamp field at all
                json.dumps({"call_id": "fc-a", "status": "harvested", "cost_actual_usd": 3.50}),
            ]
        ),
        encoding="utf-8",
    )
    out = msr.ledger_lower_bound_usd(date(2026, 8, 1), date(2026, 9, 1), ledger_path=ledger)
    assert out["ledger_calls_in_window"] == 1, "whole-row fold would drop this call entirely"
    assert Decimal(out["ledger_sum_usd"]) == Decimal("3.50")
    assert out["ledger_fold"] == "latest_non_null_value_per_field"


def test_ledger_crosscheck_latest_row_wins(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "call_id": "fc-a",
                        "dispatched_at_utc": "2026-08-10T00:00:00Z",
                        "cost_actual_usd": 1.00,
                    }
                ),
                json.dumps(
                    {
                        "call_id": "fc-a",
                        "dispatched_at_utc": "2026-08-10T00:00:00Z",
                        "cost_actual_usd": 2.00,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    out = msr.ledger_lower_bound_usd(date(2026, 8, 1), date(2026, 9, 1), ledger_path=ledger)
    # The ledger stores JSON floats, so the faithful conversion is Decimal(str(f))
    # — shortest round-trip repr, exact, but not the literal's trailing zeros.
    assert Decimal(out["ledger_sum_usd"]) == Decimal("2.00")
    assert out["ledger_calls_in_window"] == 1


def test_ledger_crosscheck_missing_file_is_not_a_zero(tmp_path: Path) -> None:
    out = msr.ledger_lower_bound_usd(date(2026, 8, 1), date(2026, 9, 1), ledger_path=tmp_path / "absent.jsonl")
    assert out["ledger_readable"] is False
    assert "ledger_sum_usd" not in out


# --------------------------------------------------------------------------
# CLI — exit codes keep blindness distinct from zero
# --------------------------------------------------------------------------


def test_cli_unreadable_exits_2_and_prints_no_total(capsys, tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.jsonl"
    original = msr._default_runner
    msr._default_runner = _runner(_completed(returncode=1, stderr="Rate limit exceeded"))
    try:
        rc = msr.main(
            [
                "--start",
                "2026-08-15",
                "--end",
                "2026-08-16",
                "--json",
                "--receipt",
                str(receipt),
                "--retries",
                "0",
            ]
        )
    finally:
        msr._default_runner = original
    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == msr.UNREADABLE
    assert payload["reason"] == "rate_limited"
    assert "total_usd" not in payload
    assert json.loads(receipt.read_text(encoding="utf-8").strip())["status"] == msr.UNREADABLE


def test_cli_unreadable_shouts_on_stdout_not_only_stderr(capsys, tmp_path: Path) -> None:
    """The original defect was a caller reading a redirected stdout as ``$0.00``.

    A refusal must therefore occupy the SAME stream a total would, so an empty
    capture is impossible even when the return code is ignored.
    """
    original = msr._default_runner
    msr._default_runner = _runner(_completed(returncode=1, stderr="Rate limit exceeded"))
    try:
        rc = msr.main(["--start", "2026-08-15", "--end", "2026-08-16", "--no-receipt", "--retries", "0"])
    finally:
        msr._default_runner = original
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out.strip(), "stdout must not be empty on a refusal"
    assert "UNREADABLE" in captured.out
    assert "NOT $0.00" in captured.out
    assert "UNREADABLE" in captured.err


def test_cli_readable_exits_0_and_writes_receipt(capsys, tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.jsonl"
    original = msr._default_runner
    msr._default_runner = _runner(_completed(stdout=json.dumps([{"cost": "1.00"}])))
    try:
        rc = msr.main(
            [
                "--start",
                "2026-08-15",
                "--end",
                "2026-08-16",
                "--json",
                "--receipt",
                str(receipt),
                "--no-ledger-crosscheck",
            ]
        )
    finally:
        msr._default_runner = original
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == msr.READABLE
    assert payload["total_usd"] == "1.00"
    assert payload["headroom_usd"] == "19.00"
    assert json.loads(receipt.read_text(encoding="utf-8").strip())["total_usd"] == "1.00"


@pytest.mark.parametrize("bad", ["abc", "NaN", "Infinity", ""])
def test_cli_rejects_a_malformed_cap_cleanly_not_with_a_traceback(bad) -> None:
    """``Decimal`` raises ``InvalidOperation`` (an ``ArithmeticError``), which
    argparse does not catch — so a typo used to print a raw traceback."""
    with pytest.raises(SystemExit) as excinfo:
        msr.main(["--cap-usd", bad, "--no-receipt"])
    assert excinfo.value.code == 2  # argparse usage error, not a crash


def test_cli_fail_over_cap_exits_3(tmp_path: Path) -> None:
    original = msr._default_runner
    msr._default_runner = _runner(_completed(stdout=json.dumps([{"cost": "25.00"}])))
    try:
        rc = msr.main(
            [
                "--start",
                "2026-08-15",
                "--end",
                "2026-08-16",
                "--fail-over-cap",
                "--no-receipt",
                "--no-ledger-crosscheck",
            ]
        )
    finally:
        msr._default_runner = original
    assert rc == 3


# --------------------------------------------------------------------------
# LIVE PROVIDER CONTROL — opt-in; the endpoint is rate-limited
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("TAC_MODAL_SPEND_LIVE_CONTROL") != "1",
    reason="live billing control is opt-in; the workspace endpoint is rate-limited",
)
def test_live_positive_control_known_billed_day_reads_nonzero() -> None:
    reading = msr.read_billing_window(date(2026, 8, 15), date(2026, 8, 16))
    assert reading.total_usd > 0
    assert reading.row_count > 0


@pytest.mark.skipif(
    os.environ.get("TAC_MODAL_SPEND_LIVE_CONTROL") != "1",
    reason="live billing control is opt-in; the workspace endpoint is rate-limited",
)
def test_live_negative_control_over_limit_span_refuses_at_the_provider(monkeypatch) -> None:
    """Lift the LOCAL guard so the PROVIDER'S own rc=1 drives the refusal.

    Without this the local span check would short-circuit and the guard chain
    would never see a real provider failure — the control would be testing our
    own arithmetic, not the meter.
    """
    monkeypatch.setattr(msr, "MAX_DAILY_SPAN_DAYS", 3650)
    with pytest.raises(msr.BillingUnreadable) as excinfo:
        msr.read_billing_window(date(2026, 7, 1), date(2026, 8, 16))
    assert excinfo.value.reason in {"nonzero_returncode", "rate_limited"}
    assert excinfo.value.returncode != 0
