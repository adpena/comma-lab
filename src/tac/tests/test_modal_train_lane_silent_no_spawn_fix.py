# SPDX-License-Identifier: MIT
"""Sister tests for the Catalog #339 SILENT-NO-SPAWN-STRUCTURAL-EXTINCTION fix.

Covers the canonical helper extension (`register_dispatched_call_id_fail_closed`
+ `poll_ledger_for_call_id` + `LedgerRegistrationFailedError`) AND the
wrapper-side fix in `experiments/modal_train_lane.py` (no silent swallow;
sys.exit(13) on registration failure).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest import mock

import pytest

from tac.deploy.modal.call_id_ledger import (
    LedgerRegistrationFailedError,
    poll_ledger_for_call_id,
    register_dispatched_call_id,
    register_dispatched_call_id_fail_closed,
)


# ----------------------------------------------------------------------------
# LedgerRegistrationFailedError + fail-closed wrapper
# ----------------------------------------------------------------------------


def test_fail_closed_helper_returns_record_on_success(tmp_path: Path) -> None:
    """Happy path: returns the record (identical to canonical helper)."""
    ledger = tmp_path / "ledger.jsonl"
    lock = tmp_path / "ledger.jsonl.lock"
    record = register_dispatched_call_id_fail_closed(
        call_id="fc-test1",
        lane_id="lane_x",
        label="label_x",
        path=ledger,
        lock_path=lock,
    )
    assert record["call_id"] == "fc-test1"
    assert record["lane_id"] == "lane_x"
    assert record["event_type"] == "dispatched"


def test_fail_closed_helper_raises_on_failure() -> None:
    """When the canonical helper raises, fail-closed wrapper RE-RAISES."""
    with mock.patch(
        "tac.deploy.modal.call_id_ledger.register_dispatched_call_id",
        side_effect=RuntimeError("simulated fcntl failure"),
    ):
        with pytest.raises(LedgerRegistrationFailedError) as excinfo:
            register_dispatched_call_id_fail_closed(
                call_id="fc-bad",
                lane_id="lane_bad",
                label="label_bad",
                write_last_resort_dump=False,
            )
    msg = str(excinfo.value)
    assert "Catalog #339" in msg
    assert "fc-bad" in msg
    assert "simulated fcntl failure" in msg


def test_fail_closed_helper_writes_last_resort_dump(tmp_path: Path) -> None:
    """On failure, the wrapper writes a recovery tmp file the harvester can use."""
    from tac.deploy.modal import call_id_ledger as ledger_mod

    dump_dir = tmp_path / "dumps"
    with mock.patch.object(ledger_mod, "_LAST_RESORT_TMP_DUMP_DIR", dump_dir), \
         mock.patch.object(
             ledger_mod, "register_dispatched_call_id",
             side_effect=OSError("disk full")
         ):
        with pytest.raises(LedgerRegistrationFailedError) as excinfo:
            register_dispatched_call_id_fail_closed(
                call_id="fc-recover",
                lane_id="lane_r",
                label="label_r",
            )
    # Last-resort dump exists.
    dumps = list(dump_dir.glob("recover_fc-recover_*.json"))
    assert len(dumps) == 1
    payload = json.loads(dumps[0].read_text())
    assert payload["call_id"] == "fc-recover"
    assert payload["event_type"] == "dispatched"
    assert payload["_recovery_reason"].startswith("canonical_append_failed:OSError:")
    assert "Last-resort dump written to" in str(excinfo.value)
    assert "harvest_modal_calls.py --recover-from-tmp" in str(excinfo.value)


def test_fail_closed_helper_diagnostic_when_dump_also_fails() -> None:
    """If even the tmp-dump write fails, diagnostic explains harvester is blind."""
    from tac.deploy.modal import call_id_ledger as ledger_mod

    with mock.patch.object(
        ledger_mod, "register_dispatched_call_id",
        side_effect=RuntimeError("primary failure")
    ), mock.patch.object(
        ledger_mod, "_write_last_resort_tmp_ledger_dump",
        side_effect=PermissionError("tmp also gone")
    ):
        with pytest.raises(LedgerRegistrationFailedError) as excinfo:
            register_dispatched_call_id_fail_closed(
                call_id="fc-doomed",
                lane_id="lane_d",
                label="label_d",
            )
    msg = str(excinfo.value)
    assert "INVISIBLE" in msg
    assert "HARVEST OR LOSE" in msg


def test_fail_closed_helper_chains_original_exception() -> None:
    """The raised exception MUST chain the original (__cause__ set)."""
    with mock.patch(
        "tac.deploy.modal.call_id_ledger.register_dispatched_call_id",
        side_effect=ValueError("orig"),
    ):
        with pytest.raises(LedgerRegistrationFailedError) as excinfo:
            register_dispatched_call_id_fail_closed(
                call_id="fc-chain",
                lane_id="lane_c",
                label="label_c",
                write_last_resort_dump=False,
            )
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert str(excinfo.value.__cause__) == "orig"


# ----------------------------------------------------------------------------
# poll_ledger_for_call_id
# ----------------------------------------------------------------------------


def test_poll_returns_true_when_row_present(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    lock = tmp_path / "ledger.jsonl.lock"
    register_dispatched_call_id(
        call_id="fc-found",
        lane_id="lane_x",
        label="label_x",
        path=ledger,
        lock_path=lock,
    )
    assert poll_ledger_for_call_id(
        "fc-found", timeout_seconds=2.0, poll_interval_seconds=0.05, path=ledger
    )


def test_poll_returns_false_on_timeout(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger_missing.jsonl"
    # Ledger does not exist; poll should time out (returns False).
    assert not poll_ledger_for_call_id(
        "fc-missing", timeout_seconds=0.3, poll_interval_seconds=0.05, path=ledger
    )


def test_poll_tolerates_corrupt_ledger(tmp_path: Path) -> None:
    """Lenient load: corruption doesn't crash the poll (returns False)."""
    ledger = tmp_path / "ledger_corrupt.jsonl"
    ledger.write_text("not valid json\n", encoding="utf-8")
    result = poll_ledger_for_call_id(
        "fc-anything", timeout_seconds=0.2, poll_interval_seconds=0.05, path=ledger
    )
    assert result is False


# ----------------------------------------------------------------------------
# experiments/modal_train_lane.py source-level verification
# ----------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_modal_train_lane_uses_fail_closed_helper() -> None:
    """The wrapper must reference the fail-closed helper, not the legacy bare one."""
    source = (REPO_ROOT / "experiments" / "modal_train_lane.py").read_text(
        encoding="utf-8"
    )
    assert "register_dispatched_call_id_fail_closed" in source, (
        "Wrapper must use the fail-closed helper per Catalog #339 Layer 2 fix."
    )
    assert "LedgerRegistrationFailedError" in source, (
        "Wrapper must import LedgerRegistrationFailedError for the explicit "
        "catch-and-sys.exit handler."
    )


def test_modal_train_lane_calls_sys_exit_on_ledger_failure() -> None:
    """The wrapper must sys.exit(13) on LedgerRegistrationFailedError."""
    source = (REPO_ROOT / "experiments" / "modal_train_lane.py").read_text(
        encoding="utf-8"
    )
    # Look for the pattern: except LedgerRegistrationFailedError ... sys.exit(13)
    pattern = re.compile(
        r"except\s+LedgerRegistrationFailedError.*?sys\.exit\s*\(\s*13\s*\)",
        re.DOTALL,
    )
    assert pattern.search(source), (
        "Wrapper must catch LedgerRegistrationFailedError and call sys.exit(13). "
        "This is the Catalog #339 fail-closed contract."
    )


def test_modal_train_lane_no_legacy_silent_swallow_around_register() -> None:
    """No `try: register_dispatched_call_id(...) except Exception: print` legacy pattern."""
    source = (REPO_ROOT / "experiments" / "modal_train_lane.py").read_text(
        encoding="utf-8"
    )
    # Match the EXACT legacy anti-pattern: bare `register_dispatched_call_id(`
    # call followed within ~50 lines by `except Exception` whose body just
    # prints WARNING. We allow `register_dispatched_call_id_fail_closed` to
    # appear (different name).
    bare_calls = re.findall(
        r"^\s*register_dispatched_call_id\s*\(",
        source,
        re.MULTILINE,
    )
    # Note: `register_dispatched_call_id_fail_closed` is a different symbol.
    assert not bare_calls, (
        f"Wrapper still contains {len(bare_calls)} call(s) to the legacy "
        "non-fail-closed helper. Use register_dispatched_call_id_fail_closed."
    )


def test_modal_train_lane_threads_caller_agent_into_call_id_ledger() -> None:
    """Modal call-id ledger rows must preserve the caller's agent string."""
    source = (REPO_ROOT / "experiments" / "modal_train_lane.py").read_text(
        encoding="utf-8"
    )
    assert 'agent: str = "claude:modal_train_lane"' in source
    assert "agent=agent" in source
    assert 'agent="claude"' not in source


# ----------------------------------------------------------------------------
# tools/operator_authorize.py sister-mitigation verification
# ----------------------------------------------------------------------------


def test_operator_authorize_has_ledger_poll_helper() -> None:
    """Sister mitigation: _poll_ledger_for_dispatched_call must exist."""
    source = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(
        encoding="utf-8"
    )
    assert "_poll_ledger_for_dispatched_call" in source


def test_operator_authorize_threads_agent_to_modal_train_lane() -> None:
    """The operator-authorize --agent value must reach modal_train_lane.py."""
    source = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(
        encoding="utf-8"
    )
    assert re.search(r'"--agent",\s*agent,', source)
    assert "agent=args.agent" in source
    assert "_extract_modal_call_id_from_output" in source


def test_operator_authorize_dispatch_modal_polls_ledger_before_return_zero() -> None:
    """After spawned_call=True, _dispatch_modal must poll the ledger."""
    source = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(
        encoding="utf-8"
    )
    # Look for the Catalog #339 anchor + poll + SystemExit pattern in
    # _dispatch_modal.
    pattern = re.compile(
        r"def _dispatch_modal.*?_poll_ledger_for_dispatched_call.*?SystemExit",
        re.DOTALL,
    )
    assert pattern.search(source), (
        "_dispatch_modal must poll the ledger AND raise SystemExit on miss "
        "per Catalog #339 sister mitigation."
    )


def test_operator_authorize_extracts_call_id_from_modal_output() -> None:
    """Sister helper must extract `call_id=fc-...` from Modal stdout."""
    from tools.operator_authorize import _extract_modal_call_id_from_output

    # Canonical Modal banner
    out1 = "[modal_train_lane] dispatch_completed call_id=fc-01ABCXYZ"
    assert _extract_modal_call_id_from_output(out1) == "fc-01ABCXYZ"

    # No call_id in output
    out2 = "Modal app initialized but never spawned"
    assert _extract_modal_call_id_from_output(out2) is None


# ----------------------------------------------------------------------------
# End-to-end smoke: fail-closed loop integration
# ----------------------------------------------------------------------------


def test_end_to_end_fail_closed_then_poll_returns_false(tmp_path: Path) -> None:
    """E2E: registration fails → no ledger row → poll returns False."""
    from tac.deploy.modal import call_id_ledger as ledger_mod

    ledger = tmp_path / "e2e.jsonl"
    dump_dir = tmp_path / "dumps"
    with mock.patch.object(ledger_mod, "_LAST_RESORT_TMP_DUMP_DIR", dump_dir), \
         mock.patch.object(
             ledger_mod, "register_dispatched_call_id",
             side_effect=RuntimeError("simulated")
         ):
        with pytest.raises(LedgerRegistrationFailedError):
            register_dispatched_call_id_fail_closed(
                call_id="fc-e2e",
                lane_id="lane_e2e",
                label="label_e2e",
            )
    # The ledger has no row; poll times out.
    assert not poll_ledger_for_call_id(
        "fc-e2e", timeout_seconds=0.2, poll_interval_seconds=0.05, path=ledger
    )
    # But the recovery tmp dump DOES exist (harvester can re-attempt).
    assert any(dump_dir.glob("recover_fc-e2e_*.json"))
