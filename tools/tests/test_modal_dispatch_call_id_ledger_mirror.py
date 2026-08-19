# SPDX-License-Identifier: MIT
"""Catalog #330 regression: ``modal_dispatch status --live`` must terminalize.

``_live_call_state`` polls ``FunctionCall.get`` — the exact surface Catalog #330
governs. Before ddm_fx3 it observed terminal provider state (a returned result,
or an expired result cache) and returned a bare status string, leaving the
canonical call-id ledger stuck at ``dispatched`` forever. These tests pin the
mirroring behaviour AND its precision: a poll timeout must stay in-flight, and
an observed nonzero rc must not be laundered into ``harvested`` by the
weaker "the provider returned something" fallback claim.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from tac.deploy.modal.call_id_ledger import (
    latest_status_by_call_id,
    register_dispatched_call_id,
)
from tools import modal_dispatch

CALL_ID = "fc-01TESTTESTTESTTESTTESTTEST"


def _ledger_path(root: Path) -> Path:
    return root / ".omx" / "state" / "modal_call_id_ledger.jsonl"


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated repo root whose ledger already holds a ``dispatched`` row."""
    ledger = _ledger_path(tmp_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    register_dispatched_call_id(
        call_id=CALL_ID,
        lane_id="lane_fx3_test",
        label="fx3_test",
        path=ledger,
        lock_path=ledger.with_suffix(ledger.suffix + ".lock"),
    )
    assert latest_status_by_call_id(path=ledger).get(CALL_ID) == "dispatched"
    monkeypatch.setattr(modal_dispatch, "REPO", tmp_path)
    return tmp_path


def _install_fake_modal(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: object = None,
    raises: BaseException | None = None,
) -> None:
    """Inject a ``modal`` stub whose FunctionCall.get returns/raises on demand."""

    class _FunctionCall:
        @staticmethod
        def from_id(call_id: str) -> "_FunctionCall":
            assert call_id == CALL_ID
            return _FunctionCall()

        def get(self, timeout: float | None = None) -> object:
            if raises is not None:
                raise raises
            return result

    fake = types.ModuleType("modal")
    fake.functions = types.SimpleNamespace(FunctionCall=_FunctionCall)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modal", fake)


class _OutputExpiredError(Exception):
    """Name matches the substring ``_live_call_state`` classifies on."""


def test_completed_rc0_mirrors_harvested(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_modal(monkeypatch, result={"rc": 0, "elapsed_seconds": 12.5})
    assert modal_dispatch._live_call_state(CALL_ID) == "completed"
    assert latest_status_by_call_id(path=_ledger_path(repo)).get(CALL_ID) == "harvested"


def test_completed_nonzero_rc_mirrors_failed_not_harvested(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Precision control: the observed rc must beat the fallback 'completed' claim.

    Without this the fallback would launder every returned payload into
    ``harvested`` and the ledger would report remote failures as successes.
    """
    _install_fake_modal(monkeypatch, result={"rc": 3})
    assert modal_dispatch._live_call_state(CALL_ID) == "completed"
    assert latest_status_by_call_id(path=_ledger_path(repo)).get(CALL_ID) == "failed"


def test_completed_unclassifiable_payload_falls_back_to_harvested(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-dict result carries no rc, but the provider did return terminally."""
    _install_fake_modal(monkeypatch, result="an opaque non-dict result")
    assert modal_dispatch._live_call_state(CALL_ID) == "completed"
    assert latest_status_by_call_id(path=_ledger_path(repo)).get(CALL_ID) == "harvested"


def test_poll_timeout_stays_in_flight(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bounded poll timeout is NONTERMINAL — terminalizing it would be a lie."""
    _install_fake_modal(monkeypatch, raises=TimeoutError("still running"))
    assert modal_dispatch._live_call_state(CALL_ID) == "running"
    assert latest_status_by_call_id(path=_ledger_path(repo)).get(CALL_ID) == "dispatched"


def test_result_cache_expiry_mirrors_stale(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_modal(monkeypatch, raises=_OutputExpiredError("gone"))
    assert modal_dispatch._live_call_state(CALL_ID) == "expired(>24h)"
    assert latest_status_by_call_id(path=_ledger_path(repo)).get(CALL_ID) == "stale"


def test_unknown_exception_does_not_terminalize(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unclassified provider error is not evidence of a terminal outcome."""
    _install_fake_modal(monkeypatch, raises=RuntimeError("transient provider blip"))
    assert modal_dispatch._live_call_state(CALL_ID) == "unknown(RuntimeError)"
    assert latest_status_by_call_id(path=_ledger_path(repo)).get(CALL_ID) == "dispatched"


def test_ledger_failure_never_breaks_the_status_command(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A read-only status command must survive a ledger fault — loudly, not silently."""
    _install_fake_modal(monkeypatch, result={"rc": 0})

    def _boom(**_kwargs: object) -> dict[str, object]:
        raise OSError("ledger volume unavailable")

    monkeypatch.setattr(
        "tac.deploy.modal.harvest_outcomes.append_terminal_call_id_ledger_event", _boom
    )
    assert modal_dispatch._live_call_state(CALL_ID) == "completed"
    assert "WARN: call_id ledger mirror failed" in capsys.readouterr().err


def test_arbitrary_result_payload_is_not_dumped_into_the_shared_ledger(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fire`` wraps arbitrary commands; a big result must not bloat the ledger.

    The mirrored row keeps every field the helper reads and drops the rest.
    """
    blob = "x" * 5000
    _install_fake_modal(
        monkeypatch,
        result={"rc": 0, "elapsed_seconds": 4.0, "artifacts": {"big": blob}, "junk": blob},
    )
    assert modal_dispatch._live_call_state(CALL_ID) == "completed"

    text = _ledger_path(repo).read_text()
    assert latest_status_by_call_id(path=_ledger_path(repo)).get(CALL_ID) == "harvested"
    assert blob not in text, "raw payload leaked into the shared ledger"
    assert "junk" not in text
    # ...but the classification signal survived.
    assert '"rc": 0' in text or '"rc":0' in text


def test_projection_keeps_only_helper_consumed_fields() -> None:
    got = modal_dispatch._ledger_signal_fields(
        {"rc": 0, "score": 0.15, "artifacts": {"a": 1}, "unrelated": "drop me"}
    )
    assert got == {"rc": 0, "score": 0.15}
    assert modal_dispatch._ledger_signal_fields("not a dict") is None
    assert modal_dispatch._ledger_signal_fields({"artifacts": {}}) is None


def test_mirror_is_idempotent_across_repeated_status_polls(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``status --live`` is polled repeatedly; it must not spam terminal rows."""
    _install_fake_modal(monkeypatch, result={"rc": 0, "elapsed_seconds": 12.5})
    for _ in range(3):
        assert modal_dispatch._live_call_state(CALL_ID) == "completed"
    rows = _ledger_path(repo).read_text().strip().splitlines()
    terminal = [r for r in rows if '"harvested"' in r]
    assert len(terminal) == 1, f"expected one terminal row, got {len(terminal)}"
