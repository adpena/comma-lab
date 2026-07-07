"""Regression test for the #247 CLOSE-side gate (tools/levelset_byte_close_and_eval.byte_close_verdict_landed).

The 2026-07-06 independent review caught a CRITICAL NO-FAKE bug: the gate read the WRONG report key
(``"parity"`` instead of ``"parity_on_inflated_frames"``), so it fired unconditionally — including on
``--skip-parity`` runs — fabricating ``measured`` activation events that silently drain the
``duty_to_measure`` queue (the exact orphan the apparatus exists to prevent). This test pins the gate
so the bug class cannot recur."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def gate():
    for p in (str(REPO / "tools"), str(REPO / "experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    mod = importlib.import_module("levelset_byte_close_and_eval")
    return mod.byte_close_verdict_landed


def test_skip_parity_does_not_record_measured(gate):
    # a --skip-parity run stores {"skipped": True}; NO verdict landed -> must NOT record measured
    assert gate({"parity_on_inflated_frames": {"skipped": True}}) is False


def test_real_verdict_records_measured(gate):
    assert gate({"parity_on_inflated_frames": {"skipped": False, "d_seg_realized_on_inflated": 0.004}}) is True


def test_missing_key_is_failsafe_false(gate):
    # a MISSING key must default to NO-measure (fail-safe: never fabricate)
    assert gate({}) is False


def test_non_dict_parity_is_false(gate):
    assert gate({"parity_on_inflated_frames": None}) is False


def test_old_wrong_key_does_not_trigger(gate):
    # the pre-fix bug: report has a "parity" key (wrong) but not the real one -> must NOT trigger
    assert gate({"parity": {"skipped": False}}) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
