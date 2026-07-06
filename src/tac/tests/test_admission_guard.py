"""Tests for the #254 governor admission-bypass GUARD (tac.admission_guard).

Locks: advisory-by-default, refuse-when-enforced-and-raw, governed/bypass pass, placeholder
rationale rejected, and PARITY with tools/system_memory_governor.admission_enforcing (so the two
predicates cannot silently drift)."""
from __future__ import annotations

import os
import sys

import pytest

from tac import admission_guard as ag

_ENF = {ag.ADMISSION_ENFORCE_ENV: "1"}
_GOV = {ag.GOVERNED_MARKER_ENV: "1"}


@pytest.fixture
def _no_enforce_flag(monkeypatch, tmp_path):
    """Point the durable enforce-flag at a NON-EXISTENT path so the env var alone controls
    enforcement (this machine has the real flag armed; the unit must isolate it)."""
    monkeypatch.setattr(ag, "_ENFORCE_FLAG", tmp_path / "noflag")
    return monkeypatch


def test_advisory_when_not_enforced_even_if_raw(_no_enforce_flag):
    ok, msg = ag.admission_status("t", env={})   # neither enforced (no flag) nor governed
    assert ok is True and "ADVISORY" in msg


def test_refuse_when_enforced_and_raw(_no_enforce_flag):
    ok, msg = ag.admission_status("t", env=dict(_ENF))
    assert ok is False and "REFUSED" in msg and "MUST go through" in msg


def test_governed_passes_under_enforce():
    ok, _ = ag.admission_status("t", env={**_ENF, **_GOV})
    assert ok is True


def test_bypass_rationale_passes_under_enforce_but_placeholder_rejected():
    ok, _ = ag.admission_status("t", env={**_ENF, ag.BYPASS_OVERRIDE_ENV: "reviewed one-off n24 smoke"})
    assert ok is True
    for placeholder in ("", "<reason>", "<rationale>", "x", "todo"):
        assert ag.bypass_rationale({ag.BYPASS_OVERRIDE_ENV: placeholder}) is None
    assert ag.bypass_rationale({ag.BYPASS_OVERRIDE_ENV: "a real 4+ char reason"}) is not None


def test_mark_admitted_env_is_idempotent_and_admits():
    e: dict = {}
    ag.mark_admitted_env(e)
    assert e[ag.GOVERNED_MARKER_ENV] == "1" and ag.is_admitted(e)
    ag.mark_admitted_env(e)  # idempotent
    assert e[ag.GOVERNED_MARKER_ENV] == "1"


def test_assert_exits_7_when_refused():
    with pytest.raises(SystemExit) as ei:
        ag.assert_governed_admission("t", env=dict(_ENF), on_refuse="exit")
    assert ei.value.code == ag.EXIT_ADMISSION_REFUSED


def test_assert_raises_when_refused_and_on_refuse_raise():
    with pytest.raises(PermissionError):
        ag.assert_governed_admission("t", env=dict(_ENF), on_refuse="raise")


def test_assert_returns_true_when_governed():
    assert ag.assert_governed_admission("t", env={**_ENF, **_GOV}) is True


def test_predicate_parity_with_governor():
    """The guard's enforce predicate MUST match tools/system_memory_governor.admission_enforcing
    (both read TAC_ADMISSION_ENFORCE + the durable flag). Parity prevents silent drift."""
    sys.path.insert(0, "tools")
    import system_memory_governor as gov  # noqa: E402
    saved = os.environ.get(ag.ADMISSION_ENFORCE_ENV)
    try:
        for val in ("1", "true", "yes", "on", "0", "no", "", "garbage"):
            os.environ[ag.ADMISSION_ENFORCE_ENV] = val
            assert ag.admission_enforcing() == gov.admission_enforcing(), f"drift at env={val!r}"
    finally:
        if saved is None:
            os.environ.pop(ag.ADMISSION_ENFORCE_ENV, None)
        else:
            os.environ[ag.ADMISSION_ENFORCE_ENV] = saved
