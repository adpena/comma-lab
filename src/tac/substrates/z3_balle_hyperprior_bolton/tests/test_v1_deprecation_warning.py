# SPDX-License-Identifier: MIT
"""SELFCOMP-1 (R2 MEDIUM, 2026-05-15): v1 import emits DeprecationWarning.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
v1 of the Z3 Ballé hyperprior bolt-on remains LIVE (the production default
trainer path) but commit message ``e54901d60`` claimed v1 was "retired as
redundant per same verdict" — documentation-vs-reality drift caught by R2
SELFCOMP voice.

The fix is OPTION C from the SELFCOMP-1 fix-recommendation menu: add a
DeprecationWarning when v1 is imported without the explicit
``Z3_BALLE_USE_V1=1`` env opt-in. v1 stays LIVE because reactivation
criterion (v2 paired Modal anchor) has not landed; per CLAUDE.md "Forbidden
premature KILL" the lane is DEPRECATED-pending-v2-empirical-confirmation,
NOT KILLED.

These tests pin the warning behavior so a future ``__init__.py`` refactor
that drops the warning is caught at CI time.

Cross-refs: ``feedback_recursive_review_r2_wave_a_*`` SELFCOMP-1 +
``feedback_r2_medium_fix_wave_selfcomp_mackay_landed_20260515.md``.
"""

from __future__ import annotations

import importlib
import os
import warnings
from typing import Iterable

import pytest


def _reload_z3() -> object:
    """Force a fresh import of the Z3 package so the one-shot warning fires.

    Pytest re-uses imported modules across tests; without explicit reload
    the deprecation-warning emission inside ``__init__.py`` only fires
    on the FIRST test in the session.
    """
    import sys

    # Drop every cached Z3 module so the next import re-executes __init__.
    to_drop = [
        name for name in list(sys.modules)
        if name == "tac.substrates.z3_balle_hyperprior_bolton"
        or name.startswith("tac.substrates.z3_balle_hyperprior_bolton.")
    ]
    for name in to_drop:
        del sys.modules[name]
    return importlib.import_module("tac.substrates.z3_balle_hyperprior_bolton")


def _category_names(records: Iterable[warnings.WarningMessage]) -> list[str]:
    return [type(r.message).__name__ for r in records]


def test_v1_import_emits_deprecation_warning_by_default(monkeypatch):
    """Bare import without the env opt-in emits a DeprecationWarning."""
    monkeypatch.delenv("Z3_BALLE_USE_V1", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _reload_z3()
    z3_warnings = [
        r for r in caught
        if "Z3 Ballé hyperprior" in str(r.message)
        and isinstance(r.message, DeprecationWarning)
    ]
    assert z3_warnings, (
        f"Expected DeprecationWarning containing 'Z3 Ballé hyperprior' "
        f"on import, got categories={_category_names(caught)}, "
        f"messages={[str(r.message) for r in caught]}"
    )


def test_v1_import_warning_mentions_z3_balle_use_v1_env_var(monkeypatch):
    """Warning message names the canonical opt-out env var."""
    monkeypatch.delenv("Z3_BALLE_USE_V1", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _reload_z3()
    msgs = [str(r.message) for r in caught if "Z3 Ballé hyperprior" in str(r.message)]
    assert msgs, "expected at least one Z3 Ballé hyperprior warning"
    assert "Z3_BALLE_USE_V1" in msgs[0], (
        f"warning must name the Z3_BALLE_USE_V1 opt-out env var; got {msgs[0]!r}"
    )


def test_v1_import_warning_mentions_v2_replacement(monkeypatch):
    """Warning message points operators at v2 / --enable-v2-latent-replacement."""
    monkeypatch.delenv("Z3_BALLE_USE_V1", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _reload_z3()
    msgs = [str(r.message) for r in caught if "Z3 Ballé hyperprior" in str(r.message)]
    assert msgs, "expected at least one Z3 Ballé hyperprior warning"
    msg = msgs[0]
    assert "v2" in msg, f"warning must reference v2 path; got {msg!r}"
    assert "--enable-v2-latent-replacement" in msg, (
        f"warning must name the trainer flag; got {msg!r}"
    )


def test_v1_import_warning_mentions_council_decision(monkeypatch):
    """Warning cites the council omnibus Decision 3 commit anchor."""
    monkeypatch.delenv("Z3_BALLE_USE_V1", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _reload_z3()
    msgs = [str(r.message) for r in caught if "Z3 Ballé hyperprior" in str(r.message)]
    assert msgs, "expected at least one Z3 Ballé hyperprior warning"
    msg = msgs[0]
    assert "7872c9f4b" in msg, (
        f"warning must cite the council Decision 3 commit anchor "
        f"(7872c9f4b) for traceability; got {msg!r}"
    )


def test_v1_import_warning_suppressed_when_env_opt_in(monkeypatch):
    """Setting Z3_BALLE_USE_V1=1 suppresses the warning (legitimate v1 use)."""
    monkeypatch.setenv("Z3_BALLE_USE_V1", "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _reload_z3()
    z3_warnings = [
        r for r in caught
        if "Z3 Ballé hyperprior" in str(r.message)
        and isinstance(r.message, DeprecationWarning)
    ]
    assert not z3_warnings, (
        f"Z3_BALLE_USE_V1=1 must suppress the v1 deprecation warning; "
        f"got {[str(r.message) for r in z3_warnings]}"
    )


def test_v1_import_warning_suppressed_when_env_opt_in_with_whitespace(monkeypatch):
    """The env-opt-in check strips whitespace (operator-friendly)."""
    monkeypatch.setenv("Z3_BALLE_USE_V1", " 1 ")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _reload_z3()
    z3_warnings = [
        r for r in caught
        if "Z3 Ballé hyperprior" in str(r.message)
        and isinstance(r.message, DeprecationWarning)
    ]
    assert not z3_warnings, (
        "Z3_BALLE_USE_V1=' 1 ' (whitespace-padded) must suppress the warning"
    )


def test_v1_import_warning_NOT_suppressed_by_other_truthy_values(monkeypatch):
    """Only the literal '1' suppresses; 'true' / 'yes' / '0' do NOT.

    This is intentional — operator MUST be explicit. We do not want
    accidental suppression from a stale env var with a non-canonical value.
    """
    for value in ("0", "true", "yes", "True", "TRUE", "false", "no", ""):
        monkeypatch.setenv("Z3_BALLE_USE_V1", value)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            _reload_z3()
        z3_warnings = [
            r for r in caught
            if "Z3 Ballé hyperprior" in str(r.message)
            and isinstance(r.message, DeprecationWarning)
        ]
        assert z3_warnings, (
            f"Z3_BALLE_USE_V1={value!r} is NOT the canonical opt-in ('1'); "
            f"the warning MUST still fire. got={[str(r.message) for r in z3_warnings]}"
        )


def test_emit_v1_deprecation_warning_returns_true_when_warning_emitted(monkeypatch):
    """_emit_v1_deprecation_warning() returns True when warning fired."""
    monkeypatch.delenv("Z3_BALLE_USE_V1", raising=False)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", DeprecationWarning)
        z3 = _reload_z3()
        result = z3._emit_v1_deprecation_warning()
    assert result is True, "must return True when warning was emitted"


def test_emit_v1_deprecation_warning_returns_false_when_suppressed(monkeypatch):
    """_emit_v1_deprecation_warning() returns False when env opt-in active."""
    monkeypatch.setenv("Z3_BALLE_USE_V1", "1")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", DeprecationWarning)
        z3 = _reload_z3()
        result = z3._emit_v1_deprecation_warning()
    assert result is False, (
        "must return False when Z3_BALLE_USE_V1=1 suppresses the warning"
    )


def test_v1_archive_module_still_importable_after_warning(monkeypatch):
    """The warning is non-fatal: v1 archive symbols still import successfully."""
    monkeypatch.delenv("Z3_BALLE_USE_V1", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        z3 = _reload_z3()
    # v1 surfaces remain LIVE per "DEPRECATED-pending-v2-empirical-confirmation"
    # — these symbols MUST still be importable.
    assert hasattr(z3, "Z3HP1_MAGIC"), "v1 Z3HP1_MAGIC must remain exported"
    assert hasattr(z3, "Z3HP1SidecarMeta"), "v1 SidecarMeta must remain exported"
    assert hasattr(z3, "pack_composition_archive"), (
        "v1 pack_composition_archive must remain exported"
    )


def test_v2_archive_module_still_importable_after_warning(monkeypatch):
    """v2 surfaces remain importable (no regression on the operational path)."""
    monkeypatch.delenv("Z3_BALLE_USE_V1", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        z3 = _reload_z3()
    assert hasattr(z3, "Z3HV2_MAGIC"), "v2 Z3HV2_MAGIC must remain exported"
    assert hasattr(z3, "Z3HV2SectionMeta"), "v2 SectionMeta must remain exported"
    assert hasattr(z3, "build_z3v2_payload_bytes"), (
        "v2 build_z3v2_payload_bytes must remain exported"
    )


def test_emit_v1_deprecation_warning_listed_in_all(monkeypatch):
    """The helper is publicly exported in __all__ for sister-subagent reuse."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        z3 = _reload_z3()
    assert "_emit_v1_deprecation_warning" in z3.__all__, (
        "_emit_v1_deprecation_warning must be in __all__ so sister tests + "
        "subagent tooling can call it directly"
    )
