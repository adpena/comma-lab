"""Catalog #406 tests for the retired launcher DSL-bypass decision surface."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_LAUNCHER = _REPO / "tools" / "launch_witness_run.py"


def _load_launcher():
    if str(_REPO / "tools") not in sys.path:
        sys.path.insert(0, str(_REPO / "tools"))
    spec = importlib.util.spec_from_file_location("launch_witness_run", _LAUNCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def lw():
    return _load_launcher()


def _act(lw, **over):
    kw = {
        "ok": False,
        "detail": "d",
        "manifest_absent": False,
        "config": "crucible_v6",
        "dry_run": False,
        "skip": False,
        "enforce": False,
        "allow_rationale": None,
    }
    kw.update(over)
    return lw.dsl_config_gate_action(**kw)


def test_ok_when_manifest_valid(lw):
    action, _ = _act(lw, ok=True)
    assert action == "ok"


def test_rationale_cannot_override_refusal(lw):
    action, msg = _act(lw, ok=False, allow_rationale="emergency: hotfix launch")
    assert action == "refuse" and "no authority" in msg


def test_override_ignored_when_rationale_blank(lw):
    """A blank rationale is NOT a valid override — falls through to refuse."""
    action, _ = _act(lw, ok=False, allow_rationale="   ", manifest_absent=False)
    assert action == "refuse"


def test_dry_run_cannot_downgrade_refusal(lw):
    action, msg = _act(lw, dry_run=True)
    assert action == "refuse" and "Catalog #406" in msg


def test_skip_flag_cannot_downgrade_refusal(lw):
    action, msg = _act(lw, skip=True)
    assert action == "refuse" and "no authority" in msg


def test_absent_manifest_refused_without_legacy_enforce_flag(lw):
    action, msg = _act(lw, manifest_absent=True, enforce=False)
    assert action == "refuse" and "Catalog #406" in msg


def test_refuse_on_absent_manifest_with_enforce(lw):
    action, msg = _act(lw, manifest_absent=True, enforce=True)
    assert action == "refuse" and "Catalog #406" in msg


def test_refuse_on_tampered_manifest_default(lw):
    """A manifest present-but-invalid (not absent) REFUSES by default (integrity is enforced)."""
    action, _ = _act(lw, manifest_absent=False, enforce=False)
    assert action == "refuse"


def test_dry_run_cannot_beat_enforce(lw):
    """Dry-run emits artifacts but never authorizes an unbound config."""
    action, _ = _act(lw, manifest_absent=True, enforce=True, dry_run=True)
    assert action == "refuse"


def test_retired_override_cannot_beat_refuse(lw):
    action, _ = _act(lw, manifest_absent=False, allow_rationale="op signed off")
    assert action == "refuse"
