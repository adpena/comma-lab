"""Tests for the canonical reclaimable-aware memory basis (tools/mem_basis.py, CLASS 1 fix)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

mem_basis = importlib.import_module("tools.mem_basis")


def test_conservative_free_gib_returns_finite_on_this_host():
    v = mem_basis.conservative_free_gib()
    # On any host with psutil or the governor, we get a real number; else the default (inf).
    assert isinstance(v, float)
    assert v >= 0.0


def test_true_committed_gib_returns_nonneg():
    v = mem_basis.true_committed_gib()
    assert isinstance(v, float)
    assert v >= 0.0


def test_prefers_reclaimable_when_governor_ok(monkeypatch):
    class _Snap:
        reclaimable_ok = True
        available_reclaimable_gib = 13.7
        available_gib = 57.3
        used_committed_gib = 114.3
        used_gib = 70.0

    monkeypatch.setattr(mem_basis, "_governor_snapshot", lambda: _Snap())
    assert mem_basis.conservative_free_gib() == pytest.approx(13.7)
    assert mem_basis.true_committed_gib() == pytest.approx(114.3)


def test_falls_back_to_legacy_avail_when_reclaimable_not_ok(monkeypatch):
    class _Snap:
        reclaimable_ok = False
        available_reclaimable_gib = 999.0
        available_gib = 57.3
        used_committed_gib = 999.0
        used_gib = 70.0

    monkeypatch.setattr(mem_basis, "_governor_snapshot", lambda: _Snap())
    assert mem_basis.conservative_free_gib() == pytest.approx(57.3)
    assert mem_basis.true_committed_gib() == pytest.approx(70.0)


def test_falls_back_to_psutil_when_no_governor(monkeypatch):
    monkeypatch.setattr(mem_basis, "_governor_snapshot", lambda: None)
    monkeypatch.setattr(mem_basis, "_psutil_available_gib", lambda: 42.0)
    monkeypatch.setattr(mem_basis, "_psutil_used_gib", lambda: 11.0)
    assert mem_basis.conservative_free_gib() == pytest.approx(42.0)
    assert mem_basis.true_committed_gib() == pytest.approx(11.0)


def test_default_used_when_everything_unavailable(monkeypatch):
    monkeypatch.setattr(mem_basis, "_governor_snapshot", lambda: None)
    monkeypatch.setattr(mem_basis, "_psutil_available_gib", lambda: None)
    monkeypatch.setattr(mem_basis, "_psutil_used_gib", lambda: None)
    assert mem_basis.conservative_free_gib(default=float("inf")) == float("inf")
    assert mem_basis.conservative_free_gib(default=0.0) == 0.0
    assert mem_basis.true_committed_gib(default=5.0) == pytest.approx(5.0)


def test_governor_snapshot_exception_is_isolated(monkeypatch):
    def _boom():
        raise RuntimeError("vm_stat exploded")

    # _governor_snapshot itself swallows; simulate it returning None then psutil path.
    monkeypatch.setattr(mem_basis, "_governor_snapshot", lambda: None)
    monkeypatch.setattr(mem_basis, "_psutil_available_gib", lambda: 7.5)
    assert mem_basis.conservative_free_gib() == pytest.approx(7.5)
