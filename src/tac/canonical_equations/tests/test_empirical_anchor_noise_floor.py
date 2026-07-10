"""Tests for the P2 noise_floor schema extension on EmpiricalAnchor + delta_exceeds_floor.

Additive + legacy-compatible: an anchor built with no noise_floor behaves exactly as before (None,
not emitted in to_dict). A set floor is validated (non-negative + requires provenance) and round-trips.
delta_exceeds_floor honestly returns None when the floor is UNMEASURED (never silently clears a Δ).
"""
from __future__ import annotations

import pytest

from tac.canonical_equations import delta_exceeds_floor
from tac.canonical_equations.equation import EmpiricalAnchor, InvalidEquationError
from tac.provenance.builders import build_provenance_for_predicted


def _prov():
    return build_provenance_for_predicted(model_id="test.noise_floor.v1", inputs_sha256="0" * 64)


def _anchor(residual: float, *, noise_floor=None, noise_floor_provenance=None) -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id="a1", measurement_utc="2026-07-09T00:00:00Z", inputs={"x": 1},
        predicted_output={"y": 0.0}, empirical_output={"y": residual}, residual=residual,
        source_artifact="art", measurement_method="m", provenance=_prov(),
        noise_floor=noise_floor, noise_floor_provenance=noise_floor_provenance,
    )


def test_legacy_anchor_no_floor_unchanged():
    a = _anchor(0.01)
    assert a.noise_floor is None
    d = a.to_dict()
    assert "noise_floor" not in d  # byte-stable: not emitted when unset


def test_set_floor_serializes_and_roundtrips():
    a = _anchor(0.02, noise_floor=0.005, noise_floor_provenance="seed-variance probe #999")
    d = a.to_dict()
    assert d["noise_floor"] == pytest.approx(0.005)
    assert d["noise_floor_provenance"] == "seed-variance probe #999"


def test_floor_requires_provenance():
    with pytest.raises(InvalidEquationError):
        _anchor(0.02, noise_floor=0.005)  # no provenance
    with pytest.raises(InvalidEquationError):
        _anchor(0.02, noise_floor=-1.0, noise_floor_provenance="x")  # negative


def test_delta_exceeds_floor_none_when_unmeasured():
    # UNMEASURED floor -> None (NEVER silently treated as 0, which would falsely clear every Δ)
    assert delta_exceeds_floor(_anchor(0.02)) is None


def test_delta_exceeds_floor_true_and_false():
    a = _anchor(0.02, noise_floor=0.005, noise_floor_provenance="p")
    assert delta_exceeds_floor(a) is True                 # 0.02 > 0.005
    a2 = _anchor(0.003, noise_floor=0.005, noise_floor_provenance="p")
    assert delta_exceeds_floor(a2) is False               # 0.003 <= 0.005 -> within noise
    # explicit delta override against the same floor
    assert delta_exceeds_floor(a, delta=0.001) is False
    assert delta_exceeds_floor(a, delta=-0.02) is True    # magnitude
