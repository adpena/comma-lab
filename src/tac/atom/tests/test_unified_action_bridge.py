# SPDX-License-Identifier: MIT
"""Tests for the unified-action bridge module."""
from __future__ import annotations

import pytest

from tac.atom.builders import (
    build_arbitrary_value_atom,
    build_meta_lagrangian_atom,
    build_probe_outcome_atom,
)
from tac.atom.types import ResolutionPath
from tac.atom.unified_action_bridge import (
    atom_pool_to_cathedral_autopilot_candidates,
    atom_pool_to_meta_lagrangian_ledger,
    evaluate_action_with_atoms,
)


def _sample_arbitrary(atom_id: str, lo: float, hi: float, cost: float = 1.0):
    return build_arbitrary_value_atom(
        atom_id=atom_id,
        file_path="f.py",
        current_value=1,
        predicted_replacement=2,
        resolution_path=ResolutionPath.EXPERIMENTAL,
        predicted_ev_delta_s=(lo, hi),
        cost_envelope_usd=cost,
        literature_citation="cit",
    )


class TestEvaluateActionWithAtoms:
    def test_empty_pool_returns_zero_zero(self) -> None:
        assert evaluate_action_with_atoms(atoms=[]) == (0.0, 0.0)

    def test_single_atom_returns_its_band(self) -> None:
        atom = _sample_arbitrary("a1", -0.005, -0.001)
        band = evaluate_action_with_atoms(atoms=[atom])
        assert band == (-0.005, -0.001)

    def test_two_atoms_aggregate_additively(self) -> None:
        a = _sample_arbitrary("a", -0.005, -0.001)
        b = _sample_arbitrary("b", -0.010, -0.002)
        band = evaluate_action_with_atoms(atoms=[a, b])
        assert band == pytest.approx((-0.015, -0.003))


class TestAtomPoolToMetaLagrangianLedger:
    def test_produces_one_legacy_row_per_atom(self) -> None:
        atoms = [
            build_meta_lagrangian_atom(
                atom_id="m1", family="f1", family_group="g1", byte_delta=-100
            ),
            build_meta_lagrangian_atom(
                atom_id="m2", family="f2", family_group="g2", byte_delta=-200
            ),
        ]
        legacy = atom_pool_to_meta_lagrangian_ledger(atoms)
        assert len(legacy) == 2
        assert legacy[0]["atom_id"] == "m1"
        assert legacy[0]["byte_delta"] == -100
        assert legacy[1]["byte_delta"] == -200


class TestAtomPoolToCathedralAutopilotCandidates:
    def test_produces_candidate_per_atom(self) -> None:
        atoms = [_sample_arbitrary("c1", -0.005, -0.001, cost=10.0)]
        candidates = atom_pool_to_cathedral_autopilot_candidates(atoms)
        assert len(candidates) == 1
        c = candidates[0]
        assert c["candidate_id"] == "c1"
        assert c["kind"] == "arbitrary_value"
        assert c["predicted_delta"] == pytest.approx(-0.003)
        assert c["predicted_delta_lower"] == -0.005
        assert c["predicted_delta_upper"] == -0.001
        assert c["cost_envelope_usd"] == 10.0
        assert c["evidence_grade"] == "predicted"
        assert "atom_metadata" in c
        assert "atom_provenance" in c

    def test_kind_routing_hint_per_kind(self) -> None:
        a_arbitrary = _sample_arbitrary("ar", -0.001, -0.0001)
        candidates = atom_pool_to_cathedral_autopilot_candidates([a_arbitrary])
        assert candidates[0]["kind_routing_hint"] == "rank_by_ev_per_dollar"

        a_probe = build_probe_outcome_atom(
            atom_id="probe:x", probe_id="x", substrate="y", verdict="INDEPENDENT"
        )
        candidates = atom_pool_to_cathedral_autopilot_candidates([a_probe])
        assert "predecessor" in candidates[0]["kind_routing_hint"]

    def test_carries_wired_hooks_and_observability_surface(self) -> None:
        atoms = [_sample_arbitrary("c2", -0.01, 0.0)]
        candidates = atom_pool_to_cathedral_autopilot_candidates(atoms)
        c = candidates[0]
        assert "cathedral_autopilot_dispatch" in c["wired_hooks"]
        assert "cite_able" in c["observability_surface"]
