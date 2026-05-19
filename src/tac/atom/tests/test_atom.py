# SPDX-License-Identifier: MIT
"""Tests for the ``tac.atom.Atom`` frozen dataclass."""
from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from tac.atom.atom import (
    ATOM_SCHEMA_VERSION,
    OBSERVABILITY_FACETS_CANONICAL,
    WIRED_HOOKS_CANONICAL,
    Atom,
)
from tac.atom.types import AtomKind, AtomValidationError, ResolutionPath


def _minimal_provenance() -> dict:
    return {
        "artifact_kind": "predicted_from_model",
        "evidence_grade": "predicted",
        "captured_at_utc": "2026-05-19T00:00:00Z",
    }


def _make_atom(**overrides) -> Atom:
    defaults: dict = dict(
        atom_id="test_atom",
        kind=AtomKind.ARBITRARY_VALUE,
        resolution_path=ResolutionPath.EXPERIMENTAL,
        predicted_impact_delta_s_lower=-0.005,
        predicted_impact_delta_s_upper=-0.001,
        cost_envelope_usd=10.0,
        provenance=_minimal_provenance(),
        wired_hooks=["cathedral_autopilot_dispatch"],
        observability_surface=["cite_able"],
        literature_citation="test citation",
        canonical_helper_repo_link="src/tac/atom/atom.py",
        metadata={},
    )
    defaults.update(overrides)
    return Atom(**defaults)


class TestAtomConstruction:
    def test_minimal_construction_passes(self) -> None:
        atom = _make_atom()
        assert atom.atom_id == "test_atom"
        assert atom.schema == ATOM_SCHEMA_VERSION

    def test_frozen_no_attribute_mutation(self) -> None:
        atom = _make_atom()
        with pytest.raises((AttributeError, Exception)):
            atom.atom_id = "mutated"  # type: ignore[misc]

    def test_replace_re_runs_validation(self) -> None:
        atom = _make_atom()
        with pytest.raises(AtomValidationError, match="lower"):
            replace(atom, predicted_impact_delta_s_upper=-0.99)


class TestPerFieldInvariants:
    def test_empty_atom_id_rejected(self) -> None:
        with pytest.raises(AtomValidationError, match="atom_id"):
            _make_atom(atom_id="")

    def test_newline_in_atom_id_rejected(self) -> None:
        with pytest.raises(AtomValidationError, match="newlines"):
            _make_atom(atom_id="foo\nbar")

    def test_lower_gt_upper_rejected(self) -> None:
        with pytest.raises(AtomValidationError, match="lower"):
            _make_atom(
                predicted_impact_delta_s_lower=0.5,
                predicted_impact_delta_s_upper=0.1,
            )

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(AtomValidationError, match="cost_envelope"):
            _make_atom(cost_envelope_usd=-1.0)

    def test_wrong_kind_type_rejected(self) -> None:
        with pytest.raises(AtomValidationError, match="kind"):
            _make_atom(kind="not_an_enum")

    def test_wrong_resolution_path_type_rejected(self) -> None:
        with pytest.raises(AtomValidationError, match="resolution_path"):
            _make_atom(resolution_path="not_an_enum")

    def test_provenance_missing_keys_rejected(self) -> None:
        with pytest.raises(AtomValidationError, match="provenance"):
            _make_atom(provenance={"artifact_kind": "x"})  # missing 2 of 3

    def test_empty_wired_hooks_rejected(self) -> None:
        with pytest.raises(AtomValidationError, match="wired_hooks"):
            _make_atom(wired_hooks=[])

    def test_non_canonical_hook_rejected(self) -> None:
        with pytest.raises(AtomValidationError, match="wired_hooks"):
            _make_atom(wired_hooks=["fake_hook"])

    def test_empty_observability_surface_rejected(self) -> None:
        with pytest.raises(AtomValidationError, match="observability_surface"):
            _make_atom(observability_surface=[])

    def test_non_canonical_facet_rejected(self) -> None:
        with pytest.raises(AtomValidationError, match="observability_surface"):
            _make_atom(observability_surface=["fake_facet"])

    def test_empty_citation_rejected_for_non_premise_kind(self) -> None:
        with pytest.raises(AtomValidationError, match="literature_citation"):
            _make_atom(literature_citation="")

    def test_empty_citation_accepted_for_premise_kind(self) -> None:
        atom = _make_atom(
            kind=AtomKind.PREMISE_VERIFICATION,
            literature_citation="",
            metadata={
                "premise": "x",
                "verified": True,
                "verification_method": "importlib",
            },
        )
        assert atom.literature_citation == ""


class TestPerKindInvariants:
    def test_probe_outcome_requires_verdict(self) -> None:
        with pytest.raises(AtomValidationError, match="verdict"):
            _make_atom(kind=AtomKind.PROBE_OUTCOME, metadata={})

    def test_probe_outcome_rejects_unknown_verdict(self) -> None:
        with pytest.raises(AtomValidationError, match="verdict"):
            _make_atom(kind=AtomKind.PROBE_OUTCOME, metadata={"verdict": "FAKE"})

    def test_probe_outcome_accepts_canonical_verdict(self) -> None:
        atom = _make_atom(
            kind=AtomKind.PROBE_OUTCOME,
            metadata={"verdict": "INDEPENDENT"},
        )
        assert atom.metadata["verdict"] == "INDEPENDENT"

    def test_council_deliberation_requires_tier(self) -> None:
        with pytest.raises(AtomValidationError, match="council_tier"):
            _make_atom(kind=AtomKind.COUNCIL_DELIBERATION, metadata={})

    def test_council_deliberation_rejects_unknown_tier(self) -> None:
        with pytest.raises(AtomValidationError, match="council_tier"):
            _make_atom(
                kind=AtomKind.COUNCIL_DELIBERATION,
                metadata={"council_tier": "T9"},
            )

    def test_council_deliberation_accepts_T2(self) -> None:
        atom = _make_atom(
            kind=AtomKind.COUNCIL_DELIBERATION,
            metadata={"council_tier": "T2"},
        )
        assert atom.metadata["council_tier"] == "T2"

    def test_cargo_cult_requires_classification(self) -> None:
        with pytest.raises(AtomValidationError, match="classification"):
            _make_atom(kind=AtomKind.CARGO_CULT_ASSUMPTION, metadata={})

    def test_cargo_cult_rejects_unknown_classification(self) -> None:
        with pytest.raises(AtomValidationError, match="classification"):
            _make_atom(
                kind=AtomKind.CARGO_CULT_ASSUMPTION,
                metadata={"classification": "BOGUS"},
            )

    def test_cargo_cult_accepts_hard_earned(self) -> None:
        atom = _make_atom(
            kind=AtomKind.CARGO_CULT_ASSUMPTION,
            metadata={"classification": "HARD-EARNED"},
        )
        assert atom.metadata["classification"] == "HARD-EARNED"


class TestSerialization:
    def test_to_jsonl_row_basic_shape(self) -> None:
        atom = _make_atom()
        row = atom.to_jsonl_row()
        assert row["atom_id"] == "test_atom"
        assert row["kind"] == "arbitrary_value"
        assert row["resolution_path"] == "experimental"
        assert row["schema"] == ATOM_SCHEMA_VERSION

    def test_to_jsonl_row_serializes_enum_as_str(self) -> None:
        atom = _make_atom(kind=AtomKind.META_LAGRANGIAN)
        row = atom.to_jsonl_row()
        assert isinstance(row["kind"], str)
        assert row["kind"] == "meta_lagrangian"

    def test_to_jsonl_row_serializes_sequences_as_lists(self) -> None:
        atom = _make_atom()
        row = atom.to_jsonl_row()
        assert isinstance(row["wired_hooks"], list)
        assert isinstance(row["observability_surface"], list)

    def test_asdict_roundtrip_preserves_fields(self) -> None:
        atom = _make_atom()
        d = asdict(atom)
        assert d["atom_id"] == "test_atom"
        assert d["schema"] == ATOM_SCHEMA_VERSION


class TestToMetaLagrangianAtom:
    def test_round_trips_legacy_shape(self) -> None:
        atom = _make_atom(kind=AtomKind.META_LAGRANGIAN, metadata={"byte_delta": -512})
        legacy = atom.to_meta_lagrangian_atom()
        assert legacy["atom_id"] == "test_atom"
        assert legacy["byte_delta"] == -512
        assert legacy["family"].startswith("atom_kind:")
        assert legacy["family_group"].startswith("resolution_path:")

    def test_midpoint_score_delta_correct(self) -> None:
        atom = _make_atom(
            predicted_impact_delta_s_lower=-0.010,
            predicted_impact_delta_s_upper=-0.002,
        )
        legacy = atom.to_meta_lagrangian_atom()
        assert legacy["expected_score_delta"] == pytest.approx(-0.006)
        assert legacy["expected_score_delta_lower"] == pytest.approx(-0.010)
        assert legacy["expected_score_delta_upper"] == pytest.approx(-0.002)

    def test_carries_evidence_grade_from_provenance(self) -> None:
        atom = _make_atom()
        legacy = atom.to_meta_lagrangian_atom()
        assert legacy["evidence_grade"] == "predicted"


class TestCanonicalConstants:
    def test_wired_hooks_canonical_six(self) -> None:
        assert len(WIRED_HOOKS_CANONICAL) == 6
        assert "cathedral_autopilot_dispatch" in WIRED_HOOKS_CANONICAL

    def test_observability_facets_canonical_six(self) -> None:
        assert len(OBSERVABILITY_FACETS_CANONICAL) == 6
        assert "cite_able" in OBSERVABILITY_FACETS_CANONICAL


class TestValidateMethod:
    def test_validate_passes_on_clean_atom(self) -> None:
        atom = _make_atom()
        # Should not raise
        atom.validate()
