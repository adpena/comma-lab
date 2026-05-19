# SPDX-License-Identifier: MIT
"""Tests for the seven canonical builders."""
from __future__ import annotations

import pytest

from tac.atom.builders import (
    build_arbitrary_value_atom,
    build_cargo_cult_atom,
    build_council_deliberation_atom,
    build_dispatch_claim_atom,
    build_meta_lagrangian_atom,
    build_premise_verification_atom,
    build_probe_outcome_atom,
)
from tac.atom.types import AtomKind, AtomValidationError, ResolutionPath


class TestArbitraryValueBuilder:
    def test_happy_path(self) -> None:
        atom = build_arbitrary_value_atom(
            atom_id="lr_5e-4",
            file_path="experiments/train_substrate_*.py",
            current_value="5e-4",
            predicted_replacement="LR finder",
            resolution_path=ResolutionPath.EXPERIMENTAL,
            predicted_ev_delta_s=(-0.008, -0.002),
            cost_envelope_usd=15.0,
            literature_citation="Smith 2017",
        )
        assert atom.kind == AtomKind.ARBITRARY_VALUE
        assert atom.predicted_impact_delta_s_lower == -0.008
        assert atom.metadata["file_path"] == "experiments/train_substrate_*.py"
        assert atom.metadata["current_value"] == "5e-4"

    def test_rejects_malformed_predicted_band(self) -> None:
        with pytest.raises(AtomValidationError, match="predicted_ev_delta_s"):
            build_arbitrary_value_atom(
                atom_id="x",
                file_path="f",
                current_value=1,
                predicted_replacement=2,
                resolution_path=ResolutionPath.EXPERIMENTAL,
                predicted_ev_delta_s=(1, 2, 3),
                cost_envelope_usd=0.0,
                literature_citation="cit",
            )


class TestMetaLagrangianBuilder:
    def test_happy_path(self) -> None:
        atom = build_meta_lagrangian_atom(
            atom_id="hnerv:recode:v1",
            family="hnerv_decoder_rate_recode",
            family_group="hnerv_rate_equivalent_recode",
            byte_delta=-832,
        )
        assert atom.kind == AtomKind.META_LAGRANGIAN
        assert atom.resolution_path == ResolutionPath.ANALYTICAL_SOLVE
        assert atom.metadata["byte_delta"] == -832
        # Default citation present
        assert atom.literature_citation

    def test_source_archive_sha_pushed_to_provenance_if_absent(self) -> None:
        atom = build_meta_lagrangian_atom(
            atom_id="x",
            family="f",
            family_group="g",
            byte_delta=0,
            source_archive_sha256="abc123" * 11,  # 66 chars; provenance validator may reject, but builder still propagates
        )
        assert atom.provenance.get("source_sha256") == "abc123" * 11


class TestCargoCultBuilder:
    def test_happy_path_hard_earned(self) -> None:
        atom = build_cargo_cult_atom(
            atom_id="cc:nscs06:chroma",
            substrate_id="nscs06",
            assumption="chroma must be preserved",
            classification="HARD-EARNED",
            rationale="empirical 105.15 -> 58.89 v6->v7",
        )
        assert atom.kind == AtomKind.CARGO_CULT_ASSUMPTION
        assert atom.metadata["classification"] == "HARD-EARNED"

    def test_rejects_invalid_classification(self) -> None:
        with pytest.raises(AtomValidationError, match="classification"):
            build_cargo_cult_atom(
                atom_id="cc:x",
                substrate_id="x",
                assumption="y",
                classification="BOGUS",
                rationale="z",
            )


class TestPremiseVerificationBuilder:
    def test_happy_path_empty_citation_allowed(self) -> None:
        atom = build_premise_verification_atom(
            atom_id="pv:tac.foo.imports",
            premise="tac.foo importable",
            verified=True,
            verification_method="importlib.import_module",
        )
        assert atom.kind == AtomKind.PREMISE_VERIFICATION
        assert atom.literature_citation == ""
        assert atom.metadata["verified"] is True
        assert atom.cost_envelope_usd == 0.0

    def test_with_callsite_metadata(self) -> None:
        atom = build_premise_verification_atom(
            atom_id="pv:test",
            premise="p",
            verified=False,
            verification_method="grep",
            callsite_path="src/tac/foo.py",
            callsite_line=42,
        )
        assert atom.metadata["callsite_path"] == "src/tac/foo.py"
        assert atom.metadata["callsite_line"] == 42


class TestProbeOutcomeBuilder:
    def test_happy_path_independent_verdict(self) -> None:
        atom = build_probe_outcome_atom(
            atom_id="probe:atw_v2_d4_independent",
            probe_id="atw_v2_d4_independent",
            substrate="atw_v2",
            verdict="INDEPENDENT",
        )
        assert atom.kind == AtomKind.PROBE_OUTCOME
        assert atom.metadata["verdict"] == "INDEPENDENT"

    def test_rejects_unknown_verdict(self) -> None:
        with pytest.raises(AtomValidationError, match="verdict"):
            build_probe_outcome_atom(
                atom_id="probe:x",
                probe_id="x",
                substrate="y",
                verdict="FAKE_VERDICT",
            )


class TestCouncilDeliberationBuilder:
    def test_happy_path_t2_tier(self) -> None:
        atom = build_council_deliberation_atom(
            atom_id="council:test_d",
            deliberation_id="test_d",
            topic="example",
            council_tier="T2",
            council_verdict="PROCEED",
        )
        assert atom.kind == AtomKind.COUNCIL_DELIBERATION
        assert atom.metadata["council_tier"] == "T2"
        assert atom.metadata["council_verdict"] == "PROCEED"

    def test_rejects_invalid_tier(self) -> None:
        with pytest.raises(AtomValidationError, match="council_tier"):
            build_council_deliberation_atom(
                atom_id="x",
                deliberation_id="x",
                topic="t",
                council_tier="T9",
                council_verdict="PROCEED",
            )


class TestDispatchClaimBuilder:
    def test_happy_path(self) -> None:
        atom = build_dispatch_claim_atom(
            atom_id="dispatch:lane_x:fc-abc",
            lane_id="lane_x",
            provider="modal",
            gpu="A100",
            instance_or_job_id="fc-abc",
            cost_envelope_usd=2.50,
        )
        assert atom.kind == AtomKind.DISPATCH_CLAIM
        assert atom.metadata["lane_id"] == "lane_x"
        assert atom.metadata["status"] == "active"

    def test_default_opened_at_utc_populated(self) -> None:
        atom = build_dispatch_claim_atom(
            atom_id="dispatch:x:y",
            lane_id="x",
            provider="vastai",
            gpu="4090",
            instance_or_job_id="y",
            cost_envelope_usd=1.0,
        )
        # opened_at_utc auto-populated to now
        assert atom.metadata["opened_at_utc"]
