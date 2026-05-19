# SPDX-License-Identifier: MIT
"""Tests for the subsumption helpers — read existing sources, emit canonical Atoms."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.atom.subsumption import (
    atom_from_arbitrariness_audit_row,
    atom_from_cargo_cult_audit_row,
    atom_from_council_deliberation_record,
    atom_from_dispatch_claim_row,
    atom_from_meta_lagrangian_row,
    atom_from_probe_outcomes_ledger_row,
)
from tac.atom.types import AtomKind, AtomValidationError, ResolutionPath

REPO_ROOT = Path(__file__).resolve().parents[4]
SISTER_AUDIT_PATH = REPO_ROOT / ".omx" / "state" / "arbitrariness_extinction_audit_20260518.jsonl"


class TestMetaLagrangianSubsumption:
    def test_accepts_canonical_allocator_row(self) -> None:
        row = {
            "atom_id": "hnerv:decoder_recode:v1",
            "family": "hnerv_decoder_rate_recode",
            "family_group": "hnerv_rate_equivalent_recode",
            "byte_delta": -832,
            "expected_seg_dist_delta": 0.0,
            "expected_pose_dist_delta": 0.0,
            "confidence": 1.0,
            "evidence_grade": "empirical_byte_raw_equal",
            "source_archive_sha256": "ab" * 32,  # 64-char hex
        }
        atom = atom_from_meta_lagrangian_row(row)
        assert atom.kind == AtomKind.META_LAGRANGIAN
        assert atom.metadata["byte_delta"] == -832

    def test_rejects_missing_atom_id(self) -> None:
        with pytest.raises(AtomValidationError, match="atom_id"):
            atom_from_meta_lagrangian_row({"family": "x"})


class TestCargoCultSubsumption:
    def test_accepts_canonical_audit_row(self) -> None:
        row = {
            "substrate_id": "nscs06",
            "assumption": "chroma must be preserved",
            "classification": "HARD-EARNED",
            "rationale": "empirical v6->v7 +44%",
            "unwind_test_plan": "ablate chroma",
            "predicted_impact": [-0.030, -0.005],
        }
        atom = atom_from_cargo_cult_audit_row(row)
        assert atom.kind == AtomKind.CARGO_CULT_ASSUMPTION
        assert atom.metadata["substrate_id"] == "nscs06"

    def test_accepts_substrate_alias(self) -> None:
        row = {"substrate": "z6", "assumption": "x", "classification": "CARGO-CULTED", "rationale": "y"}
        atom = atom_from_cargo_cult_audit_row(row)
        assert atom.metadata["substrate_id"] == "z6"


class TestProbeOutcomeSubsumption:
    def test_accepts_canonical_ledger_row(self) -> None:
        row = {
            "probe_id": "atw_v2_d4_h_latent_given_scorer",
            "substrate": "atw_v2",
            "verdict": "INDEPENDENT",
            "metric_name": "mutual_information_bits",
            "metric_value": 0.006385,
            "threshold": 0.5,
            "blocker_status": "blocking",
        }
        atom = atom_from_probe_outcomes_ledger_row(row)
        assert atom.kind == AtomKind.PROBE_OUTCOME
        assert atom.metadata["verdict"] == "INDEPENDENT"

    def test_rejects_missing_probe_id(self) -> None:
        with pytest.raises(AtomValidationError, match="probe_id"):
            atom_from_probe_outcomes_ledger_row({"verdict": "INDEPENDENT"})


class TestCouncilDeliberationSubsumption:
    def test_accepts_canonical_record(self) -> None:
        record = {
            "deliberation_id": "council_z7_phase_2",
            "topic": "z7 mamba2 phase 2 design",
            "council_tier": "T3",
            "council_verdict": "PROCEED",
            "council_attendees": ["Shannon", "Dykstra"],
            "council_quorum_met": True,
        }
        atom = atom_from_council_deliberation_record(record)
        assert atom.kind == AtomKind.COUNCIL_DELIBERATION
        assert atom.metadata["council_tier"] == "T3"


class TestDispatchClaimSubsumption:
    def test_accepts_canonical_row(self) -> None:
        row = {
            "lane_id": "lane_z7_smoke",
            "provider": "modal",
            "gpu": "A100",
            "instance_or_job_id": "fc-01KRW",
            "status": "active",
            "cost_envelope_usd": 2.50,
        }
        atom = atom_from_dispatch_claim_row(row)
        assert atom.kind == AtomKind.DISPATCH_CLAIM
        assert atom.metadata["lane_id"] == "lane_z7_smoke"

    def test_accepts_instance_id_alias(self) -> None:
        row = {"lane_id": "lane_x", "instance_id": "fc-abc", "provider": "modal", "gpu": "A100"}
        atom = atom_from_dispatch_claim_row(row)
        assert atom.atom_id == "dispatch:lane_x:fc-abc"


class TestArbitrarinessAuditSubsumption:
    def test_accepts_canonical_sister_audit_row(self) -> None:
        if not SISTER_AUDIT_PATH.is_file():
            pytest.skip(f"sister audit JSONL not present at {SISTER_AUDIT_PATH}")
        with SISTER_AUDIT_PATH.open("r") as f:
            first = json.loads(f.readline())
        atom = atom_from_arbitrariness_audit_row(first)
        assert atom.kind == AtomKind.ARBITRARY_VALUE
        assert atom.atom_id == first["value_id"]
        # Predicted band correctly mapped
        assert atom.predicted_impact_delta_s_lower == first["predicted_ev_delta_s"][0]

    def test_subsumes_all_sister_audit_rows(self) -> None:
        """End-to-end: every sister audit row converts to a canonical Atom."""
        if not SISTER_AUDIT_PATH.is_file():
            pytest.skip(f"sister audit JSONL not present at {SISTER_AUDIT_PATH}")
        atoms = []
        with SISTER_AUDIT_PATH.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                atoms.append(atom_from_arbitrariness_audit_row(row))
        # 52 sister-audit rows per landing snapshot
        assert len(atoms) >= 52
        # All canonical kind
        assert all(a.kind == AtomKind.ARBITRARY_VALUE for a in atoms)
        # All canonical resolution paths in the 6-member set
        valid_paths = {m.value for m in ResolutionPath}
        for a in atoms:
            assert a.resolution_path.value in valid_paths

    def test_rejects_unknown_resolution_path(self) -> None:
        row = {
            "value_id": "x",
            "file_path": "f",
            "current_value": 1,
            "predicted_replacement": 2,
            "resolution_path": "totally_made_up_path",
            "predicted_ev_delta_s": [-0.01, -0.001],
            "cost_envelope_usd": 1.0,
            "literature_citation": "cit",
        }
        with pytest.raises(AtomValidationError, match="resolution_path"):
            atom_from_arbitrariness_audit_row(row)

    def test_handles_contest_fixed_resolution_path(self) -> None:
        row = {
            "value_id": "rate_denom",
            "file_path": "upstream/evaluate.py",
            "current_value": 37545489,
            "predicted_replacement": 37545489,
            "resolution_path": "contest_fixed",
            "predicted_ev_delta_s": [0.0, 0.0],
            "cost_envelope_usd": 0.0,
            "literature_citation": "upstream contest spec",
        }
        atom = atom_from_arbitrariness_audit_row(row)
        assert atom.resolution_path == ResolutionPath.CONTEST_FIXED
