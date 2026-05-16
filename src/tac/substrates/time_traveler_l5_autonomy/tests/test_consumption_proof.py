# SPDX-License-Identifier: MIT
"""Tests for the TT5L temporal side-info consumption proof artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tac.optimization.l5_staircase_v2 import (
    TT5L_SIDEINFO_CONSUMPTION_PREDICATE_ID,
    TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH,
    TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_SHA256,
    L5V2GateEvidence,
    l5_v2_canonical_sideinfo_gate_evidence,
    l5_v2_dispatch_readiness,
)
from tac.substrates.time_traveler_l5_autonomy.consumption_proof import (
    TT5L_SIDEINFO_CONSUMPTION_GATE_ID,
    build_tt5l_sideinfo_consumption_proof,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tt5l_sideinfo_consumption_proof_builder_is_deterministic(
    tmp_path: Path,
) -> None:
    proof_path = tmp_path / "proof.json"
    manifest_path = tmp_path / "manifest.json"
    work_dir = tmp_path / "work"

    first = build_tt5l_sideinfo_consumption_proof(
        artifact_path=proof_path,
        manifest_path=manifest_path,
        work_dir=work_dir,
        repo_root=Path.cwd(),
    )
    first_sha = _sha256_file(first.proof_path)
    second = build_tt5l_sideinfo_consumption_proof(
        artifact_path=proof_path,
        manifest_path=manifest_path,
        work_dir=work_dir,
        repo_root=Path.cwd(),
    )

    assert _sha256_file(second.proof_path) == first_sha
    proof = second.proof
    assert proof["predicate_passed"] is True
    assert proof["score_claim"] is False
    assert proof["promotion_eligible"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False
    assert proof["byte_mutation_proof"]["parser_consumed_bytes"] is True
    assert proof["byte_mutation_proof"]["output_changed"] is True
    assert proof["component_proofs"]["ac_state"]["parser_consumed_bytes"] is True
    assert proof["component_proofs"]["ac_state"]["output_changed"] is True
    assert "not_real_range_or_ans_entropy_decoder" in proof["ac_state_status"]


def test_committed_tt5l_sideinfo_consumption_proof_satisfies_l5_gate() -> None:
    proof_path = Path(TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert _sha256_file(proof_path) == TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_SHA256

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=[
            L5V2GateEvidence(
                gate_id=TT5L_SIDEINFO_CONSUMPTION_GATE_ID,
                artifact_path=TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH,
                artifact_sha256=TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_SHA256,
                predicate_id=proof["predicate_id"],
                predicate_passed=True,
                evidence_grade="local_no_gpu_parser_and_inflate_consumption_proof",
            )
        ]
    )
    sideinfo_gate = next(
        gate for gate in readiness["gates"]
        if gate["gate_id"] == TT5L_SIDEINFO_CONSUMPTION_GATE_ID
    )
    assert sideinfo_gate["evidence_valid"] is True
    assert all(
        "byte_closed_temporal_sideinfo_consumption" not in blocker
        for blocker in readiness["blockers"]
    )
    assert readiness["ready_for_dispatch"] is False


def test_l5_v2_canonical_sideinfo_gate_evidence_is_sha_bound() -> None:
    evidence = l5_v2_canonical_sideinfo_gate_evidence()

    assert evidence is not None
    assert evidence.gate_id == TT5L_SIDEINFO_CONSUMPTION_GATE_ID
    assert evidence.artifact_path == TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH
    assert evidence.artifact_sha256 == TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_SHA256
    assert evidence.predicate_id == TT5L_SIDEINFO_CONSUMPTION_PREDICATE_ID
    assert evidence.predicate_passed is True
