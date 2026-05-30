# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tac.optimization.archive_bound_candidate_adapter_spine import (
    ARCHIVE_BOUND_CANDIDATE_EXACT_BLOCKER_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_MLX_TRIAGE_REQUEST_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_POSTERIOR_HOOK_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_RECEIVER_PROOF_GATE_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_REPLAY_BUNDLE_SCHEMA,
    build_archive_bound_candidate_adapter_package,
)
from tac.optimization.archive_bound_candidate_contract import (
    ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA,
    ArchiveBoundCandidateContractError,
    archive_bound_candidate_contracts_from_payload,
)
from tac.optimization.cross_family_candidate_portfolio import (
    CrossFamilyCandidatePortfolioError,
    build_cross_family_candidate_portfolio,
)


def _record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class _FixtureArchiveAdapter:
    adapter_id = "fixture_range_adapter"
    candidate_family = "range_coder"

    def __init__(self, root: Path) -> None:
        self.root = root

    def emit_archive_bound_candidate_rows(
        self,
        context: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        source = self.root / "source.zip"
        candidate = self.root / "candidate.zip"
        proof = self.root / "receiver_proof.json"
        source.write_bytes(b"S" * 128)
        candidate.write_bytes(b"C" * 80)
        proof.write_text('{"schema":"fixture_receiver_proof.v1"}\n', encoding="utf-8")
        return [
            {
                "candidate_id": "fixture_range_candidate",
                "target_kind": "range_coder_entropy_recode_v1",
                "candidate_archive": _record(self.root, candidate),
                "source_archive": _record(self.root, source),
                "byte_closed_candidate_emitted": True,
                "runtime_consumption_proof_status": "present",
                "runtime_consumption_proof_path": proof.relative_to(self.root).as_posix(),
                "receiver_contract_kind": "fixture_receiver_contract",
                "receiver_contract_satisfied": True,
                "runtime_adapter_ready": True,
                "score_affecting_payload_changed": True,
                "charged_bits_changed": True,
                "replay_argv": ["python", "-m", "fixture.replay"],
                "mlx_triage_argv": ["python", "-m", "fixture.mlx_probe"],
                "input_artifacts": ["source.zip"],
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
            }
        ]


class _AntiPatternFixtureArchiveAdapter(_FixtureArchiveAdapter):
    adapter_id = "fixture_range_adapter_with_anti_patterns"

    def emit_archive_bound_candidate_rows(
        self,
        context: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        row = dict(super().emit_archive_bound_candidate_rows(context)[0])
        row["archive_entropy_substrate_coverage"] = {
            "anti_pattern_protections": [
                {"anti_pattern_id": ("probe_only_side_report_orphaned_from_optimizer_v1")},
                {"anti_pattern_id": "entropy_coder_order_cargo_cult_v1"},
            ]
        }
        return [row]


class _IncompleteCustodyMlxArchiveAdapter(_FixtureArchiveAdapter):
    adapter_id = "fixture_range_adapter_with_incomplete_mlx_custody"

    def emit_archive_bound_candidate_rows(
        self,
        context: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        source = self.root / "source.zip"
        source.write_bytes(b"S" * 128)
        return [
            {
                "candidate_id": "fixture_range_candidate_no_custody",
                "target_kind": "range_coder_entropy_recode_v1",
                "source_archive": _record(self.root, source),
                "byte_closed_candidate_emitted": True,
                "runtime_consumption_proof_status": "missing",
                "receiver_contract_kind": "fixture_receiver_contract",
                "receiver_contract_satisfied": False,
                "runtime_adapter_ready": True,
                "score_affecting_payload_changed": True,
                "charged_bits_changed": True,
                "mlx_triage_argv": ["python", "-m", "fixture.mlx_probe"],
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
            }
        ]


def test_archive_bound_adapter_spine_emits_full_pipeline_package(
    tmp_path: Path,
) -> None:
    package = build_archive_bound_candidate_adapter_package(
        _FixtureArchiveAdapter(tmp_path),
        repo_root=tmp_path,
    )

    assert package["schema"] == ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA
    assert package["candidate_row_count"] == 1
    assert package["ready_contract_count"] == 1
    assert package["mlx_triage_ready_count"] == 1
    assert package["receiver_proof_gate_passed_count"] == 1

    row = package["candidate_rows"][0]
    contract = row["archive_bound_candidate_contract"]
    assert contract["family_id"] == "range_coder"
    assert contract["entropy_position_label"] == "at_entropy_coder"
    assert contract["archive_bound_candidate_ready_for_exact_handoff"] is True
    assert contract["ready_for_exact_eval_dispatch"] is False

    assert package["deterministic_replay_bundles"][0]["schema"] == (ARCHIVE_BOUND_CANDIDATE_REPLAY_BUNDLE_SCHEMA)
    assert package["deterministic_replay_bundles"][0]["replay_bundle_ready"] is True
    assert package["mlx_triage_requests"][0]["schema"] == (ARCHIVE_BOUND_CANDIDATE_MLX_TRIAGE_REQUEST_SCHEMA)
    assert package["mlx_triage_requests"][0]["blockers"] == []
    assert package["mlx_triage_requests"][0]["ready_for_mlx_local_triage"] is True
    assert package["receiver_proof_gates"][0]["schema"] == (ARCHIVE_BOUND_CANDIDATE_RECEIVER_PROOF_GATE_SCHEMA)
    assert package["exact_axis_blockers"][0]["schema"] == (ARCHIVE_BOUND_CANDIDATE_EXACT_BLOCKER_SCHEMA)
    assert "contest_cpu_or_cuda_authority_required" in package["exact_axis_blockers"][0]["blockers"]
    assert package["posterior_update_hooks"][0]["schema"] == (ARCHIVE_BOUND_CANDIDATE_POSTERIOR_HOOK_SCHEMA)
    extracted = archive_bound_candidate_contracts_from_payload(package)
    assert [contract["contract_key"] for contract in extracted] == [contract["contract_key"]]


def test_mlx_triage_fails_closed_on_incomplete_archive_custody(
    tmp_path: Path,
) -> None:
    package = build_archive_bound_candidate_adapter_package(
        _IncompleteCustodyMlxArchiveAdapter(tmp_path),
        repo_root=tmp_path,
    )

    assert package["ready_contract_count"] == 0
    assert package["mlx_triage_ready_count"] == 0
    request = package["mlx_triage_requests"][0]
    assert request["mlx_triage_argv"] == ["python", "-m", "fixture.mlx_probe"]
    assert request["ready_for_mlx_local_triage"] is False
    assert request["archive_file_custody_complete"] is False
    assert request["runtime_consumption_proof_ready"] is False
    assert request["receiver_contract_satisfied"] is False
    assert "archive_bound_candidate_not_ready_for_mlx_triage" in request["blockers"]
    assert "archive_bound_candidate_file_custody_incomplete" in request["blockers"]
    assert "receiver_runtime_proof_missing" in request["blockers"]
    assert "receiver_contract_not_satisfied" in request["blockers"]
    assert "archive_bound_candidate_path_missing" in request["contract_blockers"]


def test_portfolio_consumes_adapter_package_contract_surface(tmp_path: Path) -> None:
    package = build_archive_bound_candidate_adapter_package(
        _FixtureArchiveAdapter(tmp_path),
        repo_root=tmp_path,
    )

    portfolio = build_cross_family_candidate_portfolio(
        incumbent_score=0.2,
        archive_contract_surfaces=[package],
    )

    assert portfolio["portfolio_summary"]["source_counts"] == {"archive_bound_candidate_contract": 1}
    row = portfolio["ranked_rows"][0]
    assert row["source_kind"] == "archive_bound_candidate_contract"
    assert row["family_id"] == "range_coder"
    assert row["operator_next_action"] == ("promote_archive_contract_to_receiver_exact_bridge")
    assert row["ready_for_exact_eval_dispatch"] is False


def test_archive_contract_reader_rejects_stale_duplicate_fields(
    tmp_path: Path,
) -> None:
    package = build_archive_bound_candidate_adapter_package(
        _FixtureArchiveAdapter(tmp_path),
        repo_root=tmp_path,
    )
    row = dict(package["candidate_rows"][0])
    row["receiver_contract_satisfied"] = False

    with pytest.raises(
        ArchiveBoundCandidateContractError,
        match="archive_bound_contract_stale_duplicate_field:receiver_contract_satisfied",
    ):
        archive_bound_candidate_contracts_from_payload(row)

    stale_package = dict(package)
    stale_package["candidate_rows"] = [row]
    with pytest.raises(
        ArchiveBoundCandidateContractError,
        match="archive_bound_contract_stale_duplicate_field:receiver_contract_satisfied",
    ):
        archive_bound_candidate_contracts_from_payload(stale_package)

    with pytest.raises(
        CrossFamilyCandidatePortfolioError,
        match="archive_bound_contract_stale_duplicate_field:receiver_contract_satisfied",
    ):
        build_cross_family_candidate_portfolio(
            incumbent_score=0.2,
            archive_contract_surfaces=[row],
        )


def test_archive_contract_routes_anti_patterns_into_acquisition_penalty(
    tmp_path: Path,
) -> None:
    package = build_archive_bound_candidate_adapter_package(
        _AntiPatternFixtureArchiveAdapter(tmp_path),
        repo_root=tmp_path,
    )

    surface = package["archive_bound_candidate_contract_surfaces"][0]
    contract = surface["candidate_contracts"][0]

    assert contract["canonical_anti_pattern_ids"] == [
        "probe_only_side_report_orphaned_from_optimizer_v1",
        "entropy_coder_order_cargo_cult_v1",
    ]
    assert contract["anti_pattern_protection_count"] == 2
    assert contract["anti_pattern_acquisition_penalty"] > 0
    assert contract["acquisition_penalty"] >= contract["anti_pattern_acquisition_penalty"]
    assert surface["anti_pattern_acquisition_penalty_sum"] == contract["anti_pattern_acquisition_penalty"]

    portfolio = build_cross_family_candidate_portfolio(
        incumbent_score=0.2,
        archive_contract_surfaces=[package],
    )
    row = portfolio["ranked_rows"][0]
    metadata = row["source_metadata"]
    assert metadata["contract_acquisition_penalty_score"] > 0
    assert (
        metadata["archive_bound_candidate_contract"]["canonical_anti_pattern_ids"]
        == contract["canonical_anti_pattern_ids"]
    )
