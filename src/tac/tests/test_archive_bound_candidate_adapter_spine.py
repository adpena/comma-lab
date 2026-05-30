# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.archive_bound_candidate_adapter_spine import (
    ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_EXACT_BLOCKER_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_MLX_TRIAGE_REQUEST_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_POSTERIOR_HOOK_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_RECEIVER_PROOF_GATE_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_REPLAY_BUNDLE_SCHEMA,
    build_archive_bound_candidate_adapter_package,
)
from tac.optimization.cross_family_candidate_portfolio import (
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
                "runtime_consumption_proof_path": proof.relative_to(
                    self.root
                ).as_posix(),
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

    assert package["deterministic_replay_bundles"][0]["schema"] == (
        ARCHIVE_BOUND_CANDIDATE_REPLAY_BUNDLE_SCHEMA
    )
    assert package["deterministic_replay_bundles"][0]["replay_bundle_ready"] is True
    assert package["mlx_triage_requests"][0]["schema"] == (
        ARCHIVE_BOUND_CANDIDATE_MLX_TRIAGE_REQUEST_SCHEMA
    )
    assert package["receiver_proof_gates"][0]["schema"] == (
        ARCHIVE_BOUND_CANDIDATE_RECEIVER_PROOF_GATE_SCHEMA
    )
    assert package["exact_axis_blockers"][0]["schema"] == (
        ARCHIVE_BOUND_CANDIDATE_EXACT_BLOCKER_SCHEMA
    )
    assert "contest_cpu_or_cuda_authority_required" in package[
        "exact_axis_blockers"
    ][0]["blockers"]
    assert package["posterior_update_hooks"][0]["schema"] == (
        ARCHIVE_BOUND_CANDIDATE_POSTERIOR_HOOK_SCHEMA
    )


def test_portfolio_consumes_adapter_package_contract_surface(tmp_path: Path) -> None:
    package = build_archive_bound_candidate_adapter_package(
        _FixtureArchiveAdapter(tmp_path),
        repo_root=tmp_path,
    )

    portfolio = build_cross_family_candidate_portfolio(
        incumbent_score=0.2,
        archive_contract_surfaces=[package],
    )

    assert portfolio["portfolio_summary"]["source_counts"] == {
        "archive_bound_candidate_contract": 1
    }
    row = portfolio["ranked_rows"][0]
    assert row["source_kind"] == "archive_bound_candidate_contract"
    assert row["family_id"] == "range_coder"
    assert row["operator_next_action"] == (
        "promote_archive_contract_to_receiver_exact_bridge"
    )
    assert row["ready_for_exact_eval_dispatch"] is False
